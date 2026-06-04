# load packages
import os
import os.path as osp
import copy
import random
import yaml
import time
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import click
import shutil
import traceback
import warnings
warnings.simplefilter('ignore')
from torch.utils.tensorboard import SummaryWriter

from meldataset import build_dataloader

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from Utils.PLBERT.util import load_plbert

from models import *
from losses import *
from utils import *

from Modules.slmadv import SLMAdversarialLoss
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

from optimizers import build_optimizer

# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


def _scalar(x):
    """Convert tensor/number to plain float for logging."""
    if torch.is_tensor(x):
        return float(x.detach().mean().cpu().item())
    return float(x)


def _get_resume_batch_idx(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return int(checkpoint.get('batch_idx', 0) or 0)
    except Exception:
        return 0


def _cleanup_step_checkpoints(log_dir, prefix, save_total_limit):
    if save_total_limit is None or save_total_limit <= 0:
        return
    step_checkpoints = [
        osp.join(log_dir, f)
        for f in os.listdir(log_dir)
        if f.startswith(prefix) and f.endswith('.pth')
    ]
    if len(step_checkpoints) <= save_total_limit:
        return
    step_checkpoints = sorted(step_checkpoints, key=lambda x: osp.getmtime(x))
    for old_path in step_checkpoints[:-save_total_limit]:
        try:
            os.remove(old_path)
            print(f'Removed old step checkpoint: {old_path}')
        except OSError as e:
            print(f'Warning: failed to remove old step checkpoint {old_path}: {e}')


def _set_stage2_train_mode(model):
    """
    Restore the exact train/eval mode pattern used by the original Stage 2 loop.
    The original code keeps most modules in eval mode and trains selected modules.
    """
    _ = [model[key].eval() for key in model]
    model.predictor.train()
    model.bert_encoder.train()
    model.bert.train()
    model.msd.train()
    model.mpd.train()


def _safe_randint(rng, low, high):
    """np.random.randint fails when high <= low; return low in that case."""
    if high <= low:
        return low
    return int(rng.randint(low, high))


def _process_stage2_val_batch(
    batch,
    batch_idx,
    model,
    stft_loss,
    n_down,
    max_len,
    device,
    eval_seed,
):
    """
    Run one validation batch and return losses plus tensors needed for audio logging.
    This is intentionally based on the original train_second.py validation logic,
    but uses a local RandomState so the validation subset/crop is fixed across calls.
    """
    rng = np.random.RandomState(int(eval_seed) + int(batch_idx))

    waves = batch[0]
    batch = [b.to(device) for b in batch[1:]]
    texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch

    mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
    text_mask = length_to_mask(input_lengths).to(texts.device)

    _, _, s2s_attn = model.text_aligner(mels, mask, texts)
    s2s_attn = s2s_attn.transpose(-1, -2)
    s2s_attn = s2s_attn[..., 1:]
    s2s_attn = s2s_attn.transpose(-1, -2)

    mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
    s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

    # encode
    t_en = model.text_encoder(texts, input_lengths, text_mask)
    asr = (t_en @ s2s_attn_mono)
    d_gt = s2s_attn_mono.sum(axis=-1).detach()

    ss = []
    gs = []
    for bib in range(len(mel_input_length)):
        mel = mels[bib, :, :mel_input_length[bib]]
        s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
        ss.append(s)
        s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
        gs.append(s)

    s = torch.stack(ss).squeeze()
    gs = torch.stack(gs).squeeze()

    bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    d, p = model.predictor(d_en, s,
                           input_lengths,
                           s2s_attn_mono,
                           text_mask)

    # get clips
    mel_len = int(mel_input_length.min().item() / 2 - 1)
    if max_len is not None and max_len > 0:
        mel_len = min(mel_len, int(max_len))
    if mel_len <= 1:
        return None

    en = []
    gt = []
    p_en = []
    wav = []

    for bib in range(len(mel_input_length)):
        mel_length = int(mel_input_length[bib].item() / 2)
        random_start = _safe_randint(rng, 0, mel_length - mel_len)
        en.append(asr[bib, :, random_start:random_start + mel_len])
        p_en.append(p[bib, :, random_start:random_start + mel_len])
        gt.append(mels[bib, :, (random_start * 2):((random_start + mel_len) * 2)])

        y = waves[bib][(random_start * 2) * 300:((random_start + mel_len) * 2) * 300]
        wav.append(torch.from_numpy(y).to(device))

    wav = torch.stack(wav).float().detach()
    en = torch.stack(en)
    p_en = torch.stack(p_en)
    gt = torch.stack(gt).detach()

    s = model.predictor_encoder(gt.unsqueeze(1))
    F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s)

    loss_dur = 0
    for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
        _s2s_pred = _s2s_pred[:_text_length, :]
        _text_input = _text_input[:_text_length].long()
        _s2s_trg = torch.zeros_like(_s2s_pred)
        for bib in range(_s2s_trg.shape[0]):
            _s2s_trg[bib, :_text_input[bib]] = 1
        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
        loss_dur += F.l1_loss(_dur_pred[1:_text_length - 1],
                              _text_input[1:_text_length - 1])
    loss_dur /= texts.size(0)

    s_style = model.style_encoder(gt.unsqueeze(1))
    y_rec = model.decoder(en, F0_fake, N_fake, s_style)
    loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

    F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
    loss_F0 = F.l1_loss(F0_real, F0_fake) / 10

    return {
        'loss_mel': loss_mel,
        'loss_dur': loss_dur,
        'loss_F0': loss_F0,
        'waves': waves,
        'texts': texts,
        'input_lengths': input_lengths,
        'ref_texts': ref_texts,
        'ref_lengths': ref_lengths,
        'mels': mels,
        'mel_input_length': mel_input_length,
        'ref_mels': ref_mels,
        'text_mask': text_mask,
        'asr': asr,
        't_en': t_en,
        'bert_dur': bert_dur,
        'd_en': d_en,
        'p': p,
    }


def _log_stage2_audio_samples(
    writer,
    state,
    model,
    sampler,
    epoch,
    step,
    sr,
    device,
    multispeaker,
    diff_epoch,
    joint_epoch,
    max_audios,
    audio_prefix,
    sample_seed,
    log_gt=True,
):
    """
    Log 3-5 fixed validation examples to TensorBoard.
    Before joint_epoch: log reconstruction-style audio like original train_second.py.
    After joint_epoch: log sampled speech from text directly like original train_second.py.
    """
    if state is None:
        return

    waves = state['waves']
    mels = state['mels']
    mel_input_length = state['mel_input_length']
    ref_mels = state['ref_mels']
    input_lengths = state['input_lengths']
    text_mask = state['text_mask']
    asr = state['asr']
    t_en = state['t_en']
    bert_dur = state['bert_dur']
    d_en = state['d_en']
    p = state['p']

    max_audios = max(1, int(max_audios))
    tag_step = int(step)

    # Keep sampled speech deterministic across eval calls.
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(int(sample_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(sample_seed))

        if epoch < joint_epoch:
            # generating reconstruction examples with GT duration
            for bib in range(len(asr)):
                mel_length = int(mel_input_length[bib].item())
                gt = mels[bib, :, :mel_length].unsqueeze(0)
                en = asr[bib, :, :mel_length // 2].unsqueeze(0)

                F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
                F0_real = F0_real.unsqueeze(0)
                s = model.style_encoder(gt.unsqueeze(1))
                real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)

                y_rec = model.decoder(en, F0_real, real_norm, s)
                writer.add_audio(f'{audio_prefix}/eval_y{bib}', y_rec.cpu().numpy().squeeze(), tag_step, sample_rate=sr)

                s_dur = model.predictor_encoder(gt.unsqueeze(1))
                p_en = p[bib, :, :mel_length // 2].unsqueeze(0)
                F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)
                y_pred = model.decoder(en, F0_fake, N_fake, s)
                writer.add_audio(f'{audio_prefix}/pred_y{bib}', y_pred.cpu().numpy().squeeze(), tag_step, sample_rate=sr)

                if log_gt:
                    writer.add_audio(f'{audio_prefix}/gt_y{bib}', waves[bib].squeeze(), tag_step, sample_rate=sr)

                if bib >= max_audios - 1:
                    break
        else:
            # generating sampled speech from text directly
            if multispeaker and epoch >= diff_epoch:
                ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                ref_s = torch.cat([ref_ss, ref_sp], dim=1)

            for bib in range(len(d_en)):
                torch.manual_seed(int(sample_seed) + int(bib))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(sample_seed) + int(bib))

                if multispeaker:
                    s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(device),
                                     embedding=bert_dur[bib].unsqueeze(0),
                                     embedding_scale=1,
                                     features=ref_s[bib].unsqueeze(0),
                                     num_steps=5).squeeze(1)
                else:
                    s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(device),
                                     embedding=bert_dur[bib].unsqueeze(0),
                                     embedding_scale=1,
                                     num_steps=5).squeeze(1)

                s = s_pred[:, 128:]
                ref = s_pred[:, :128]

                d = model.predictor.text_encoder(
                    d_en[bib, :, :input_lengths[bib]].unsqueeze(0),
                    s,
                    input_lengths[bib, ...].unsqueeze(0),
                    text_mask[bib, :input_lengths[bib]].unsqueeze(0)
                )

                x, _ = model.predictor.lstm(d)
                duration = model.predictor.duration_proj(x)
                duration = torch.sigmoid(duration).sum(axis=-1)
                pred_dur = torch.round(duration.squeeze()).clamp(min=1)
                pred_dur[-1] += 5

                pred_aln_trg = torch.zeros(input_lengths[bib], int(pred_dur.sum().data))
                c_frame = 0
                for j in range(pred_aln_trg.size(0)):
                    pred_aln_trg[j, c_frame:c_frame + int(pred_dur[j].data)] = 1
                    c_frame += int(pred_dur[j].data)

                # encode prosody
                en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
                F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
                out = model.decoder(
                    (t_en[bib, :, :input_lengths[bib]].unsqueeze(0) @ pred_aln_trg.unsqueeze(0).to(device)),
                    F0_pred,
                    N_pred,
                    ref.squeeze().unsqueeze(0)
                )

                writer.add_audio(f'{audio_prefix}/pred_y{bib}', out.cpu().numpy().squeeze(), tag_step, sample_rate=sr)

                if log_gt:
                    writer.add_audio(f'{audio_prefix}/gt_y{bib}', waves[bib].squeeze(), tag_step, sample_rate=sr)

                if bib >= max_audios - 1:
                    break
    finally:
        torch.set_rng_state(cpu_rng_state)
        if torch.cuda.is_available() and cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)


def _run_stage2_validation(
    model,
    val_dataloader,
    stft_loss,
    writer,
    logger,
    n_down,
    max_len,
    device,
    epoch,
    step,
    sr,
    sampler,
    multispeaker,
    diff_epoch,
    joint_epoch,
    max_batches=None,
    scalar_prefix='eval_step',
    log_audio=False,
    audio_prefix='step_sample',
    sample_num_audios=5,
    eval_seed=1234,
    sample_seed=4321,
    log_gt=True,
):
    """
    Validation helper.
    - max_batches=None => full validation.
    - max_batches=N    => fixed validation subset from the beginning of val_dataloader.
    """
    loss_test = 0.0
    loss_align = 0.0
    loss_f = 0.0
    iters_test = 0
    audio_state = None

    _ = [model[key].eval() for key in model]

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            if max_batches is not None and iters_test >= int(max_batches):
                break

            try:
                state = _process_stage2_val_batch(
                    batch=batch,
                    batch_idx=batch_idx,
                    model=model,
                    stft_loss=stft_loss,
                    n_down=n_down,
                    max_len=max_len,
                    device=device,
                    eval_seed=eval_seed,
                )
                if state is None:
                    continue

                loss_test += _scalar(state['loss_mel'])
                loss_align += _scalar(state['loss_dur'])
                loss_f += _scalar(state['loss_F0'])
                iters_test += 1

                if audio_state is None:
                    audio_state = state

            except Exception as e:
                print(f"run into exception", e)
                traceback.print_exc()
                continue

    if iters_test == 0:
        logger.warning('Validation skipped: no valid validation batch was processed.')
        return None

    avg_mel = loss_test / iters_test
    avg_dur = loss_align / iters_test
    avg_f0 = loss_f / iters_test

    if scalar_prefix:
        writer.add_scalar(f'{scalar_prefix}/mel_loss', avg_mel, step)
        writer.add_scalar(f'{scalar_prefix}/dur_loss', avg_dur, step)
        writer.add_scalar(f'{scalar_prefix}/F0_loss', avg_f0, step)

    if log_audio:
        _log_stage2_audio_samples(
            writer=writer,
            state=audio_state,
            model=model,
            sampler=sampler,
            epoch=epoch,
            step=step,
            sr=sr,
            device=device,
            multispeaker=multispeaker,
            diff_epoch=diff_epoch,
            joint_epoch=joint_epoch,
            max_audios=sample_num_audios,
            audio_prefix=audio_prefix,
            sample_seed=sample_seed,
            log_gt=log_gt,
        )

    writer.flush()
    return {
        'mel_loss': avg_mel,
        'dur_loss': avg_dur,
        'F0_loss': avg_f0,
        'num_batches': iters_test,
    }


@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))
    
    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    batch_size = config.get('batch_size', 10)

    epochs = config.get('epochs_2nd', 200)
    save_freq = config.get('save_freq', 2)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)

    # Step checkpoint settings. Kept compatible with your current config.
    save_step_freq = int(config.get('save_step_freq', 0) or 0)
    save_total_limit = int(config.get('save_total_limit', 0) or 0)
    resume_skip_batches = bool(config.get('resume_skip_batches', False))
    resume_batch_idx = 0

    # Step validation/audio settings.
    # Defaults match your requested schedule:
    #   every 4000 steps: subset validation loss on 100 batches
    #   every 10000 steps: 3-5 fixed audio samples
    eval_step_freq = int(config.get('eval_step_freq', 4000) or 0)
    eval_step_max_batches = int(config.get('eval_step_max_batches', 100) or 100)
    sample_step_freq = int(config.get('sample_step_freq', 10000) or 0)
    sample_num_audios = int(config.get('sample_num_audios', 5) or 5)
    sample_num_audios = max(1, min(sample_num_audios, 5))
    eval_seed = int(config.get('eval_seed', 1234) or 1234)
    sample_seed = int(config.get('sample_seed', 4321) or 4321)
    full_eval_each_epoch = bool(config.get('full_eval_each_epoch', True))

    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    min_length = data_params['min_length']
    OOD_data = data_params['OOD_data']

    max_len = config.get('max_len', 200)
    
    loss_params = Munch(config['loss_params'])
    diff_epoch = loss_params.diff_epoch
    joint_epoch = loss_params.joint_epoch
    
    optimizer_params = Munch(config['optimizer_params'])
    
    train_list, val_list = get_data_path_list(train_path, val_path)
    device = 'cuda'

    train_dataloader = build_dataloader(train_list,
                                        root_path,
                                        OOD_data=OOD_data,
                                        min_length=min_length,
                                        batch_size=batch_size,
                                        num_workers=2,
                                        dataset_config={},
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                      root_path,
                                      OOD_data=OOD_data,
                                      min_length=min_length,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=0,
                                      device=device,
                                      dataset_config={})
    
    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)
    
    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)
    
    # load PL-BERT model
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)
    
    # build model
    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].to(device) for key in model]
    
    # DP
    for key in model:
        if key != "mpd" and key != "msd" and key != "wd":
            model[key] = MyDataParallel(model[key])
            
    start_epoch = 0
    iters = 0

    load_pretrained = config.get('pretrained_model', '') != '' and config.get('second_stage_load_pretrained', False)
    
    if not load_pretrained:
        if config.get('first_stage_path', '') != '':
            first_stage_path = osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
            print('Loading the first stage model at %s ...' % first_stage_path)
            model, _, start_epoch, iters = load_checkpoint(model, 
                None, 
                first_stage_path,
                load_only_params=True,
                ignore_modules=['bert', 'bert_encoder', 'predictor', 'predictor_encoder', 'msd', 'mpd', 'wd', 'diffusion']) # keep starting epoch for tensorboard log

            # these epochs should be counted from the start epoch
            diff_epoch += start_epoch
            joint_epoch += start_epoch
            epochs += start_epoch
            
            model.predictor_encoder = copy.deepcopy(model.style_encoder)
        else:
            raise ValueError('You need to specify the path to the first stage model.') 

    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = WavLMLoss(model_params.slm.model, 
                   model.wd, 
                   sr, 
                   model_params.slm.sr).to(device)

    gl = MyDataParallel(gl)
    dl = MyDataParallel(dl)
    wl = MyDataParallel(wl)
    
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )
    
    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    scheduler_params_dict= {key: scheduler_params.copy() for key in model}
    scheduler_params_dict['bert']['max_lr'] = optimizer_params.bert_lr * 2
    scheduler_params_dict['decoder']['max_lr'] = optimizer_params.ft_lr * 2
    scheduler_params_dict['style_encoder']['max_lr'] = optimizer_params.ft_lr * 2
    
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                          scheduler_params_dict=scheduler_params_dict, lr=optimizer_params.lr)
    
    # adjust BERT learning rate
    for g in optimizer.optimizers['bert'].param_groups:
        g['betas'] = (0.9, 0.99)
        g['lr'] = optimizer_params.bert_lr
        g['initial_lr'] = optimizer_params.bert_lr
        g['min_lr'] = 0
        g['weight_decay'] = 0.01
        
    # adjust acoustic module learning rate
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g['betas'] = (0.0, 0.99)
            g['lr'] = optimizer_params.ft_lr
            g['initial_lr'] = optimizer_params.ft_lr
            g['min_lr'] = 0
            g['weight_decay'] = 1e-4
        
    # load models if there is a model
    if load_pretrained:
        model, optimizer, start_epoch, iters = load_checkpoint(model,  optimizer, config['pretrained_model'],
                                    load_only_params=config.get('load_only_params', True))
        if resume_skip_batches:
            resume_batch_idx = _get_resume_batch_idx(config['pretrained_model'])
            if resume_batch_idx > 0:
                print(f'[resume] will skip {resume_batch_idx} batches in epoch {start_epoch}')
        
    n_down = model.text_aligner.n_down

    best_loss = float('inf')  # best test loss
    loss_train_record = list([])
    loss_test_record = list([])
    if not load_pretrained:
        iters = 0
    
    criterion = nn.L1Loss() # F0 loss (regression)
    torch.cuda.empty_cache()
    
    stft_loss = MultiResolutionSTFTLoss().to(device)
    
    print('BERT', optimizer.optimizers['bert'])
    print('decoder', optimizer.optimizers['decoder'])

    start_ds = False
    
    running_std = []
    
    slmadv_params = Munch(config['slmadv_params'])
    slmadv = SLMAdversarialLoss(model, wl, sampler, 
                                slmadv_params.min_len, 
                                slmadv_params.max_len,
                                batch_percentage=slmadv_params.batch_percentage,
                                skip_update=slmadv_params.iter, 
                                sig=slmadv_params.sig
                               )

    logger.info(
        'Stage 2 monitoring schedule: eval_step_freq=%s, eval_step_max_batches=%s, '
        'sample_step_freq=%s, sample_num_audios=%s, full_eval_each_epoch=%s'
        % (eval_step_freq, eval_step_max_batches, sample_step_freq, sample_num_audios, full_eval_each_epoch)
    )

    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()

        _set_stage2_train_mode(model)

        if epoch >= diff_epoch:
            start_ds = True

        for i, batch in enumerate(train_dataloader):
            if resume_skip_batches and epoch == start_epoch and i < resume_batch_idx:
                if (i + 1) % 1000 == 0:
                    print(f'[resume] skipping batch {i + 1}/{resume_batch_idx}')
                continue

            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch

            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                mel_mask = length_to_mask(mel_input_length).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)

                try:
                    _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)
                except:
                    continue

                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                # encode
                t_en = model.text_encoder(texts, input_lengths, text_mask)
                asr = (t_en @ s2s_attn_mono)

                d_gt = s2s_attn_mono.sum(axis=-1).detach()
                
                # compute reference styles
                if multispeaker and epoch >= diff_epoch:
                    ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                    ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                    ref = torch.cat([ref_ss, ref_sp], dim=1)

            # compute the style of the entire utterance
            # this operation cannot be done in batch because of the avgpool layer (may need to work on masked avgpool)
            ss = []
            gs = []
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item())
                mel = mels[bib, :, :mel_input_length[bib]]
                s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                ss.append(s)
                s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                gs.append(s)

            s_dur = torch.stack(ss).squeeze()  # global prosodic styles
            gs = torch.stack(gs).squeeze() # global acoustic styles
            s_trg = torch.cat([gs, s_dur], dim=-1).detach() # ground truth for denoiser

            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 
            
            # denoiser training
            if epoch >= diff_epoch:
                num_steps = np.random.randint(3, 5)
                
                if model_params.diffusion.dist.estimate_sigma_data:
                    model.diffusion.module.diffusion.sigma_data = s_trg.std(axis=-1).mean().item() # batch-wise std estimation
                    running_std.append(model.diffusion.module.diffusion.sigma_data)
                    
                if multispeaker:
                    s_preds = sampler(noise = torch.randn_like(s_trg).unsqueeze(1).to(device), 
                          embedding=bert_dur,
                          embedding_scale=1,
                                   features=ref, # reference from the same speaker as the embedding
                             embedding_mask_proba=0.1,
                             num_steps=num_steps).squeeze(1)
                    loss_diff = model.diffusion(s_trg.unsqueeze(1), embedding=bert_dur, features=ref).mean() # EDM loss
                    loss_sty = F.l1_loss(s_preds, s_trg.detach()) # style reconstruction loss
                else:
                    s_preds = sampler(noise = torch.randn_like(s_trg).unsqueeze(1).to(device), 
                          embedding=bert_dur,
                          embedding_scale=1,
                             embedding_mask_proba=0.1,
                             num_steps=num_steps).squeeze(1)                    
                    loss_diff = model.diffusion.module.diffusion(s_trg.unsqueeze(1), embedding=bert_dur).mean() # EDM loss
                    loss_sty = F.l1_loss(s_preds, s_trg.detach()) # style reconstruction loss
            else:
                loss_sty = 0
                loss_diff = 0

            d, p = model.predictor(d_en, s_dur, 
                                                    input_lengths, 
                                                    s2s_attn_mono, 
                                                    text_mask)
            
            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)
            en = []
            gt = []
            st = []
            p_en = []
            wav = []

            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)

                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start:random_start+mel_len])
                p_en.append(p[bib, :, random_start:random_start+mel_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                
                y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).to(device))

                # style reference (better to be different from the GT)
                random_start = np.random.randint(0, mel_length - mel_len_st)
                st.append(mels[bib, :, (random_start * 2):((random_start+mel_len_st) * 2)])
                
            wav = torch.stack(wav).float().detach()

            en = torch.stack(en)
            p_en = torch.stack(p_en)
            gt = torch.stack(gt).detach()
            st = torch.stack(st).detach()
            
            if gt.size(-1) < 80:
                continue

            s_dur = model.predictor_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
            s = model.style_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
            
            with torch.no_grad():
                F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                F0 = F0.reshape(F0.shape[0], F0.shape[1] * 2, F0.shape[2], 1).squeeze()

                asr_real = model.text_aligner.get_feature(gt)

                N_real = log_norm(gt.unsqueeze(1)).squeeze(1)
                
                y_rec_gt = wav.unsqueeze(1)
                y_rec_gt_pred = model.decoder(en, F0_real, N_real, s)

                if epoch >= joint_epoch:
                    # ground truth from recording
                    wav = y_rec_gt # use recording since decoder is tuned
                else:
                    # ground truth from reconstruction
                    wav = y_rec_gt_pred # use reconstruction since decoder is fixed

            F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)

            y_rec = model.decoder(en, F0_fake, N_fake, s)

            loss_F0_rec =  (F.smooth_l1_loss(F0_real, F0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)

            if start_ds:
                optimizer.zero_grad()
                d_loss = dl(wav.detach(), y_rec.detach()).mean()
                d_loss.backward()
                optimizer.step('msd')
                optimizer.step('mpd')
            else:
                d_loss = 0

            # generator loss
            optimizer.zero_grad()

            loss_mel = stft_loss(y_rec, wav)
            if start_ds:
                loss_gen_all = gl(wav, y_rec).mean()
            else:
                loss_gen_all = 0
            loss_lm = wl(wav.detach().squeeze(), y_rec.squeeze()).mean()

            loss_ce = 0
            loss_dur = 0
            for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                _s2s_pred = _s2s_pred[:_text_length, :]
                _text_input = _text_input[:_text_length].long()
                _s2s_trg = torch.zeros_like(_s2s_pred)
                for p_idx in range(_s2s_trg.shape[0]):
                    _s2s_trg[p_idx, :_text_input[p_idx]] = 1
                _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

                loss_dur += F.l1_loss(_dur_pred[1:_text_length-1], 
                                       _text_input[1:_text_length-1])
                loss_ce += F.binary_cross_entropy_with_logits(_s2s_pred.flatten(), _s2s_trg.flatten())

            loss_ce /= texts.size(0)
            loss_dur /= texts.size(0)

            g_loss = loss_params.lambda_mel * loss_mel + \
                     loss_params.lambda_F0 * loss_F0_rec + \
                     loss_params.lambda_ce * loss_ce + \
                     loss_params.lambda_norm * loss_norm_rec + \
                     loss_params.lambda_dur * loss_dur + \
                     loss_params.lambda_gen * loss_gen_all + \
                     loss_params.lambda_slm * loss_lm + \
                     loss_params.lambda_sty * loss_sty + \
                     loss_params.lambda_diff * loss_diff

            running_loss += loss_mel.item()
            g_loss.backward()
            if torch.isnan(g_loss):
                from IPython.core.debugger import set_trace
                set_trace()

            optimizer.step('bert_encoder')
            optimizer.step('bert')
            optimizer.step('predictor')
            optimizer.step('predictor_encoder')
            
            if epoch >= diff_epoch:
                optimizer.step('diffusion')
            
            if epoch >= joint_epoch:
                optimizer.step('style_encoder')
                optimizer.step('decoder')
        
                # randomly pick whether to use in-distribution text
                if np.random.rand() < 0.5:
                    use_ind = True
                else:
                    use_ind = False

                if use_ind:
                    ref_lengths = input_lengths
                    ref_texts = texts
                    
                slm_out = slmadv(i, 
                                 y_rec_gt, 
                                 y_rec_gt_pred, 
                                 waves, 
                                 mel_input_length,
                                 ref_texts, 
                                 ref_lengths, use_ind, s_trg.detach(), ref if multispeaker else None)

                if slm_out is None:
                    continue
                    
                d_loss_slm, loss_gen_lm, y_pred = slm_out
                
                # SLM generator loss
                optimizer.zero_grad()
                loss_gen_lm.backward()

                # compute the gradient norm
                total_norm = {}
                for key in model.keys():
                    total_norm[key] = 0
                    parameters = [p for p in model[key].parameters() if p.grad is not None and p.requires_grad]
                    for p_param in parameters:
                        param_norm = p_param.grad.detach().data.norm(2)
                        total_norm[key] += param_norm.item() ** 2
                    total_norm[key] = total_norm[key] ** 0.5

                # gradient scaling
                if total_norm['predictor'] > slmadv_params.thresh:
                    for key in model.keys():
                        for p_param in model[key].parameters():
                            if p_param.grad is not None:
                                p_param.grad *= (1 / total_norm['predictor']) 

                for p_param in model.predictor.duration_proj.parameters():
                    if p_param.grad is not None:
                        p_param.grad *= slmadv_params.scale

                for p_param in model.predictor.lstm.parameters():
                    if p_param.grad is not None:
                        p_param.grad *= slmadv_params.scale

                for p_param in model.diffusion.parameters():
                    if p_param.grad is not None:
                        p_param.grad *= slmadv_params.scale

                optimizer.step('bert_encoder')
                optimizer.step('bert')
                optimizer.step('predictor')
                optimizer.step('diffusion')

                # SLM discriminator loss
                if d_loss_slm != 0:
                    optimizer.zero_grad()
                    d_loss_slm.backward(retain_graph=True)
                    optimizer.step('wd')

            else:
                d_loss_slm, loss_gen_lm = 0, 0
                
            iters = iters + 1

            if save_step_freq > 0 and iters % save_step_freq == 0:
                print('Saving step checkpoint..')
                state = {
                    'net':  {key: model[key].state_dict() for key in model}, 
                    'optimizer': optimizer.state_dict(),
                    'iters': iters,
                    'val_loss': None,
                    'epoch': epoch,
                    'batch_idx': i + 1,
                }
                save_path = osp.join(log_dir, 'step_2nd_%09d.pth' % iters)
                torch.save(state, save_path)
                _cleanup_step_checkpoints(log_dir, 'step_2nd_', save_total_limit)
            
            if (i+1) % log_interval == 0:
                logger.info ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Total G Loss: %.5f, Disc Loss: %.5f, Dur Loss: %.5f, CE Loss: %.5f, Norm Loss: %.5f, F0 Loss: %.5f, LM Loss: %.5f, Gen Loss: %.5f, Sty Loss: %.5f, Diff Loss: %.5f, DiscLM Loss: %.5f, GenLM Loss: %.5f'
                    %(epoch+1, epochs, i+1, len(train_list)//batch_size, running_loss / log_interval, _scalar(g_loss), _scalar(d_loss), _scalar(loss_dur), _scalar(loss_ce), _scalar(loss_norm_rec), _scalar(loss_F0_rec), _scalar(loss_lm), _scalar(loss_gen_all), _scalar(loss_sty), _scalar(loss_diff), _scalar(d_loss_slm), _scalar(loss_gen_lm)))
                
                writer.add_scalar('train/mel_loss', running_loss / log_interval, iters)
                writer.add_scalar('train/g_loss', _scalar(g_loss), iters)
                writer.add_scalar('train/gen_loss', _scalar(loss_gen_all), iters)
                writer.add_scalar('train/d_loss', _scalar(d_loss), iters)
                writer.add_scalar('train/ce_loss', _scalar(loss_ce), iters)
                writer.add_scalar('train/dur_loss', _scalar(loss_dur), iters)
                writer.add_scalar('train/slm_loss', _scalar(loss_lm), iters)
                writer.add_scalar('train/norm_loss', _scalar(loss_norm_rec), iters)
                writer.add_scalar('train/F0_loss', _scalar(loss_F0_rec), iters)
                writer.add_scalar('train/sty_loss', _scalar(loss_sty), iters)
                writer.add_scalar('train/diff_loss', _scalar(loss_diff), iters)
                writer.add_scalar('train/d_loss_slm', _scalar(d_loss_slm), iters)
                writer.add_scalar('train/gen_loss_slm', _scalar(loss_gen_lm), iters)
                
                running_loss = 0
                
                print('Time elasped:', time.time()-start_time)

            # Step-based validation on a fixed validation subset.
            if eval_step_freq > 0 and iters % eval_step_freq == 0:
                print(f'Running subset validation at global step {iters} ...')
                stats = _run_stage2_validation(
                    model=model,
                    val_dataloader=val_dataloader,
                    stft_loss=stft_loss,
                    writer=writer,
                    logger=logger,
                    n_down=n_down,
                    max_len=max_len,
                    device=device,
                    epoch=epoch,
                    step=iters,
                    sr=sr,
                    sampler=sampler,
                    multispeaker=multispeaker,
                    diff_epoch=diff_epoch,
                    joint_epoch=joint_epoch,
                    max_batches=eval_step_max_batches,
                    scalar_prefix='eval_step',
                    log_audio=False,
                    sample_num_audios=sample_num_audios,
                    eval_seed=eval_seed,
                    sample_seed=sample_seed,
                    log_gt=False,
                )
                if stats is not None:
                    logger.info(
                        'Step validation @ %d: Mel loss: %.3f, Dur loss: %.3f, F0 loss: %.3f, batches: %d'
                        % (iters, stats['mel_loss'], stats['dur_loss'], stats['F0_loss'], stats['num_batches'])
                    )
                _set_stage2_train_mode(model)

            # Step-based fixed audio sampling from validation examples.
            if sample_step_freq > 0 and iters % sample_step_freq == 0:
                print(f'Generating validation audio samples at global step {iters} ...')
                _run_stage2_validation(
                    model=model,
                    val_dataloader=val_dataloader,
                    stft_loss=stft_loss,
                    writer=writer,
                    logger=logger,
                    n_down=n_down,
                    max_len=max_len,
                    device=device,
                    epoch=epoch,
                    step=iters,
                    sr=sr,
                    sampler=sampler,
                    multispeaker=multispeaker,
                    diff_epoch=diff_epoch,
                    joint_epoch=joint_epoch,
                    max_batches=1,
                    scalar_prefix=None,
                    log_audio=True,
                    audio_prefix='step_sample',
                    sample_num_audios=sample_num_audios,
                    eval_seed=eval_seed,
                    sample_seed=sample_seed,
                    log_gt=True,
                )
                _set_stage2_train_mode(model)

        # End-of-epoch full validation.
        if full_eval_each_epoch:
            print('Running full validation at epoch end ...')
            stats = _run_stage2_validation(
                model=model,
                val_dataloader=val_dataloader,
                stft_loss=stft_loss,
                writer=writer,
                logger=logger,
                n_down=n_down,
                max_len=max_len,
                device=device,
                epoch=epoch,
                step=epoch + 1,
                sr=sr,
                sampler=sampler,
                multispeaker=multispeaker,
                diff_epoch=diff_epoch,
                joint_epoch=joint_epoch,
                max_batches=None,
                scalar_prefix='eval',
                log_audio=True,
                audio_prefix='eval',
                sample_num_audios=5,
                eval_seed=eval_seed,
                sample_seed=sample_seed,
                log_gt=(epoch == 0),
            )
        else:
            stats = None

        print('Epochs:', epoch + 1)
        if stats is not None:
            loss_test_value = stats['mel_loss']
            logger.info('Validation loss: %.3f, Dur loss: %.3f, F0 loss: %.3f' % (stats['mel_loss'], stats['dur_loss'], stats['F0_loss']) + '\n\n\n')
        else:
            loss_test_value = None
            logger.warning('No full validation stats available at epoch end. Check validation dataloader/errors.')
        print('\n\n\n')

        if epoch % saving_epoch == 0:
            if loss_test_value is not None and loss_test_value < best_loss:
                best_loss = loss_test_value
            print('Saving..')
            state = {
                'net':  {key: model[key].state_dict() for key in model}, 
                'optimizer': optimizer.state_dict(),
                'iters': iters,
                'val_loss': loss_test_value,
                'epoch': epoch,
            }
            save_path = osp.join(log_dir, 'epoch_2nd_%05d.pth' % epoch)
            torch.save(state, save_path)
            
            # if estimate sigma, save the estimated simga
            if model_params.diffusion.dist.estimate_sigma_data and len(running_std) > 0:
                config['model_params']['diffusion']['dist']['sigma_data'] = float(np.mean(running_std))
                
                with open(osp.join(log_dir, osp.basename(config_path)), 'w') as outfile:
                    yaml.dump(config, outfile, default_flow_style=True)
        
if __name__=="__main__":
    main()
