"""
=============================================================
  D0: INFERENCE ENGINE — Core module cho StyleTTS2-lite-vi
=============================================================
Mục đích:
  Module CORE để load checkpoint fine-tuned + sinh audio.
  Mọi file D1-D3 + UI sẽ import từ đây.

Thiết kế:
  - 1 class StyleTTS2LiteVNInference với 4 method chính:
        - __init__()       : load model, config, vocab — chạy 1 lần
        - compute_style()  : extract style vector từ audio reference
        - text_to_phoneme(): convert text -> espeak phoneme (đã normalize)
        - synthesize()     : phoneme + style -> audio numpy 24kHz

Hardware target: RTX 3050Ti 4GB VRAM + 16GB RAM (Win 11)
  - FP16 inference (autocast) giảm VRAM ~50%
  - Tự động fallback CPU nếu CUDA OOM
  - Cache style vectors để inference batch nhanh

Cách dùng:
    from inference_engine import StyleTTS2LiteVNInference

    engine = StyleTTS2LiteVNInference(
        checkpoint_path="Models/epoch_00030.pth",
        repo_root="StyleTTS2-lite",                # repo gốc clone từ GitHub
        config_path="Configs/config_ngan_kaggle.yml",  # hoặc config.yaml gốc lite-vi
    )

    # 1) Trích style từ audio Ngạn (làm 1 lần, cache lại)
    ngan_style = engine.compute_style("ref/ngan_sample.wav")

    # 2) Sinh audio cho 1 câu
    phn = engine.text_to_phoneme("Đêm hôm ấy trời tối đen như mực.")
    wav = engine.synthesize(phn, ngan_style)

    # 3) Save
    import soundfile as sf
    sf.write("out.wav", wav, 24000)

Lệnh chạy smoke test (đọc checkpoint/repo/model_config/ref/text/out từ inference_config.yaml):
    python inference/inference_engine.py --config inference/inference_config.yaml

Override nhanh qua CLI (ưu tiên CLI > config > default):
    python inference/inference_engine.py --config inference/inference_config.yaml \\
        --no-fp16 --text "Câu test khác." --out output/smoke_test.wav
=============================================================
"""

from __future__ import annotations

import os
import re
import sys
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import yaml
from munch import Munch


# ============================================================
# Constants & module-level setup
# ============================================================
SR = 24000  # sample rate cố định của lite-vi
N_FFT = 2048
WIN_LENGTH = 1200
HOP_LENGTH = 300

# Combining marks và hyphens cần strip khỏi phoneme espeak
# để khớp vocab 189 symbols của lite-vi.
_PHONEME_REPLACEMENTS = {
    "\u032A": "",   # combining dental bridge below (t̪/d̪/n̪ → t/d/n)
    "-":      " ",  # ASCII hyphen
    "\u2010": " ",  # Unicode hyphen
    "\u2011": " ",  # non-breaking hyphen
    "\u2012": " ",  # figure dash
    "\u2013": " ",  # en dash
    "\u2014": " ",  # em dash
    "\u2212": " ",  # minus sign
}

# Modules CẦN cho inference (4 modules theo README lite-vi)
INFERENCE_MODULES = ["decoder", "predictor", "text_encoder", "style_encoder"]

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)


# ============================================================
# Utility functions (module-level, dùng nội bộ)
# ============================================================
def _recursive_munch(d):
    """Convert nested dict -> nested Munch (giống utils.recursive_munch của repo)."""
    if isinstance(d, dict):
        return Munch((k, _recursive_munch(v)) for k, v in d.items())
    if isinstance(d, list):
        return [_recursive_munch(v) for v in d]
    return d


def _length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Helper từ inference.py của lite-vi."""
    mask = (
        torch.arange(lengths.max())
        .unsqueeze(0)
        .expand(lengths.shape[0], -1)
        .type_as(lengths)
    )
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


def _replace_outliers_zscore(
    tensor: torch.Tensor, threshold: float = 3.0, factor: float = 0.95
) -> torch.Tensor:
    """Helper từ inference.py — smooth outlier durations."""
    mean = tensor.mean()
    std = tensor.std()
    z = (tensor - mean) / (std + 1e-9)
    outlier_mask = torch.abs(z) > threshold
    sign = torch.sign(tensor - mean)
    replacement = mean + sign * (threshold * std * factor)
    result = tensor.clone()
    result[outlier_mask] = replacement[outlier_mask]
    return result


def _build_symbol_dict_from_config(cfg: dict) -> tuple[dict, int]:
    """
    Build symbol_dict ĐÚNG CÁCH GIỐNG inference.py của lite-vi:
        symbols = pad + punctuation + letters + letters_ipa + extend
        n_token = len(symbol_dict) + 1
    Đã verified ở file A2.
    """
    sym = cfg["symbol"]
    symbols = (
        list(sym["pad"])
        + list(sym["punctuation"])
        + list(sym["letters"])
        + list(sym["letters_ipa"])
        + list(sym["extend"])
    )
    symbol_dict = {symbols[i]: i for i in range(len(symbols))}
    n_token = len(symbol_dict) + 1
    return symbol_dict, n_token


def _normalize_espeak_phoneme(phn: str) -> str:
    """
    Normalize espeak output:
      1. Strip combining marks không có trong vocab lite-vi (̪ ...)
      2. Replace hyphens bằng space
      3. Collapse multi-space, strip
    Giống hệt logic step1b_rephonemize_espeak_lite.py của user.
    """
    if not phn:
        return ""
    for src, dst in _PHONEME_REPLACEMENTS.items():
        phn = phn.replace(src, dst)
    phn = re.sub(r"\s+", " ", phn).strip()
    return phn


# ============================================================
# Main class
# ============================================================
class StyleTTS2LiteVNInference:
    """
    Inference wrapper cho StyleTTS2-lite-vi đã fine-tune trên giọng Ngạn.

    Tham số __init__:
        checkpoint_path: file .pth fine-tuned (vd epoch_00030.pth)
        repo_root:       thư mục chứa source code lite-vi (models.py, Modules/, ...)
        config_path:     YAML config (dùng key 'symbol' và 'model_params')
        device:          'cuda' / 'cpu' / 'auto' (auto: cuda nếu có, ngược lại cpu)
        use_fp16:        True -> dùng autocast FP16 (giảm VRAM ~50%, tăng tốc)
        cpu_fallback:    True -> khi CUDA OOM, tự động retry trên CPU
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        repo_root: Union[str, Path],
        config_path: Union[str, Path],
        device: str = "auto",
        use_fp16: bool = True,
        cpu_fallback: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.repo_root = Path(repo_root).resolve()
        self.config_path = Path(config_path).resolve()
        self.cpu_fallback = cpu_fallback

        # ===== Validate files =====
        for p, label in [
            (self.checkpoint_path, "checkpoint"),
            (self.repo_root, "repo_root"),
            (self.config_path, "config"),
        ]:
            if not p.exists():
                raise FileNotFoundError(f"Không tìm thấy {label}: {p}")

        # ===== Resolve device =====
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # FP16 chỉ có ý nghĩa trên cuda
        self.use_fp16 = use_fp16 and self.device == "cuda"

        logger.info(f"Device       : {self.device}")
        logger.info(f"FP16 autocast: {self.use_fp16}")

        # ===== Insert repo_root vào sys.path để import models.py, Modules/, ... =====
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        # ===== Load config + build vocab =====
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.symbol_dict, self.n_token = _build_symbol_dict_from_config(self.config)
        logger.info(f"Vocab        : {len(self.symbol_dict)} symbols, n_token={self.n_token}")

        # ===== Build 4 components inference =====
        self._build_model()

        # ===== Load checkpoint weights =====
        self._load_checkpoint()

        # ===== Setup helpers (mel transform, TextCleaner, phonemizer) =====
        self._setup_helpers()

        logger.info("✅ Inference engine ready.")

    # --------------------------------------------------------
    # Setup methods (private)
    # --------------------------------------------------------
    def _build_model(self) -> None:
        """Build 4 components inference. Import từ repo lite-vi."""
        from models import ProsodyPredictor, TextEncoder, StyleEncoder
        from Modules.hifigan import Decoder

        args = _recursive_munch(self.config["model_params"])
        args.n_token = self.n_token

        if args.decoder.type != "hifigan":
            raise ValueError(
                f"Decoder type phải là 'hifigan', got {args.decoder.type}. "
                "Lite-vi chỉ support hifigan."
            )

        self.decoder = Decoder(
            dim_in=args.hidden_dim,
            style_dim=args.style_dim,
            dim_out=args.n_mels,
            resblock_kernel_sizes=args.decoder.resblock_kernel_sizes,
            upsample_rates=args.decoder.upsample_rates,
            upsample_initial_channel=args.decoder.upsample_initial_channel,
            resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
            upsample_kernel_sizes=args.decoder.upsample_kernel_sizes,
        ).to(self.device)

        self.predictor = ProsodyPredictor(
            style_dim=args.style_dim,
            d_hid=args.hidden_dim,
            nlayers=args.n_layer,
            max_dur=args.max_dur,
            dropout=args.dropout,
        ).to(self.device)

        self.text_encoder = TextEncoder(
            channels=args.hidden_dim,
            kernel_size=5,
            depth=args.n_layer,
            n_symbols=args.n_token,
        ).to(self.device)

        self.style_encoder = StyleEncoder(
            dim_in=args.dim_in,
            style_dim=args.style_dim,
            max_conv_dim=args.hidden_dim,
        ).to(self.device)

        # Dict cho việc load weights, đặt eval mode
        self._modules_dict = {
            "decoder": self.decoder,
            "predictor": self.predictor,
            "text_encoder": self.text_encoder,
            "style_encoder": self.style_encoder,
        }

    def _load_checkpoint(self) -> None:
        """
        Load weights từ checkpoint .pth.
        Format expected (từ train.py):
            state = {'net': {key: state_dict, ...}, 'optimizer': ..., 'epoch': ..., 'iters': ...}
        Vì train.py wrap mỗi module bằng MyDataParallel, state_dict CÓ THỂ có
        prefix 'module.' ở keys → cần strip.
        """
        logger.info(f"Loading checkpoint: {self.checkpoint_path}")
        size_mb = self.checkpoint_path.stat().st_size / 1e6
        logger.info(f"  Size: {size_mb:.1f} MB")

        params_whole = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )

        # Chấp nhận cả 2 format: {'net': {...}} hoặc thuần state_dicts
        if "net" in params_whole:
            params = params_whole["net"]
            self.checkpoint_epoch = params_whole.get("epoch", "?")
            self.checkpoint_iters = params_whole.get("iters", "?")
            self.checkpoint_val_loss = params_whole.get("val_loss", "?")
            logger.info(
                f"  Trained: epoch={self.checkpoint_epoch}, "
                f"iters={self.checkpoint_iters}, "
                f"val_loss={self.checkpoint_val_loss}"
            )
        else:
            params = params_whole
            self.checkpoint_epoch = None
            self.checkpoint_iters = None

        total_params = 0
        for key in INFERENCE_MODULES:
            if key not in params:
                raise KeyError(f"Checkpoint thiếu module '{key}'")
            state_dict = params[key]
            # Strip 'module.' prefix nếu có (do DataParallel khi training)
            cleaned = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                cleaned[name] = v
            try:
                self._modules_dict[key].load_state_dict(cleaned, strict=True)
            except RuntimeError as e:
                logger.warning(f"  '{key}' strict load fail, retry strict=False: {e}")
                missing, unexpected = self._modules_dict[key].load_state_dict(
                    cleaned, strict=False
                )
                if missing:
                    logger.warning(f"    Missing keys ({len(missing)}): {missing[:3]}...")
                if unexpected:
                    logger.warning(f"    Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")

            self._modules_dict[key].eval()
            n = sum(p.numel() for p in self._modules_dict[key].parameters())
            total_params += n
            logger.info(f"  {key:14s}: {n/1e6:6.2f} M params")

        logger.info(f"  {'TOTAL':14s}: {total_params/1e6:6.2f} M params")

    def _setup_helpers(self) -> None:
        """Setup mel transform, TextCleaner, phonemizer."""
        import torchaudio
        from nltk.tokenize import word_tokenize
        import nltk

        # NLTK setup
        for resource in ["punkt_tab", "punkt"]:
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)
        self._word_tokenize = word_tokenize

        # Mel transform — settings GIỐNG hệt training (inference.py + meldataset.py)
        self._to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH
        )

        # TextCleaner từ meldataset.py của lite-vi
        from meldataset import TextCleaner
        self._cleaner = TextCleaner(self.symbol_dict, debug=False)

        # Espeak phonemizer (settings GIỐNG inference.py)
        # Trên Windows: cần espeakng_loader để set library path
        if sys.platform.startswith("win"):
            try:
                from phonemizer.backend.espeak.wrapper import EspeakWrapper
                import espeakng_loader
                EspeakWrapper.set_library(espeakng_loader.get_library_path())
            except Exception as e:
                logger.warning(f"Setup espeak-ng trên Windows fail: {e}")
                logger.warning("Đảm bảo đã: pip install espeakng-loader")

        from phonemizer.backend import EspeakBackend
        self._espeak_backend = EspeakBackend(
            language="vi",
            preserve_punctuation=True,
            with_stress=True,
            language_switch="remove-flags",
        )

        # Noise reduce (lazy import — chỉ dùng khi compute_style với denoise > 0)
        self._noisereduce = None

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    def text_to_phoneme(self, text: str) -> str:
        """
        Convert text tiếng Việt -> chuỗi phoneme espeak (đã normalize).

        Pipeline:
            text -> EspeakBackend.phonemize() -> normalize (strip combining, hyphens)
        """
        if not text or not text.strip():
            return ""
        out = self._espeak_backend.phonemize([text])[0]
        out = _normalize_espeak_phoneme(out)
        return out

    def _wave_to_mel(self, wave_np: np.ndarray) -> torch.Tensor:
        """numpy wave -> mel tensor, giống y nguyên preprocess của inference.py."""
        mean, std = -4, 4
        wave_tensor = torch.from_numpy(wave_np).float()
        mel = self._to_mel(wave_tensor)
        mel = (torch.log(1e-5 + mel.unsqueeze(0)) - mean) / std
        return mel

    @torch.no_grad()
    def compute_style(
        self,
        audio_path: Union[str, Path],
        denoise: float = 0.3,
        split_dur: float = 2.0,
        max_seconds: float = 20.0,
    ) -> torch.Tensor:
        """
        Trích style vector từ audio reference.

        Args:
            audio_path:  file .wav/.mp3 reference
            denoise:     0.0-1.0, % blend với noisereduce output (0 = không denoise)
            split_dur:   chia audio thành các đoạn split_dur giây, tính style trung bình.
                         Đặt 0 để dùng full audio. Recommend 2.0 cho audio dài.
            max_seconds: cắt audio nếu > max_seconds (tránh tốn VRAM với file dài)

        Returns:
            style tensor shape (1, style_dim) trên cùng device với model
        """
        import librosa

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio không tồn tại: {audio_path}")

        # Load + resample về 24kHz mono
        wave, _ = librosa.load(str(audio_path), sr=SR, mono=True)

        # Trim silence
        wave, _ = librosa.effects.trim(wave, top_db=30)

        # Cắt nếu quá dài
        max_samples = int(SR * max_seconds)
        if len(wave) > max_samples:
            wave = wave[:max_samples]

        if len(wave) < SR * 0.5:
            raise ValueError(
                f"Audio quá ngắn ({len(wave)/SR:.2f}s sau trim). "
                "Cần >= 0.5s để extract style ổn định."
            )

        # Denoise (lazy import)
        if denoise > 0:
            if self._noisereduce is None:
                import noisereduce as nr
                self._noisereduce = nr
            wave_dn = self._noisereduce.reduce_noise(
                y=wave, sr=SR,
                n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
            )
            wave = wave * (1.0 - denoise) + wave_dn * denoise

        # Compute style — chia thành đoạn nếu split_dur > 0 và audio đủ dài
        if split_dur > 0 and len(wave) / SR >= 4:
            jump = int(SR * split_dur)
            total_len = len(wave)
            mel = self._wave_to_mel(wave[0:jump]).to(self.device)
            ref_s = self.style_encoder(mel.unsqueeze(1))
            count = 1
            for i in range(jump, total_len, jump):
                if i + jump >= total_len:
                    left = (total_len - i) / SR
                    if left >= 1.0:
                        mel = self._wave_to_mel(wave[i:total_len]).to(self.device)
                        ref_s = ref_s + self.style_encoder(mel.unsqueeze(1))
                        count += 1
                    break
                mel = self._wave_to_mel(wave[i:i + jump]).to(self.device)
                ref_s = ref_s + self.style_encoder(mel.unsqueeze(1))
                count += 1
            ref_s = ref_s / count
        else:
            mel = self._wave_to_mel(wave).to(self.device)
            ref_s = self.style_encoder(mel.unsqueeze(1))

        return ref_s

    def synthesize(
        self,
        text_or_phoneme: str,
        style_vector: torch.Tensor,
        speed: float = 1.0,
        t: float = 0.1,
        already_phonemized: bool = False,
        trim_edges_samples: int = 4000,
    ) -> np.ndarray:
        """
        Sinh audio từ text (hoặc phoneme có sẵn) + style vector.

        Args:
            text_or_phoneme:    string đầu vào
            style_vector:       output của compute_style()
            speed:              tốc độ đọc (1.0 = bình thường, 0.8 = chậm, 1.2 = nhanh)
            t:                  noise scale duration (0.0-1.0, 0.1 = stable nhất)
            already_phonemized: True nếu text_or_phoneme đã là chuỗi phoneme rồi
                                (vd: D2 đã phonemize trước, D3 chỉ gọi synthesize)
            trim_edges_samples: cắt N samples đầu/cuối để bỏ silent artifacts
                                (mặc định 4000 = ~167ms, giống inference.py lite-vi)

        Returns:
            numpy array float32 1D, 24kHz mono, normalized peak ~1.0
        """
        # 1) Phonemize nếu cần
        if already_phonemized:
            phoneme = text_or_phoneme
        else:
            phoneme = self.text_to_phoneme(text_or_phoneme)
            if not phoneme:
                raise ValueError(f"Phoneme rỗng cho text: {text_or_phoneme!r}")

        # 2) Inference với try/fallback CPU nếu OOM
        try:
            return self._synthesize_core(phoneme, style_vector, speed, t, trim_edges_samples)
        except torch.cuda.OutOfMemoryError as e:
            if not self.cpu_fallback:
                raise
            logger.warning(
                f"CUDA OOM khi synthesize. Fallback CPU. Câu dài: {len(phoneme)} chars. Lỗi: {e}"
            )
            torch.cuda.empty_cache()
            return self._synthesize_core_cpu(phoneme, style_vector, speed, t, trim_edges_samples)

    @torch.no_grad()
    def _synthesize_core(
        self,
        phoneme: str,
        style_vector: torch.Tensor,
        speed: float,
        t: float,
        trim_edges: int,
    ) -> np.ndarray:
        """Core inference logic — giống __inference của inference.py lite-vi."""
        speed = min(max(speed, 0.0001), 2.0)
        device = self.device

        # word_tokenize + cleaner -> tokens
        phn = " ".join(self._word_tokenize(phoneme))
        tokens = self._cleaner(phn)
        if len(tokens) == 0:
            raise ValueError(
                f"Tokens rỗng sau cleaner. Phoneme có ký tự không trong vocab.\n"
                f"  Phoneme: {phoneme[:100]!r}"
            )
        tokens.insert(0, 0)  # BOS = pad index = 0
        tokens.append(0)     # EOS = pad index = 0
        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = _length_to_mask(input_lengths).to(device)
        s = style_vector.to(device)

        # ===== Forward (autocast nếu FP16) =====
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_fp16 else
            torch.autocast(device_type="cuda", enabled=False) if device == "cuda" else
            torch.autocast(device_type="cpu", enabled=False)
        )

        with autocast_ctx:
            t_en = self.text_encoder(tokens, input_lengths, text_mask)
            d = self.predictor.text_encoder(t_en, s, input_lengths, text_mask)
            x, _ = self.predictor.lstm(d)
            duration = self.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)

        # Stabilize duration (compute ở float32 vì sensitive)
        duration = duration.float()
        dur_stats = torch.empty(duration.shape).normal_(
            mean=duration.mean().item(), std=duration.std().item() + 1e-9
        ).to(device)
        duration = duration * (1 - t) + dur_stats * t
        if duration.shape[1] >= 4:
            duration[:, 1:-2] = _replace_outliers_zscore(duration[:, 1:-2])
        duration = duration / speed

        pred_dur = torch.round(duration.squeeze()).clamp(min=1).long()
        pred_aln_trg = torch.zeros(input_lengths.item(), int(pred_dur.sum().item()))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].item())] = 1
            c_frame += int(pred_dur[i].item())
        alignment = pred_aln_trg.unsqueeze(0).to(device)

        # F0 + N + decode (autocast lại)
        with autocast_ctx:
            en = d.transpose(-1, -2) @ alignment
            F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
            asr = t_en @ alignment
            out = self.decoder(asr, F0_pred, N_pred, s)

        wav = out.squeeze().float().cpu().numpy()

        # Trim silent edges (theo inference.py)
        if trim_edges > 0 and len(wav) > trim_edges * 2:
            wav = wav[trim_edges:-trim_edges]

        # Normalize peak để tránh clipping khi save
        peak = max(np.abs(wav).max(), 1e-9)
        wav = wav / peak * 0.95

        return wav.astype(np.float32)

    def _synthesize_core_cpu(
        self,
        phoneme: str,
        style_vector: torch.Tensor,
        speed: float,
        t: float,
        trim_edges: int,
    ) -> np.ndarray:
        """Fallback: move tất cả về CPU, retry. Chậm hơn ~10-20x nhưng không OOM."""
        original_device = self.device
        try:
            for m in self._modules_dict.values():
                m.cpu()
            self.device = "cpu"
            wav = self._synthesize_core(
                phoneme, style_vector.cpu(), speed, t, trim_edges
            )
            return wav
        finally:
            # Restore device
            for m in self._modules_dict.values():
                m.to(original_device)
            self.device = original_device
            torch.cuda.empty_cache()

    def info(self) -> dict:
        """Trả về thông tin engine (cho UI hiển thị)."""
        return {
            "device": self.device,
            "fp16": self.use_fp16,
            "n_token": self.n_token,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_size_mb": round(self.checkpoint_path.stat().st_size / 1e6, 1),
            "trained_epoch": self.checkpoint_epoch,
            "trained_iters": self.checkpoint_iters,
            "n_params_million": round(
                sum(
                    sum(p.numel() for p in self._modules_dict[k].parameters())
                    for k in INFERENCE_MODULES
                ) / 1e6, 2
            ),
        }


# ============================================================
# Smoke test (chạy độc lập file để verify load + synthesize 1 câu)
# ============================================================
if __name__ == "__main__":
    import argparse
    import time
    import soundfile as sf

    # Config tập trung (xem inference/config_loader.py + inference/inference_config.yaml)
    from config_loader import (
        DEFAULT_CONFIG_PATH, load_config, cfg_value, engine_kwargs, resolve_path,
    )

    parser = argparse.ArgumentParser(description="Smoke test inference engine (D0)")
    # File config YAML — nguồn chính của mọi đường dẫn & tham số
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH),
                        help=f"Path tới inference config YAML (default: {DEFAULT_CONFIG_PATH.name})")
    # ---- Override CLI (None = không truyền -> dùng config -> default) ----
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="[engine.checkpoint] path tới epoch_*.pth")
    parser.add_argument("--repo", type=str, default=None,
                        help="[engine.repo] folder StyleTTS2-lite")
    parser.add_argument("--model-config", type=str, default=None,
                        help="[engine.model_config] config.yaml KIẾN TRÚC model (khác --config)")
    parser.add_argument("--device", type=str, default=None,
                        help="[engine.device] auto | cuda | cpu")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None,
                        help="[engine.use_fp16] --fp16 / --no-fp16")
    parser.add_argument("--ref", type=str, default=None,
                        help="[references.male_ref] audio reference giọng test")
    parser.add_argument("--text", type=str, default=None,
                        help="[smoke.text] câu test")
    parser.add_argument("--out", type=str, default=None,
                        help="[smoke.output] file wav output")
    args = parser.parse_args()

    print("=" * 60)
    print("D0 — Inference Engine SMOKE TEST")
    print("=" * 60)

    # ===== Resolve config (CLI > YAML > default) =====
    cfg = load_config(args.config)
    eng_kwargs = engine_kwargs(cfg, args)
    ref_path = resolve_path(cfg_value(cfg, "references", "male_ref", args.ref))
    text = cfg_value(cfg, "smoke", "text", args.text)
    out_path = resolve_path(cfg_value(cfg, "smoke", "output", args.out))

    engine = StyleTTS2LiteVNInference(**eng_kwargs)

    print("\nEngine info:")
    for k, v in engine.info().items():
        print(f"  {k:20s}: {v}")

    print(f"\nText  : {text}")
    phoneme = engine.text_to_phoneme(text)
    print(f"Phn   : {phoneme}")

    print(f"\nCompute style từ: {ref_path}")
    style = engine.compute_style(str(ref_path), denoise=0.3, split_dur=2.0)
    print(f"  Style shape: {tuple(style.shape)}")

    print(f"\nSynthesize...")
    t0 = time.time()
    wav = engine.synthesize(text, style)
    elapsed = time.time() - t0
    audio_dur = len(wav) / SR
    rtf = elapsed / audio_dur
    print(f"  Wav shape : {wav.shape}  ({audio_dur:.2f}s audio)")
    print(f"  Time      : {elapsed:.2f}s")
    print(f"  RTF       : {rtf:.3f}x  (lower = faster than realtime)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wav, SR)
    print(f"\n✅ Saved: {out_path}")
    print("Mở file để nghe — nếu giọng giống Ngạn → fine-tune thành công.")