"""
=============================================================================
  PL-BERT v2 — TRAIN PL-BERT TIẾNG VIỆT (ALBERT ARCHITECTURE)
=============================================================================
Mục tiêu: Train mô hình PL-BERT từ đầu trên corpus phoneme tiếng Việt
          sử dụng kiến trúc AlbertModel (HuggingFace transformers) để
          checkpoint tương thích 100% với Utils/PLBERT/util.py của StyleTTS2.

Kiến trúc:
  - Backbone : AlbertModel (transformers.AlbertConfig)
  - Wrapper  : MultiTaskModel(encoder=AlbertModel, mask_predictor=Linear)
  - Training : Masked Language Modeling (MLM) trên phoneme sequences
  - Checkpoint: step_XXXX.t7 với state_dict prefix "module.encoder.XXX"
  - Config   : config.yml với key "model_params" (AlbertConfig dict)

Đầu vào : all_corpus_phoneme.txt   (từ C1)
           phoneme_vocab.json        (từ A4)

Đầu ra  : checkpoints/config.yml
           checkpoints/step_XXXX.t7

Chạy lệnh:
    python step2_train_plbert_v2.py --config config_step2.yaml

Tham khảo: yl4579/PL-BERT (model.py, train.ipynb)
           yl4579/StyleTTS2 (Utils/PLBERT/util.py)
=============================================================================
"""

import os
import sys
import json
import time
import math
import random
import logging
import argparse
from pathlib import Path
from typing import Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AlbertConfig, AlbertModel

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# KHẮC PHỤC LỖI ENCODING TRÊN WINDOWS
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"

# Tối ưu CUDA memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# =============================================================================
#  MODEL — Giống chính xác kiến trúc PL-BERT gốc (yl4579/PL-BERT/model.py)
# =============================================================================

class MultiTaskModel(nn.Module):
    """
    Wrapper model cho PL-BERT training.
    Tái tạo chính xác cấu trúc yl4579/PL-BERT/model.py:
      - self.encoder = AlbertModel (backbone)
      - self.mask_predictor = Linear (MLM head → dự đoán phoneme bị mask)

    Khi load vào StyleTTS2, util.py chỉ extract self.encoder (AlbertModel).
    mask_predictor bị bỏ qua → không ảnh hưởng inference.

    Lưu ý: PL-BERT gốc có thêm word_predictor cho task dự đoán từ.
    Với tiếng Việt, ta bỏ word_predictor vì không có word-level alignment data.
    Điều này không ảnh hưởng vì load_plbert() chỉ lấy encoder weights.
    """

    def __init__(self, encoder: AlbertModel, num_tokens: int, hidden_size: int = 768):
        super().__init__()
        self.encoder = encoder
        self.mask_predictor = nn.Linear(hidden_size, num_tokens)

    def forward(self, input_ids, attention_mask=None):
        output = self.encoder(input_ids, attention_mask=attention_mask)
        token_pred = self.mask_predictor(output.last_hidden_state)
        return token_pred


# =============================================================================
#  TOKENIZER — Character-level tokenizer từ phoneme_vocab.json
# =============================================================================

class PhonemeTokenizer:
    """
    Character-level tokenizer cho phoneme IPA.
    Đọc mapping từ phoneme_vocab.json (output của A4).
    Thêm [MASK] token nếu chưa có.
    """

    MASK_TOKEN = "[MASK]"

    def __init__(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        self.char_to_id = dict(vocab_data["char_to_id"])
        self.id_to_char = {int(k): v for k, v in vocab_data["id_to_char"].items()}
        self.n_token = vocab_data["n_token"]

        # Special token IDs
        self.pad_id = self.char_to_id.get("<pad>", 0)
        self.unk_id = self.char_to_id.get("<unk>", 1)
        self.bos_id = self.char_to_id.get("<bos>", 2)
        self.eos_id = self.char_to_id.get("<eos>", 3)

        # Thêm [MASK] token cho MLM
        if self.MASK_TOKEN not in self.char_to_id:
            self.mask_id = self.n_token
            self.char_to_id[self.MASK_TOKEN] = self.mask_id
            self.id_to_char[self.mask_id] = self.MASK_TOKEN
            self.n_token += 1
        else:
            self.mask_id = self.char_to_id[self.MASK_TOKEN]

    def encode(self, text: str, max_length: int = 256) -> list:
        """Encode chuỗi phoneme thành list token IDs (không thêm BOS/EOS)."""
        ids = []
        for char in text:
            ids.append(self.char_to_id.get(char, self.unk_id))
        # Truncate
        if len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    @property
    def vocab_size(self) -> int:
        """vocab_size cho AlbertConfig (bao gồm [MASK])."""
        return self.n_token


# =============================================================================
#  DATASET — Phoneme MLM Dataset
# =============================================================================

class PhonemeMLMDataset(Dataset):
    """
    Dataset cho Masked Language Modeling trên phoneme sequences.
    Mỗi dòng trong corpus file là 1 chuỗi phoneme character-level.
    """

    def __init__(self, corpus_path: str, tokenizer: PhonemeTokenizer,
                 max_seq_length: int = 256, mlm_probability: float = 0.15):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability

        # Đọc toàn bộ corpus vào RAM
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]

        # Encode (character-level, không thêm BOS/EOS)
        input_ids = self.tokenizer.encode(text, max_length=self.max_seq_length)

        # Pad
        seq_len = len(input_ids)
        attention_mask = [1] * seq_len
        padding_length = self.max_seq_length - seq_len
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_id] * padding_length
            attention_mask += [0] * padding_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # Apply MLM masking
        input_ids, labels = self._apply_mlm_mask(input_ids, attention_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _apply_mlm_mask(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Chuẩn BERT MLM masking:
        - 15% tokens được chọn để predict
        - 80% thay bằng [MASK], 10% random, 10% giữ nguyên
        """
        labels = input_ids.clone()

        # Chỉ mask các vị trí không phải special token và không phải padding
        special_ids = {self.tokenizer.pad_id, self.tokenizer.bos_id,
                       self.tokenizer.eos_id}

        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        for i, tid in enumerate(input_ids.tolist()):
            if tid in special_ids or attention_mask[i] == 0:
                probability_matrix[i] = 0.0

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Labels: -100 cho vị trí không mask (bỏ qua trong loss)
        labels[~masked_indices] = -100

        # 80% → [MASK]
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_id

        # 10% → random token (trong phạm vi vocab gốc, không bao gồm special tokens)
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices & ~indices_replaced
        )
        # Random từ ID 4 trở đi (bỏ qua pad/unk/bos/eos)
        random_words = torch.randint(4, self.tokenizer.vocab_size,
                                     labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # 10% còn lại → giữ nguyên (model vẫn phải predict)
        return input_ids, labels


# =============================================================================
#  SCHEDULER — Cosine with Linear Warmup
# =============================================================================

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine scheduler với linear warmup (giống HuggingFace)."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
#  EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """Early stopping theo val_loss."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss: float) -> bool:
        """Returns True nếu cần dừng training."""
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    @property
    def improved(self) -> bool:
        return self.counter == 0


# =============================================================================
#  LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    """Thiết lập logging ghi ra console + file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "plbert_v2_train.log"

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), encoding="utf-8", mode="w"),
        ],
    )
    return logging.getLogger("plbert_v2_train")


# =============================================================================
#  TRAINING
# =============================================================================

def train_plbert(config: dict, logger: logging.Logger):
    """Vòng lặp huấn luyện chính cho PL-BERT v2."""

    cfg = config["plbert_train"]
    wandb_cfg = config.get("wandb", {})

    # --- Device ---
    device = torch.device(cfg.get("device", "cuda")
                          if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Tokenizer ---
    vocab_file = cfg["vocab_file"]
    logger.info(f"Loading vocab: {vocab_file}")
    tokenizer = PhonemeTokenizer(vocab_file)
    vocab_size = tokenizer.vocab_size  # 150 + 1([MASK]) = 151
    logger.info(f"Vocab size (incl. [MASK]): {vocab_size}")

    # --- Dataset ---
    corpus_file = cfg["corpus_file"]
    max_seq_length = cfg.get("max_seq_length", 256)
    mlm_probability = cfg.get("mlm_probability", 0.15)

    logger.info(f"Loading corpus: {corpus_file}")
    full_dataset = PhonemeMLMDataset(
        corpus_path=corpus_file,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        mlm_probability=mlm_probability,
    )
    logger.info(f"Corpus size: {len(full_dataset):,} sequences")

    # Split train/val
    val_ratio = cfg.get("val_ratio", 0.02)
    val_size = max(1, int(len(full_dataset) * val_ratio))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info(f"Train: {train_size:,} | Val: {val_size:,}")

    batch_size = cfg.get("batch_size", 64)
    num_workers = cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # --- AlbertConfig (PHẢI khớp với AlbertConfig mà load_plbert sẽ dùng) ---
    hidden_size = cfg.get("hidden_size", 768)
    model_params = {
        "vocab_size": vocab_size,
        "embedding_size": cfg.get("embedding_size", 128),
        "hidden_size": hidden_size,
        "num_attention_heads": cfg.get("num_attention_heads", 12),
        "num_hidden_layers": cfg.get("num_hidden_layers", 12),
        "num_hidden_groups": cfg.get("num_hidden_groups", 1),
        "intermediate_size": cfg.get("intermediate_size", 2048),
        "hidden_act": "gelu_new",
        "hidden_dropout_prob": cfg.get("hidden_dropout_prob", 0.1),
        "attention_probs_dropout_prob": cfg.get("attention_probs_dropout_prob", 0.1),
        "max_position_embeddings": cfg.get("max_position_embeddings", 512),
        "type_vocab_size": 1,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
    }

    albert_config = AlbertConfig(**model_params)
    albert = AlbertModel(albert_config)

    # Wrap trong MultiTaskModel (giống PL-BERT gốc)
    model = MultiTaskModel(
        encoder=albert,
        num_tokens=vocab_size,
        hidden_size=hidden_size,
    )

    # Wrap trong nn.DataParallel để state_dict có prefix "module."
    # load_plbert() kỳ vọng: module.encoder.XXX → strip module. → strip encoder.
    model = nn.DataParallel(model)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,} ({total_params / 1e6:.1f}M)")
    logger.info(f"Trainable: {trainable_params:,}")

    # --- Optimizer ---
    lr = cfg.get("learning_rate", 1e-4)
    weight_decay = cfg.get("weight_decay", 0.01)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )

    # --- Scheduler ---
    epochs = cfg.get("epochs", 30)
    total_steps = len(train_loader) * epochs
    warmup_steps = cfg.get("warmup_steps", 2000)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    logger.info(f"Total training steps: {total_steps:,}")
    logger.info(f"Warmup steps: {warmup_steps}")

    # --- Mixed Precision ---
    fp16 = cfg.get("fp16", True)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)

    # --- Early Stopping ---
    patience = cfg.get("early_stop_patience", 5)
    min_delta = cfg.get("early_stop_min_delta", 0.0005)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    logger.info(f"Early stopping: patience={patience}, min_delta={min_delta}")

    # --- Output dir ---
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Lưu config.yml cho load_plbert() ---
    # ĐÂY LÀ FILE QUAN TRỌNG NHẤT: load_plbert() đọc config.yml → AlbertConfig
    config_yml = {"model_params": model_params}
    config_yml_path = output_dir / "config.yml"
    with open(config_yml_path, "w", encoding="utf-8") as f:
        yaml.dump(config_yml, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"Saved config.yml (for load_plbert): {config_yml_path}")

    # --- Resume from checkpoint ---
    start_step = 0
    ckpt_files = sorted(output_dir.glob("step_*.t7"))
    if ckpt_files:
        latest_ckpt = ckpt_files[-1]
        logger.info(f"Resuming from checkpoint: {latest_ckpt}")
        checkpoint = torch.load(str(latest_ckpt), map_location="cpu")
        model.load_state_dict(checkpoint["net"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint.get("step", 0)
        logger.info(f"  Resumed at step {start_step}")

    # --- Wandb Init ---
    use_wandb = wandb_cfg.get("enabled", True) and WANDB_AVAILABLE
    if wandb_cfg.get("enabled", True) and not WANDB_AVAILABLE:
        logger.warning("wandb chưa cài đặt! Chạy: pip install wandb")
        logger.warning("Tiếp tục training KHÔNG có wandb logging.")

    if use_wandb:
        tags = None
        tags_str = wandb_cfg.get("tags", "")
        if tags_str:
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]

        wandb.init(
            project=wandb_cfg.get("project", "story-ai-narrator"),
            name=wandb_cfg.get("run_name", "albert-vi-base-ep30-bs64"),
            entity=wandb_cfg.get("entity", None),
            tags=tags,
            notes=wandb_cfg.get("notes", ""),
            config={
                # Model
                **model_params,
                "total_params": total_params,
                # Training
                "epochs": epochs,
                "batch_size": batch_size,
                "max_seq_length": max_seq_length,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "max_grad_norm": max_grad_norm,
                "warmup_steps": warmup_steps,
                "mlm_probability": mlm_probability,
                "total_steps": total_steps,
                "fp16": fp16,
                "early_stop_patience": patience,
                "early_stop_min_delta": min_delta,
                # Data
                "corpus_size": len(full_dataset),
                "train_size": train_size,
                "val_size": val_size,
            },
        )
        wandb.watch(model, log="all", log_freq=500)
        logger.info(f"Wandb initialized: project='{wandb_cfg.get('project')}', "
                     f"run='{wandb.run.name}', url={wandb.run.get_url()}")

    # =================================================================
    #  TRAINING LOOP
    # =================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("  BẮT ĐẦU TRAINING PL-BERT v2 (ALBERT)")
    logger.info("=" * 60)

    global_step = start_step
    best_val_loss = float("inf")
    train_start_time = time.time()

    log_interval = cfg.get("log_interval", 200)
    save_freq = cfg.get("save_freq", 2)
    save_total_limit = cfg.get("save_total_limit", 5)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            with torch.amp.autocast("cuda", enabled=fp16):
                logits = model(input_ids, attention_mask)
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    labels.view(-1),
                    ignore_index=-100,
                )

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # Log
            if global_step % log_interval == 0:
                avg_loss = epoch_loss / epoch_steps
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                steps_per_sec = epoch_steps / elapsed if elapsed > 0 else 0

                logger.info(
                    f"  Epoch {epoch}/{epochs} | "
                    f"Step {global_step:,} | "
                    f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) | "
                    f"LR: {current_lr:.2e} | "
                    f"Speed: {steps_per_sec:.1f} steps/s"
                )

                if use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/loss_avg": avg_loss,
                        "train/learning_rate": current_lr,
                        "train/speed_steps_per_sec": steps_per_sec,
                        "train/epoch": epoch,
                    }, step=global_step)

        # --- End of epoch ---
        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        epoch_elapsed = time.time() - epoch_start

        # --- Validation ---
        val_loss = evaluate(model, val_loader, device, vocab_size, fp16)

        logger.info(
            f"\n  Epoch {epoch}/{epochs} DONE | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {epoch_elapsed:.0f}s"
        )

        if use_wandb:
            wandb.log({
                "epoch/train_loss": avg_train_loss,
                "epoch/val_loss": val_loss,
                "epoch/best_val_loss": min(best_val_loss, val_loss),
                "epoch/duration_s": epoch_elapsed,
                "epoch/epoch": epoch,
            }, step=global_step)

        # --- Save checkpoint ---
        if epoch % save_freq == 0 or epoch == epochs:
            save_checkpoint(model, optimizer, global_step, output_dir)
            logger.info(f"  Saved: step_{global_step}.t7")

            # Track best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, global_step, output_dir,
                                filename="plbert_vi_best.t7")
                logger.info(f"  ★ New best val loss: {best_val_loss:.4f}")

            # Cleanup old checkpoints
            cleanup_checkpoints(output_dir, save_total_limit, logger)

        # --- Early Stopping ---
        if early_stopping(val_loss):
            logger.info(f"\n  ⚠ Early stopping triggered! "
                        f"No improvement for {patience} epochs.")
            logger.info(f"  Best val loss: {early_stopping.best_loss:.4f}")
            break

    # --- Training complete ---
    total_elapsed = time.time() - train_start_time

    logger.info("")
    logger.info("=" * 60)
    logger.info("  TRAINING PL-BERT v2 HOÀN TẤT!")
    logger.info("=" * 60)
    logger.info(f"  Tổng thời gian   : {total_elapsed:.0f}s ({total_elapsed / 3600:.1f}h)")
    logger.info(f"  Tổng steps       : {global_step:,}")
    logger.info(f"  Best val loss    : {best_val_loss:.4f}")
    logger.info(f"  Checkpoints tại  : {output_dir}")
    logger.info("")
    logger.info("  Cách sử dụng trong StyleTTS2:")
    logger.info(f"    PLBERT_dir: '{output_dir}'")
    logger.info("    (config.yml + step_XXXX.t7 đã sẵn sàng cho load_plbert())")
    logger.info("=" * 60)

    if use_wandb:
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.summary["total_steps"] = global_step
        wandb.summary["total_time_h"] = total_elapsed / 3600
        wandb.finish()
        logger.info("Wandb run finished.")


# =============================================================================
#  EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(model, val_loader, device, vocab_size, fp16=False) -> float:
    """Tính validation loss."""
    model.eval()
    total_loss = 0.0
    total_steps = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=fp16):
            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        total_loss += loss.item()
        total_steps += 1

    model.train()
    return total_loss / max(total_steps, 1)


# =============================================================================
#  CHECKPOINT MANAGEMENT
# =============================================================================

def save_checkpoint(model, optimizer, step, output_dir, filename=None):
    """
    Lưu checkpoint theo format tương thích với load_plbert().

    Format checkpoint:
      - key "net": model.state_dict()
        → state_dict có prefix "module." (từ DataParallel)
        → Cấu trúc: module.encoder.XXX, module.mask_predictor.XXX
      - key "step": global step number
      - key "optimizer": optimizer state

    load_plbert() sẽ:
      1. Lấy checkpoint["net"]
      2. Strip "module." → "encoder.XXX", "mask_predictor.XXX"
      3. Chỉ lấy keys bắt đầu bằng "encoder." → strip → AlbertModel weights
    """
    if filename is None:
        filename = f"step_{step}.t7"

    state = {
        "net": model.state_dict(),
        "step": step,
        "optimizer": optimizer.state_dict(),
    }

    save_path = Path(output_dir) / filename
    torch.save(state, str(save_path))


def cleanup_checkpoints(output_dir: Path, keep_n: int,
                        logger: logging.Logger):
    """Giữ lại N checkpoints mới nhất (step_*.t7), không xóa best."""
    ckpt_files = sorted(
        output_dir.glob("step_*.t7"),
        key=lambda p: p.stat().st_mtime,
    )
    if len(ckpt_files) > keep_n:
        to_delete = ckpt_files[:len(ckpt_files) - keep_n]
        for f in to_delete:
            f.unlink()
            logger.info(f"  Deleted old checkpoint: {f.name}")


# =============================================================================
#  CONFIG LOADING
# =============================================================================

def load_config(yaml_path: str) -> dict:
    """Load config từ YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PL-BERT v2 — Train PL-BERT tiếng Việt (ALBERT architecture)"
    )
    parser.add_argument("--config", "-c", type=str, default="config_step2.yaml",
                        help="Config YAML file")
    parser.add_argument("--corpus", type=str, default=None,
                        help="Override corpus file")
    parser.add_argument("--vocab", type=str, default=None,
                        help="Override vocab file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        print(f"[LỖI] Config file không tồn tại: {config_path}")
        sys.exit(1)

    cfg = config.get("plbert_train", {})

    # Override từ CLI
    if args.corpus:
        cfg["corpus_file"] = args.corpus
    if args.vocab:
        cfg["vocab_file"] = args.vocab
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.lr:
        cfg["learning_rate"] = args.lr

    config["plbert_train"] = cfg

    # Validate
    if not cfg.get("corpus_file") or not Path(cfg["corpus_file"]).exists():
        print(f"[LỖI] Corpus file không tồn tại: {cfg.get('corpus_file', '')}")
        sys.exit(1)
    if not cfg.get("vocab_file") or not Path(cfg["vocab_file"]).exists():
        print(f"[LỖI] Vocab file không tồn tại: {cfg.get('vocab_file', '')}")
        sys.exit(1)

    # Setup logging
    work_dir = config.get("paths", {}).get("work_dir", cfg.get("output_dir", "."))
    log_dir = Path(work_dir) / "logs"
    logger = setup_logging(log_dir)

    # Header
    logger.info("=" * 60)
    logger.info("  PL-BERT v2 — TRAIN PL-BERT TIẾNG VIỆT (ALBERT)")
    logger.info("=" * 60)
    logger.info(f"Config          : {config_path}")
    logger.info(f"Corpus          : {cfg['corpus_file']}")
    logger.info(f"Vocab           : {cfg['vocab_file']}")
    logger.info(f"Output dir      : {cfg['output_dir']}")
    logger.info(f"Model           : ALBERT hidden={cfg.get('hidden_size', 768)}, "
                f"layers={cfg.get('num_hidden_layers', 12)}, "
                f"heads={cfg.get('num_attention_heads', 12)}, "
                f"embed={cfg.get('embedding_size', 128)}")
    logger.info(f"Training        : epochs={cfg.get('epochs', 30)}, "
                f"batch={cfg.get('batch_size', 64)}, "
                f"lr={cfg.get('learning_rate', 1e-4)}, "
                f"warmup={cfg.get('warmup_steps', 2000)}")
    logger.info(f"MLM probability : {cfg.get('mlm_probability', 0.15)}")
    logger.info(f"FP16            : {cfg.get('fp16', True)}")
    logger.info(f"Early stopping  : patience={cfg.get('early_stop_patience', 5)}, "
                f"min_delta={cfg.get('early_stop_min_delta', 0.0005)}")

    # Train
    try:
        train_plbert(config, logger)
    except Exception as e:
        logger.exception(f"THẤT BẠI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()