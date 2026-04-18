"""
=============================================================================
  PL-BERT — BƯỚC 2: TRAIN PL-BERT TIẾNG VIỆT
=============================================================================
Mục tiêu: Train mô hình PL-BERT từ đầu trên corpus phoneme tiếng Việt
          (all_corpus_phoneme.txt) với Masked Language Modeling (MLM).

          PL-BERT cung cấp contextualized phoneme embeddings cho StyleTTS2,
          giúp mô hình hiểu ngữ nghĩa tiếng Việt ở cấp độ âm vị.

Đầu vào : output/all_corpus_phoneme.txt   (từ C1)
           output/phoneme_vocab.json        (từ A4)

Đầu ra  : checkpoints/plbert_vi_stepXXXX.t7

Chạy lệnh:
    python step2_train_plbert.py --config config.yaml
    python step2_train_plbert.py \
        --corpus  "output/all_corpus_phoneme.txt" \
        --vocab   "output/phoneme_vocab.json"

Tham khảo: Utils/PLBERT/ trong repo gốc yl4579/StyleTTS2
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
from dataclasses import dataclass
from typing import Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv

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

# CONFIGURATION
@dataclass
class PLBERTConfig:
    """Cấu hình cho training PL-BERT tiếng Việt."""

    # --- Đường dẫn ---
    corpus_file: str = ""           # all_corpus_phoneme.txt
    vocab_file: str = ""            # phoneme_vocab.json
    output_dir: str = "./plbert_checkpoints"
    work_dir: str = "./workdir"

    # --- Model Architecture ---
    # Phải khớp với kỳ vọng của StyleTTS2 khi load PL-BERT
    hidden_size: int = 768          # Hidden dimension (StyleTTS2 mặc định)
    num_attention_heads: int = 12   # Số attention heads
    num_hidden_layers: int = 12     # Số transformer layers
    intermediate_size: int = 2048   # FFN intermediate size
    max_position_embeddings: int = 512  # Max sequence length
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # --- Training ---
    epochs: int = 20
    batch_size: int = 64
    max_seq_length: int = 256       # Cắt/pad sequence về độ dài này
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Scheduler
    warmup_steps: int = 1000
    lr_scheduler_type: str = "cosine"  # 'cosine' hoặc 'linear'

    # MLM
    mlm_probability: float = 0.15   # Tỷ lệ mask (chuẩn BERT)

    # --- Hardware ---
    device: str = "cuda"
    num_workers: int = 4
    fp16: bool = False              # Mixed precision (nếu đủ VRAM)

    # --- Logging & Checkpoint ---
    log_interval: int = 100         # Log mỗi N steps
    save_freq: int = 2              # Save checkpoint mỗi N epochs
    save_total_limit: int = 5       # Giữ tối đa N checkpoints

    # --- Val ---
    val_ratio: float = 0.02         # 2% corpus dùng làm validation

    # --- Weights & Biases ---
    wandb_enabled: bool = True          # Bật/tắt wandb logging
    wandb_project: str = "plbert-vietnamese"  # Tên project trên wandb
    wandb_run_name: Optional[str] = None     # Tên phiên chạy (None = wandb tự sinh)
    wandb_entity: Optional[str] = None       # Team/username trên wandb (None = mặc định)
    wandb_tags: Optional[str] = None         # Tags phân loại, cách nhau bằng dấu phẩy
    wandb_notes: str = ""                    # Ghi chú cho phiên chạy

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PLBERTConfig":
        """Load config từ file YAML."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        paths = full_config.get("paths", {})
        plbert = full_config.get("plbert_train", {})

        config = cls()
        # Paths
        config.corpus_file = plbert.get("corpus_file", config.corpus_file)
        config.vocab_file = plbert.get("vocab_file", config.vocab_file)
        config.output_dir = plbert.get("output_dir", config.output_dir)
        config.work_dir = paths.get("work_dir", plbert.get("work_dir", config.work_dir))

        # Model
        config.hidden_size = plbert.get("hidden_size", config.hidden_size)
        config.num_attention_heads = plbert.get("num_attention_heads", config.num_attention_heads)
        config.num_hidden_layers = plbert.get("num_hidden_layers", config.num_hidden_layers)
        config.intermediate_size = plbert.get("intermediate_size", config.intermediate_size)
        config.max_position_embeddings = plbert.get("max_position_embeddings", config.max_position_embeddings)

        # Training
        config.epochs = plbert.get("epochs", config.epochs)
        config.batch_size = plbert.get("batch_size", config.batch_size)
        config.max_seq_length = plbert.get("max_seq_length", config.max_seq_length)
        config.learning_rate = plbert.get("learning_rate", config.learning_rate)
        config.weight_decay = plbert.get("weight_decay", config.weight_decay)
        config.max_grad_norm = plbert.get("max_grad_norm", config.max_grad_norm)
        config.warmup_steps = plbert.get("warmup_steps", config.warmup_steps)
        config.mlm_probability = plbert.get("mlm_probability", config.mlm_probability)

        # Hardware
        config.device = plbert.get("device", config.device)
        config.num_workers = plbert.get("num_workers", config.num_workers)
        config.fp16 = plbert.get("fp16", config.fp16)

        # Logging
        config.log_interval = plbert.get("log_interval", config.log_interval)
        config.save_freq = plbert.get("save_freq", config.save_freq)

        # Wandb
        wb = full_config.get("wandb", plbert.get("wandb", {}))
        if isinstance(wb, dict):
            config.wandb_enabled = wb.get("enabled", config.wandb_enabled)
            config.wandb_project = wb.get("project", config.wandb_project)
            config.wandb_run_name = wb.get("run_name", config.wandb_run_name)
            config.wandb_entity = wb.get("entity", config.wandb_entity)
            config.wandb_tags = wb.get("tags", config.wandb_tags)
            config.wandb_notes = wb.get("notes", config.wandb_notes)

        return config

# LOGGING SETUP
def setup_logging(log_dir: Path) -> logging.Logger:
    """Thiết lập logging ghi ra console + file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "plbert_step2_train.log"

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
    return logging.getLogger("plbert_train")

# VOCABULARY / TOKENIZER
class PhonemeTokenizer:
    """
    Character-level tokenizer cho phoneme IPA.
    Đọc mapping từ phoneme_vocab.json.
    """

    def __init__(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        self.char_to_id = vocab_data["char_to_id"]
        self.id_to_char = {int(k): v for k, v in vocab_data["id_to_char"].items()}
        self.n_token = vocab_data["n_token"]

        # Special token IDs
        self.pad_id = self.char_to_id.get("<pad>", 0)
        self.unk_id = self.char_to_id.get("<unk>", 1)
        self.bos_id = self.char_to_id.get("<bos>", 2)
        self.eos_id = self.char_to_id.get("<eos>", 3)
        self.mask_id = self._ensure_mask_token()

    def _ensure_mask_token(self) -> int:
        """Đảm bảo có <mask> token cho MLM. Thêm vào nếu chưa có."""
        if "<mask>" in self.char_to_id:
            return self.char_to_id["<mask>"]
        # Thêm <mask> token với ID tiếp theo
        mask_id = self.n_token
        self.char_to_id["<mask>"] = mask_id
        self.id_to_char[mask_id] = "<mask>"
        self.n_token += 1
        return mask_id

    def encode(self, text: str, max_length: int = 256, add_special_tokens: bool = True) -> list:
        """Encode chuỗi phoneme thành list token IDs."""
        ids = []
        if add_special_tokens:
            ids.append(self.bos_id)

        for char in text:
            ids.append(self.char_to_id.get(char, self.unk_id))

        if add_special_tokens:
            ids.append(self.eos_id)

        # Truncate
        if len(ids) > max_length:
            ids = ids[:max_length]

        return ids

    def decode(self, ids: list) -> str:
        """Decode list token IDs → chuỗi phoneme."""
        chars = []
        for tid in ids:
            if tid in (self.pad_id, self.bos_id, self.eos_id, self.mask_id):
                continue
            chars.append(self.id_to_char.get(tid, "?"))
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        return self.n_token

# DATASET
class PhonemeMLMDataset(Dataset):
    """
    Dataset cho Masked Language Modeling trên phoneme sequences.
    Mỗi dòng trong corpus file là 1 chuỗi phoneme.
    """

    def __init__(
        self,
        corpus_path: str,
        tokenizer: PhonemeTokenizer,
        max_seq_length: int = 256,
        mlm_probability: float = 0.15,
    ):
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

        # Encode
        input_ids = self.tokenizer.encode(text, max_length=self.max_seq_length)

        # Pad
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_seq_length - len(input_ids)
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
        - Trong đó: 80% thay bằng [MASK], 10% random, 10% giữ nguyên
        """
        labels = input_ids.clone()

        # Chỉ mask các vị trí không phải special token và không phải padding
        special_tokens = {
            self.tokenizer.pad_id,
            self.tokenizer.bos_id,
            self.tokenizer.eos_id,
        }

        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Không mask special tokens và padding
        for i, tid in enumerate(input_ids.tolist()):
            if tid in special_tokens or attention_mask[i] == 0:
                probability_matrix[i] = 0.0

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Labels: -100 cho vị trí không mask (bỏ qua trong loss)
        labels[~masked_indices] = -100

        # 80% → [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_id

        # 10% → random token
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(self.tokenizer.vocab_size, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # 10% còn lại → giữ nguyên (model vẫn phải predict)

        return input_ids, labels

# PL-BERT MODEL
class PLBERTModel(nn.Module):
    """
    BERT model cho phoneme-level masked language modeling.
    Kiến trúc tương thích với Utils/PLBERT trong StyleTTS2 gốc.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 12,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 512,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # Embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=hidden_dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN (ổn định hơn khi train từ đầu)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_hidden_layers,
        )

        # MLM Head
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Linear(hidden_size, vocab_size),
        )

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Khởi tạo trọng số theo chuẩn BERT."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Embeddings
        word_embeds = self.word_embeddings(input_ids)
        pos_embeds = self.position_embeddings(position_ids)
        hidden_states = self.layer_norm(word_embeds + pos_embeds)
        hidden_states = self.dropout(hidden_states)

        # Attention mask cho TransformerEncoder
        # PyTorch TransformerEncoder expects: True = ignore, False = attend
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        # Encode
        hidden_states = self.encoder(
            hidden_states,
            src_key_padding_mask=src_key_padding_mask,
        )

        # MLM prediction
        logits = self.mlm_head(hidden_states)

        # Loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states,
        }

# SCHEDULER
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine scheduler với linear warmup (giống HuggingFace)."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# TRAINING LOOP
def train_plbert(config: PLBERTConfig, logger: logging.Logger):
    """Vòng lặp huấn luyện chính cho PL-BERT."""

    # --- Device ---
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # --- Tokenizer ---
    logger.info(f"Loading vocab: {config.vocab_file}")
    tokenizer = PhonemeTokenizer(config.vocab_file)
    vocab_size = tokenizer.vocab_size
    logger.info(f"Vocab size (incl. <mask>): {vocab_size}")

    # --- Dataset ---
    logger.info(f"Loading corpus: {config.corpus_file}")
    full_dataset = PhonemeMLMDataset(
        corpus_path=config.corpus_file,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        mlm_probability=config.mlm_probability,
    )
    logger.info(f"Corpus size: {len(full_dataset):,} sequences")

    # Split train/val
    val_size = max(1, int(len(full_dataset) * config.val_ratio))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info(f"Train: {train_size:,} | Val: {val_size:,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # --- Model ---
    logger.info("Initializing PL-BERT model...")
    model = PLBERTModel(
        vocab_size=vocab_size,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_hidden_layers=config.num_hidden_layers,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        hidden_dropout_prob=config.hidden_dropout_prob,
        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,} ({total_params / 1e6:.1f}M)")
    logger.info(f"Trainable: {trainable_params:,}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # --- Scheduler ---
    total_steps = len(train_loader) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.warmup_steps, total_steps,
    )
    logger.info(f"Total training steps: {total_steps:,}")
    logger.info(f"Warmup steps: {config.warmup_steps}")

    # --- Mixed Precision ---
    scaler = torch.amp.GradScaler("cuda", enabled=config.fp16)

    # --- Output dir ---
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Lưu config model (để load lại) ---
    model_config = {
        "vocab_size": vocab_size,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_hidden_layers": config.num_hidden_layers,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.max_position_embeddings,
    }
    config_save_path = output_dir / "plbert_config.json"
    with open(config_save_path, "w") as f:
        json.dump(model_config, f, indent=2)

    # --- Wandb Init ---
    use_wandb = config.wandb_enabled and WANDB_AVAILABLE
    if config.wandb_enabled and not WANDB_AVAILABLE:
        logger.warning("wandb chưa được cài đặt! Chạy: pip install wandb")
        logger.warning("Tiếp tục training KHÔNG có wandb logging.")

    if use_wandb:
        # Parse tags từ chuỗi "tag1,tag2" thành list
        tags = None
        if config.wandb_tags:
            tags = [t.strip() for t in config.wandb_tags.split(",") if t.strip()]

        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            entity=config.wandb_entity,
            tags=tags,
            notes=config.wandb_notes or None,
            config={
                # Model
                "vocab_size": vocab_size,
                "hidden_size": config.hidden_size,
                "num_attention_heads": config.num_attention_heads,
                "num_hidden_layers": config.num_hidden_layers,
                "intermediate_size": config.intermediate_size,
                "max_position_embeddings": config.max_position_embeddings,
                "total_params": total_params,
                # Training
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "max_seq_length": config.max_seq_length,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "max_grad_norm": config.max_grad_norm,
                "warmup_steps": config.warmup_steps,
                "mlm_probability": config.mlm_probability,
                "total_steps": total_steps,
                "fp16": config.fp16,
                # Data
                "corpus_size": len(full_dataset),
                "train_size": train_size,
                "val_size": val_size,
            },
        )

        # Theo dõi gradient & weight distributions (log mỗi 500 steps)
        wandb.watch(model, log="all", log_freq=500)
        logger.info(f"Wandb initialized: project='{config.wandb_project}', "
                     f"run='{wandb.run.name}', url={wandb.run.get_url()}")

    # TRAINING LOOP
    logger.info("")
    logger.info("=" * 60)
    logger.info("  BẮT ĐẦU TRAINING PL-BERT")
    logger.info("=" * 60)

    global_step = 0
    best_val_loss = float("inf")
    train_start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            with torch.amp.autocast("cuda", enabled=config.fp16):
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs["loss"]

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # Log
            if global_step % config.log_interval == 0:
                avg_loss = epoch_loss / epoch_steps
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                steps_per_sec = epoch_steps / elapsed if elapsed > 0 else 0

                logger.info(
                    f"  Epoch {epoch}/{config.epochs} | "
                    f"Step {global_step:,} | "
                    f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) | "
                    f"LR: {lr:.2e} | "
                    f"Speed: {steps_per_sec:.1f} steps/s"
                )

                if use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/loss_avg": avg_loss,
                        "train/learning_rate": lr,
                        "train/speed_steps_per_sec": steps_per_sec,
                        "train/epoch": epoch,
                    }, step=global_step)

        # --- End of epoch ---
        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        epoch_elapsed = time.time() - epoch_start

        # --- Validation ---
        val_loss = evaluate(model, val_loader, device, config.fp16)

        logger.info(
            f"\n  Epoch {epoch}/{config.epochs} DONE | "
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
        if epoch % config.save_freq == 0 or epoch == config.epochs:
            ckpt_name = f"plbert_vi_step{global_step:06d}.t7"
            ckpt_path = output_dir / ckpt_name

            save_checkpoint(model, optimizer, scheduler, epoch, global_step, val_loss, ckpt_path)
            logger.info(f"  Saved checkpoint: {ckpt_path}")

            # Track best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_dir / "plbert_vi_best.t7"
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, val_loss, best_path)
                logger.info(f"  New best val loss: {best_val_loss:.4f} → {best_path}")

            # Cleanup old checkpoints
            cleanup_checkpoints(output_dir, config.save_total_limit, logger)

    # --- Training complete ---
    total_elapsed = time.time() - train_start_time

    logger.info("")
    logger.info("=" * 60)
    logger.info("  TRAINING PL-BERT HOÀN TẤT!")
    logger.info("=" * 60)
    logger.info(f"  Tổng thời gian   : {total_elapsed:.0f}s ({total_elapsed / 3600:.1f}h)")
    logger.info(f"  Tổng steps       : {global_step:,}")
    logger.info(f"  Best val loss    : {best_val_loss:.4f}")
    logger.info(f"  Checkpoints tại  : {output_dir}")
    logger.info("")
    logger.info("  Cách sử dụng trong StyleTTS2:")
    logger.info(f"    PLBERT_dir: '{output_dir}'")
    logger.info("    (Trỏ vào config YAML của StyleTTS2)")
    logger.info("=" * 60)

    # --- Wandb Finish ---
    if use_wandb:
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.summary["total_steps"] = global_step
        wandb.summary["total_time_h"] = total_elapsed / 3600
        wandb.finish()
        logger.info("Wandb run finished.")

# EVALUATION
@torch.no_grad()
def evaluate(model, val_loader, device, fp16=False) -> float:
    """Tính validation loss."""
    model.eval()
    total_loss = 0.0
    total_steps = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=fp16):
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]

        total_loss += loss.item()
        total_steps += 1

    model.train()
    return total_loss / max(total_steps, 1)

# CHECKPOINT MANAGEMENT
def save_checkpoint(model, optimizer, scheduler, epoch, step, val_loss, path):
    """Lưu checkpoint theo format .t7 tương thích StyleTTS2."""
    torch.save(
        {
            "net": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss,
        },
        str(path),
    )

def cleanup_checkpoints(output_dir: Path, keep_n: int, logger: logging.Logger):
    """Giữ lại N checkpoints mới nhất + best, xóa cũ."""
    ckpt_files = sorted(
        output_dir.glob("plbert_vi_step*.t7"),
        key=lambda p: p.stat().st_mtime,
    )

    if len(ckpt_files) > keep_n:
        to_delete = ckpt_files[: len(ckpt_files) - keep_n]
        for f in to_delete:
            f.unlink()
            logger.info(f"  Deleted old checkpoint: {f.name}")

# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="PL-BERT — Bước 2: Train PL-BERT tiếng Việt trên corpus phoneme"
    )
    parser.add_argument("--config", "-c", type=str, default="config.yaml")
    parser.add_argument("--corpus", type=str, default=None, help="Override corpus file")
    parser.add_argument("--vocab", type=str, default=None, help="Override vocab file")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    # Load .env
    env_candidates = [Path(".env"), Path("../.env"), Path("../../.env")]
    for env_path in env_candidates:
        if env_path.exists():
            load_dotenv(str(env_path))
            break

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = PLBERTConfig.from_yaml(str(config_path))
    else:
        config = PLBERTConfig()

    # Override từ CLI
    if args.corpus:
        config.corpus_file = args.corpus
    if args.vocab:
        config.vocab_file = args.vocab
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr

    # Validate
    if not config.corpus_file or not Path(config.corpus_file).exists():
        print(f"[LỖI] Corpus file không tồn tại: {config.corpus_file}")
        print("  Dùng --corpus hoặc đặt 'plbert_train.corpus_file' trong config.yaml")
        sys.exit(1)

    if not config.vocab_file or not Path(config.vocab_file).exists():
        print(f"[LỖI] Vocab file không tồn tại: {config.vocab_file}")
        print("  Dùng --vocab hoặc đặt 'plbert_train.vocab_file' trong config.yaml")
        sys.exit(1)

    # Setup logging
    log_dir = Path(config.work_dir) / "logs"
    logger = setup_logging(log_dir)

    # Header
    logger.info("=" * 60)
    logger.info("  PL-BERT — BƯỚC 2: TRAIN PL-BERT TIẾNG VIỆT")
    logger.info("=" * 60)
    logger.info(f"Config          : {config_path}")
    logger.info(f"Corpus          : {config.corpus_file}")
    logger.info(f"Vocab           : {config.vocab_file}")
    logger.info(f"Output dir      : {config.output_dir}")
    logger.info(f"Model           : hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
                f"heads={config.num_attention_heads}")
    logger.info(f"Training        : epochs={config.epochs}, batch={config.batch_size}, "
                f"lr={config.learning_rate}, warmup={config.warmup_steps}")
    logger.info(f"MLM probability : {config.mlm_probability}")
    logger.info(f"Max seq length  : {config.max_seq_length}")
    logger.info(f"FP16            : {config.fp16}")
    logger.info(f"Wandb           : {config.wandb_enabled} "
                f"(project='{config.wandb_project}', run='{config.wandb_run_name or 'auto'}')")

    # Train
    try:
        train_plbert(config, logger)
    except Exception as e:
        logger.exception(f"THẤT BẠI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()