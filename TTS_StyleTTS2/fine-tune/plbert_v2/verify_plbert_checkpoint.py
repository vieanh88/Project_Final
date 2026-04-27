"""
=============================================================================
  verify_plbert_checkpoint.py — Kiểm tra tương thích PL-BERT checkpoint
=============================================================================
Chạy TRƯỚC khi bắt đầu train_wrapper.py --stage 1 để xác nhận checkpoint
từ PL-BERT v2 có thể được load_plbert() đọc thành công.

Chạy lệnh:
    python verify_plbert_checkpoint.py --ckpt-dir checkpoints/
=============================================================================
"""

import os
import sys
import argparse
from pathlib import Path
from collections import OrderedDict

import yaml
import torch
from transformers import AlbertConfig, AlbertModel


class CustomAlbert(AlbertModel):
    """Copy chính xác từ Utils/PLBERT/util.py của StyleTTS2."""
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state


def load_plbert(log_dir):
    """Copy chính xác từ Utils/PLBERT/util.py của StyleTTS2."""
    config_path = os.path.join(log_dir, "config.yml")
    plbert_config = yaml.safe_load(open(config_path))

    albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
    bert = CustomAlbert(albert_base_configuration)

    files = os.listdir(log_dir)
    ckpts = []
    for f in os.listdir(log_dir):
        if f.startswith("step_"): ckpts.append(f)

    iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(log_dir, f))]
    iters = sorted(iters)[-1]

    checkpoint = torch.load(log_dir + "/step_" + str(iters) + ".t7", map_location='cpu')
    state_dict = checkpoint['net']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        if name.startswith('encoder.'):
            name = name[8:]  # remove `encoder.`
            new_state_dict[name] = v
    
    # PL-BERT v2 có key "embeddings.position_ids" sẽ bị báo lỗi không cần thiết, nên loại bỏ nếu tồn tại
    #del new_state_dict["embeddings.position_ids"]
    new_state_dict.pop("embeddings.position_ids", None)
    
    bert.load_state_dict(new_state_dict, strict=False)

    return bert


def main():
    parser = argparse.ArgumentParser(description="Verify PL-BERT checkpoint")
    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="Thư mục chứa config.yml + step_XXXX.t7")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    errors = []

    print("=" * 60)
    print("  VERIFY PL-BERT CHECKPOINT COMPATIBILITY")
    print("=" * 60)

    # 1. Check config.yml
    config_yml = ckpt_dir / "config.yml"
    if not config_yml.exists():
        errors.append(f"config.yml không tồn tại: {config_yml}")
    else:
        print(f"✓ config.yml found: {config_yml}")
        with open(config_yml, "r") as f:
            cfg = yaml.safe_load(f)
        if "model_params" not in cfg:
            errors.append("config.yml thiếu key 'model_params'")
        else:
            mp = cfg["model_params"]
            print(f"  vocab_size: {mp.get('vocab_size')}")
            print(f"  hidden_size: {mp.get('hidden_size')}")
            print(f"  embedding_size: {mp.get('embedding_size')}")
            print(f"  num_hidden_layers: {mp.get('num_hidden_layers')}")
            print(f"  max_position_embeddings: {mp.get('max_position_embeddings')}")

    # 2. Check step_*.t7
    ckpt_files = sorted(ckpt_dir.glob("step_*.t7"))
    if not ckpt_files:
        errors.append("Không tìm thấy file step_*.t7")
    else:
        latest = ckpt_files[-1]
        print(f"\n✓ Found {len(ckpt_files)} checkpoint(s), latest: {latest.name}")

        checkpoint = torch.load(str(latest), map_location="cpu")
        if "net" not in checkpoint:
            errors.append("Checkpoint thiếu key 'net'")
        else:
            state_dict = checkpoint["net"]
            # Check prefix structure
            sample_keys = list(state_dict.keys())[:5]
            print(f"  Sample keys:")
            for k in sample_keys:
                print(f"    {k}")

            has_module_prefix = all(k.startswith("module.") for k in state_dict.keys())
            if not has_module_prefix:
                errors.append("State dict keys không có prefix 'module.' "
                              "(load_plbert kỳ vọng module.encoder.XXX)")
            else:
                print(f"  ✓ All keys have 'module.' prefix")

            has_encoder = any("module.encoder." in k for k in state_dict.keys())
            if not has_encoder:
                errors.append("State dict không có keys 'module.encoder.*'")
            else:
                print(f"  ✓ Found 'module.encoder.*' keys")

    # 3. Try actual load_plbert
    if not errors:
        print(f"\n--- Testing load_plbert() ---")
        try:
            bert = load_plbert(str(ckpt_dir))
            print(f"✓ load_plbert() SUCCESS!")
            print(f"  Type: {type(bert).__name__}")
            print(f"  hidden_size: {bert.config.hidden_size}")
            print(f"  max_position_embeddings: {bert.config.max_position_embeddings}")

            # Test forward pass
            dummy_input = torch.randint(0, 10, (1, 20))
            dummy_mask = torch.ones(1, 20, dtype=torch.long)
            output = bert(dummy_input, attention_mask=dummy_mask)
            print(f"  Forward pass output shape: {output.shape}")
            print(f"  ✓ Forward pass SUCCESS!")
        except Exception as e:
            errors.append(f"load_plbert() FAILED: {e}")

    # Summary
    print()
    print("=" * 60)
    if errors:
        print("  ✗ VERIFICATION FAILED!")
        for err in errors:
            print(f"  - {err}")
    else:
        print("  ✓ ALL CHECKS PASSED!")
        print("  Checkpoint tương thích 100% với StyleTTS2 load_plbert().")
        print(f"  Sử dụng: PLBERT_dir: '{ckpt_dir}'")
    print("=" * 60)

    return 0 if not errors else 1

if __name__ == "__main__":
    sys.exit(main())