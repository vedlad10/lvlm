"""Stage 1 Training: Fine-tune Qwen with frozen LVLM features.

This is the first stage where:
- LVLM temporal binding features are FROZEN (no gradient)
- Only Qwen LoRA parameters are trained
- Fastest training, lowest memory requirement
- Good for initial adaptation

Usage:
    python scripts/train_qwen_stage1.py --dataset tvqa --epochs 15
    python scripts/train_qwen_stage1.py --dataset tvqa --epochs 15 --batch_size 8
"""

import subprocess
import sys
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Qwen fine-tuning with frozen LVLM")
    parser.add_argument('--dataset', type=str, default='tvqa', help='Dataset: tvqa, activitynet')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_qwen_stage1',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("STAGE 1: QWEN FINE-TUNING WITH FROZEN LVLM")
    print("=" * 80)
    print(f"Dataset:      {args.dataset}")
    print(f"Epochs:       {args.epochs}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Output dir:   {args.output_dir}")
    print("=" * 80)
    print("📌 LVLM features are FROZEN - only Qwen LoRA parameters train")
    print("📌 Memory efficient - suitable for GPUs with 16GB+ VRAM")
    print("=" * 80 + "\n")
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "experiments" / "train_qwen.py"),
        "--config", "configs/qwen_finetuning.yaml",
        "--dataset", args.dataset,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--output_dir", args.output_dir,
        "--freeze_lvlm",  # Key: freeze LVLM
    ]
    
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
