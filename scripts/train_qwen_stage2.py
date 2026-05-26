"""Stage 2 Training: Joint fine-tuning of Qwen + LVLM temporal binding.

This is the second stage where:
- LVLM temporal binding features are TRAINABLE (gradients flow)
- Qwen LoRA parameters continue training
- More computationally expensive
- Achieves best performance through end-to-end optimization

Prerequisites:
    1. Stage 1 must be completed and checkpoints available
    2. Load best_qwen_lora checkpoint from Stage 1
    3. Requires more GPU memory (24GB+)

Usage:
    python scripts/train_qwen_stage2.py --dataset tvqa --epochs 10 --resume checkpoints_qwen_stage1/best_qwen_lora
    python scripts/train_qwen_stage2.py --dataset tvqa --epochs 10 --batch_size 12
"""

import subprocess
import sys
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Joint Qwen + LVLM fine-tuning")
    parser.add_argument('--dataset', type=str, default='tvqa', help='Dataset: tvqa, activitynet')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size (reduced from Stage 1)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (reduced for stability)')
    parser.add_argument('--resume', type=str, default='./checkpoints_qwen_stage1/best_qwen_lora',
                       help='Resume from Stage 1 checkpoint')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_qwen_stage2',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("STAGE 2: JOINT QWEN + LVLM FINE-TUNING")
    print("=" * 80)
    print(f"Dataset:      {args.dataset}")
    print(f"Epochs:       {args.epochs}")
    print(f"Batch size:   {args.batch_size} (reduced for memory)")
    print(f"Learning rate: {args.lr} (reduced for stability)")
    print(f"Resume from:  {args.resume}")
    print(f"Output dir:   {args.output_dir}")
    print("=" * 80)
    print("⚠️  WARNING: LVLM features are now TRAINABLE")
    print("⚠️  Requires 24GB+ GPU VRAM - enable gradient checkpointing if OOM")
    print("⚠️  Learning rate is reduced to prevent catastrophic forgetting")
    print("=" * 80 + "\n")
    
    # Check if resume checkpoint exists
    resume_path = Path(args.resume)
    if not resume_path.exists():
        print(f"❌ ERROR: Resume checkpoint not found at {args.resume}")
        print("   Please complete Stage 1 first or provide valid checkpoint path")
        return 1
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "experiments" / "train_qwen.py"),
        "--config", "configs/qwen_finetuning.yaml",
        "--dataset", args.dataset,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--output_dir", args.output_dir,
        "--resume", args.resume,
        # Note: --freeze_lvlm is NOT passed, so LVLM is trainable
    ]
    
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
