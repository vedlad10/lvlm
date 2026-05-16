"""
Quick validation test for LVLM code locally.
Tests imports, GPU, module loading, and dummy data pipeline.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image

print("=" * 80)
print("LVLM CODE VALIDATION TEST")
print("=" * 80)

# ============================================================================
# 1. Check Python version and environment
# ============================================================================
print("\n[1/8] Environment Check")
print(f"  Python: {sys.version}")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("  WARNING: CUDA not available, will run on CPU (slow)")

# ============================================================================
# 2. Test imports
# ============================================================================
print("\n[2/8] Testing Imports")
try:
    from models.llava_clip_encoder import FrozenCLIPEncoder
    print("  ✓ FrozenCLIPEncoder imported")
except Exception as e:
    print(f"  ✗ Failed to import FrozenCLIPEncoder: {e}")
    sys.exit(1)

try:
    from models.llava_projection import VisualProjection
    print("  ✓ VisualProjection imported")
except Exception as e:
    print(f"  ✗ Failed to import VisualProjection: {e}")
    sys.exit(1)

try:
    from models.llava_lvlm import LVLM, build_lora_llama
    print("  ✓ LVLM and build_lora_llama imported")
except Exception as e:
    print(f"  ✗ Failed to import LVLM: {e}")
    sys.exit(1)

try:
    from data.llava_dataset import LLaVADataset, llava_collate_fn
    print("  ✓ LLaVADataset and collate_fn imported")
except Exception as e:
    print(f"  ✗ Failed to import dataset: {e}")
    sys.exit(1)

print("  All imports successful!")

# ============================================================================
# 3. Test dummy data generation
# ============================================================================
print("\n[3/8] Generating Dummy Data")
os.makedirs("./test_data/images", exist_ok=True)

# Create dummy images
print("  Creating 4 dummy images...")
for i in range(4):
    image_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(image_array)
    img.save(f"./test_data/images/image_{i:03d}.jpg")

# Create dummy JSON
dummy_samples = [
    {
        "image": f"image_{i:03d}.jpg",
        "conversations": [
            {"from": "human", "value": "What do you see?"},
            {"from": "gpt", "value": "I see an image with colors."},
        ],
    }
    for i in range(4)
]

with open("./test_data/train_data.json", "w") as f:
    json.dump(dummy_samples, f)

print(f"  ✓ Created 4 dummy images and JSON")

# ============================================================================
# 4. Test CLIP encoder
# ============================================================================
print("\n[4/8] Testing FrozenCLIPEncoder")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_encoder = FrozenCLIPEncoder(model_name="ViT-B/32", device=device)
    print(f"  ✓ CLIP encoder loaded (device={device})")
    
    # Test forward pass
    test_image = Image.open("./test_data/images/image_000.jpg").convert("RGB")
    pixel_values = clip_encoder.preprocess(test_image).unsqueeze(0).to(device)
    with torch.no_grad():
        embeddings = clip_encoder(pixel_values)
    print(f"  ✓ Forward pass successful")
    print(f"    Output shape: {embeddings.shape} (expected: (1, 512))")
    assert embeddings.shape == (1, 512), f"Unexpected shape: {embeddings.shape}"
except Exception as e:
    print(f"  ✗ CLIP encoder test failed: {e}")
    sys.exit(1)

# ============================================================================
# 5. Test Visual Projection
# ============================================================================
print("\n[5/8] Testing VisualProjection")
try:
    projection = VisualProjection(
        clip_dim=512,
        llm_dim=4096,
        num_tokens=32,
    ).to(device)
    print("  ✓ VisualProjection initialized")
    
    # Test forward pass
    with torch.no_grad():
        visual_tokens = projection(embeddings)
    print(f"  ✓ Forward pass successful")
    print(f"    Output shape: {visual_tokens.shape} (expected: (1, 32, 4096))")
    assert visual_tokens.shape == (1, 32, 4096), f"Unexpected shape: {visual_tokens.shape}"
except Exception as e:
    print(f"  ✗ VisualProjection test failed: {e}")
    sys.exit(1)

# ============================================================================
# 6. Test LLaVA Dataset (without LLaMA to save memory)
# ============================================================================
print("\n[6/8] Testing LLaVADataset")
try:
    from transformers import AutoTokenizer
    
    # Load tokenizer (doesn't require GPU)
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    print("  ✓ Tokenizer loaded")
    
    # Create dataset
    dataset = LLaVADataset(
        samples=dummy_samples,
        image_dir="./test_data/images",
        tokenizer=tokenizer,
        clip_preprocess=clip_encoder.preprocess,
        num_visual_tokens=32,
        max_length=512,
    )
    print(f"  ✓ Dataset created with {len(dataset)} samples")
    
    # Get one sample
    sample = dataset[0]
    print(f"  ✓ Sample retrieved")
    print(f"    pixel_values shape: {sample['pixel_values'].shape}")
    print(f"    input_ids shape: {sample['input_ids'].shape}")
    print(f"    labels shape: {sample['labels'].shape}")
except Exception as e:
    print(f"  ✗ LLaVADataset test failed: {e}")
    sys.exit(1)

# ============================================================================
# 7. Test DataLoader and Collator
# ============================================================================
print("\n[7/8] Testing DataLoader & Collator")
try:
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: llava_collate_fn(b, num_visual_tokens=32),
    )
    print(f"  ✓ DataLoader created")
    
    # Get one batch
    batch = next(iter(dataloader))
    print(f"  ✓ Batch retrieved from DataLoader")
    print(f"    batch['pixel_values'] shape: {batch['pixel_values'].shape}")
    print(f"    batch['input_ids'] shape: {batch['input_ids'].shape}")
    print(f"    batch['labels'] shape: {batch['labels'].shape}")
    
    # Verify shapes
    assert batch['pixel_values'].size(0) == 2, "Batch size mismatch"
    assert batch['labels'].size(0) == 2, "Labels batch size mismatch"
    assert batch['labels'].size(1) == 32 + 512, "Labels should include visual tokens (32) + max_length (512)"
    print("  ✓ All shapes verified")
except Exception as e:
    print(f"  ✗ DataLoader test failed: {e}")
    sys.exit(1)

# ============================================================================
# 8. Summary
# ============================================================================
print("\n[8/8] Test Summary")
print("=" * 80)
print("✓ ALL TESTS PASSED!")
print("=" * 80)
print("\nCode validation complete. Ready for:")
print("  1. Full training run on real data")
print("  2. Google Colab deployment")
print("  3. Research paper experiments")
print("\nNext steps:")
print("  - Prepare real LLaVA dataset (JSON + images)")
print("  - Run: python experiments/train_llava.py")
print("  - Or use: notebooks/06_llava_training_colab.ipynb on Colab")
