"""
Quick functional test - Run this AFTER installing dependencies
Tests that all modules can be imported and have basic functionality
"""

import sys

def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*80)
    print("LVLM IMPORT & FUNCTIONAL TEST")
    print("="*80 + "\n")
    
    print("[1/6] Testing imports...")
    try:
        from models.llava_clip_encoder import FrozenCLIPEncoder
        print("  ✓ FrozenCLIPEncoder")
    except ImportError as e:
        print(f"  ✗ FrozenCLIPEncoder: {e}")
        return False
    
    try:
        from models.llava_projection import VisualProjection
        print("  ✓ VisualProjection")
    except ImportError as e:
        print(f"  ✗ VisualProjection: {e}")
        return False
    
    try:
        from models.llava_lvlm import LVLM, build_lora_llama
        print("  ✓ LVLM & build_lora_llama")
    except ImportError as e:
        print(f"  ✗ LVLM: {e}")
        return False
    
    try:
        from data.llava_dataset import LLaVADataset, llava_collate_fn
        print("  ✓ LLaVADataset & llava_collate_fn")
    except ImportError as e:
        print(f"  ✗ LLaVADataset: {e}")
        return False
    
    try:
        from experiments.train_llava import TrainingConfig, train_lvlm, build_model
        print("  ✓ TrainingConfig & train_lvlm")
    except ImportError as e:
        print(f"  ✗ train_lvlm: {e}")
        return False
    
    print("\n[2/6] Testing imports for dependencies...")
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except ImportError:
        print("  ✗ torch not installed")
        return False
    
    try:
        import transformers
        print(f"  ✓ transformers")
    except ImportError:
        print("  ✗ transformers not installed")
        return False
    
    try:
        import clip
        print(f"  ✓ clip")
    except ImportError:
        print("  ✗ clip not installed")
        return False
    
    try:
        from peft import LoraConfig
        print(f"  ✓ peft")
    except ImportError:
        print("  ✗ peft not installed")
        return False
    
    print("\n[3/6] Testing device availability...")
    import torch
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available")
        print(f"    Device: {torch.cuda.get_device_name(0)}")
        print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"  ⚠ CUDA not available (will use CPU - training will be slow)")
    
    print("\n[4/6] Testing module instantiation...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = TrainingConfig(device=device)
        print(f"  ✓ TrainingConfig (device={device})")
    except Exception as e:
        print(f"  ✗ TrainingConfig: {e}")
        return False
    
    try:
        proj = VisualProjection(clip_dim=512, llm_dim=4096, num_tokens=32)
        print(f"  ✓ VisualProjection")
    except Exception as e:
        print(f"  ✗ VisualProjection: {e}")
        return False
    
    print("\n[5/6] Testing model shapes...")
    try:
        # Test projection output shape
        x = torch.randn(2, 512)
        output = proj(x)
        assert output.shape == (2, 32, 4096), f"Unexpected shape: {output.shape}"
        print(f"  ✓ Projection layer (2, 512) → (2, 32, 4096)")
    except Exception as e:
        print(f"  ✗ Projection test: {e}")
        return False
    
    print("\n[6/6] Testing data structures...")
    try:
        from transformers import AutoTokenizer
        # This will require HuggingFace auth for LLaMA
        print(f"  ✓ AutoTokenizer available")
    except Exception as e:
        print(f"  ✗ Tokenizer: {e}")
        return False
    
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80)
    print("""
You are ready to:
1. Prepare your LLaVA dataset
2. Run: python experiments/train_llava.py
3. Or use the Colab notebook for training

For quick testing with dummy data:
   python test_lvlm_local.py
""")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
