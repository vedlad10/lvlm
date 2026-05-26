"""Test script for Qwen 2.5 adapter integration with LVLM.

Validates:
1. Imports and dependencies
2. Qwen model loading
3. LoRA configuration
4. Forward pass compatibility
5. Loss computation
"""

import sys
import logging
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test 1: Verify all imports work."""
    logger.info("=" * 80)
    logger.info("TEST 1: Checking imports...")
    logger.info("=" * 80)
    
    try:
        from models.qwen_adapter import QwenAdapter, QwenLVLMFusion
        logger.info("✅ Successfully imported QwenAdapter and QwenLVLMFusion")
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False
    
    try:
        from models import LVLM
        logger.info("✅ Successfully imported LVLM")
    except ImportError as e:
        logger.error(f"❌ LVLM import failed: {e}")
        return False
    
    return True


def test_qwen_loading():
    """Test 2: Load Qwen model with LoRA."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Loading Qwen 2.5 model...")
    logger.info("=" * 80)
    
    try:
        from models.qwen_adapter import QwenAdapter
        
        logger.info("Creating QwenAdapter (this may take 1-2 minutes on first run)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        qwen = QwenAdapter(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            lora_rank=16,
            lora_alpha=32,
            use_4bit=True,
            device=device,
        )
        
        logger.info("✅ Qwen 2.5 loaded successfully with LoRA configuration")
        
        # Check trainable parameters
        trainable = sum(p.numel() for p in qwen.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in qwen.model.parameters())
        logger.info(f"   - Trainable params: {trainable:,} / {total:,}")
        logger.info(f"   - % Trainable: {100 * trainable / total:.2f}%")
        
        return True, qwen, device
    
    except Exception as e:
        logger.error(f"❌ Qwen loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_forward_pass(qwen, device):
    """Test 3: Forward pass through Qwen."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Testing forward pass...")
    logger.info("=" * 80)
    
    try:
        # Prepare dummy input
        batch_size = 2
        seq_len = 128
        
        question_ids = torch.randint(0, 32000, (batch_size, seq_len)).to(device)
        temporal_features = torch.randn(batch_size, 8, 768).to(device)
        memory_nodes = torch.randn(batch_size, 8, 768).to(device)
        
        logger.info(f"Input shapes:")
        logger.info(f"   - question_ids: {question_ids.shape}")
        logger.info(f"   - temporal_features: {temporal_features.shape}")
        logger.info(f"   - memory_nodes: {memory_nodes.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = qwen(
                question_ids=question_ids,
                temporal_features=temporal_features,
                memory_nodes=memory_nodes,
                return_grounding=True,
            )
        
        logger.info("✅ Forward pass successful!")
        logger.info(f"Output keys: {output.keys()}")
        logger.info(f"   - logits shape: {output['logits'].shape}")
        logger.info(f"   - grounding shape: {output['grounding'].shape if output['grounding'] is not None else 'None'}")
        
        return True, output
    
    except Exception as e:
        logger.error(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_loss_computation(qwen, output, device):
    """Test 4: Loss computation."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Testing loss computation...")
    logger.info("=" * 80)
    
    try:
        batch_size = 2
        
        # Create dummy batch targets
        batch = {
            'answers': torch.randint(0, 512, (batch_size,)).to(device),
            'grounding_targets': torch.rand(batch_size, 2).to(device),
        }
        
        # Compute losses
        qa_loss = nn.CrossEntropyLoss()(output['logits'], batch['answers'])
        grounding_loss = nn.SmoothL1Loss()(output['grounding'], batch['grounding_targets'])
        
        total_loss = qa_loss + 0.5 * grounding_loss
        
        logger.info("✅ Loss computation successful!")
        logger.info(f"   - QA loss: {qa_loss.item():.4f}")
        logger.info(f"   - Grounding loss: {grounding_loss.item():.4f}")
        logger.info(f"   - Total loss: {total_loss.item():.4f}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test 5: Load Qwen configuration."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Loading Qwen YAML config...")
    logger.info("=" * 80)
    
    try:
        from utils import load_config
        
        config_path = Path("configs/qwen_finetuning.yaml")
        if not config_path.exists():
            logger.error(f"❌ Config file not found: {config_path}")
            return False
        
        config = load_config(str(config_path))
        
        logger.info("✅ Config loaded successfully!")
        logger.info(f"   - Project: {config.get('project_name')}")
        logger.info(f"   - Epochs: {config.get('training', {}).get('num_epochs')}")
        logger.info(f"   - Learning rate: {config.get('training', {}).get('optimizer', {}).get('lr')}")
        logger.info(f"   - Qwen model: {config.get('qwen', {}).get('model_name')}")
        logger.info(f"   - LoRA rank: {config.get('qwen', {}).get('lora_rank')}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_script_syntax():
    """Test 6: Validate training script syntax."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: Validating training script...")
    logger.info("=" * 80)
    
    try:
        import py_compile
        
        script_path = Path("experiments/train_qwen.py")
        if not script_path.exists():
            logger.error(f"❌ Training script not found: {script_path}")
            return False
        
        py_compile.compile(str(script_path), doraise=True)
        logger.info("✅ Training script syntax valid!")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Training script validation failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("\n\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + "QWEN 2.5 INTEGRATION TEST SUITE".center(78) + "║")
    logger.info("╚" + "=" * 78 + "╝")
    
    results = {
        "Imports": test_imports(),
        "Config Loading": test_config_loading(),
        "Training Script": test_training_script_syntax(),
    }
    
    # Only continue with model tests if imports work
    if results["Imports"]:
        success, qwen, device = test_qwen_loading()
        results["Qwen Loading"] = success
        
        if success:
            success, output = test_forward_pass(qwen, device)
            results["Forward Pass"] = success
            
            if success:
                results["Loss Computation"] = test_loss_computation(qwen, output, device)
    
    # Print summary
    logger.info("\n\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + "TEST SUMMARY".center(78) + "║")
    logger.info("╠" + "=" * 78 + "╣")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"║ {test_name:<40} {status:>35} ║")
    
    logger.info("╚" + "=" * 78 + "╝\n")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED! Ready for fine-tuning.")
        logger.info("\nNext steps:")
        logger.info("1. Stage 1 (freeze LVLM): python experiments/train_qwen.py --config configs/qwen_finetuning.yaml --freeze_lvlm")
        logger.info("2. Stage 2 (joint training): python experiments/train_qwen.py --config configs/qwen_finetuning.yaml --stage 2")
    else:
        logger.error("❌ Some tests failed. Please fix issues before training.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
