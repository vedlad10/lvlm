#!/usr/bin/env python
"""Verification script for LVLM project structure.

Run this to verify that all components are properly set up and can
be imported and execute forward passes.
"""

import sys
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger.info("=" * 60)
    logger.info("Testing Module Imports...")
    logger.info("=" * 60)
    
    try:
        from utils import setup_logger, load_config, Metrics
        logger.info("✓ utils module imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import utils: {e}")
        return False
    
    try:
        from data import TVQADataset, ActivityNetQADataset
        logger.info("✓ data loaders imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import data loaders: {e}")
        return False
    
    try:
        from models import (
            FeatureExtractor,
            TemporalBindingModule,
            CHIMRT,
            AdaptiveDepthController,
            MultimodalVDB,
            LVLM,
        )
        logger.info("✓ all model modules imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import models: {e}")
        return False
    
    return True


def test_model_forward_pass():
    """Test forward pass through LVLM."""
    logger.info("=" * 60)
    logger.info("Testing Model Forward Pass...")
    logger.info("=" * 60)
    
    try:
        from models import LVLM
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Create model
        model = LVLM(
            vocab_size=1000,
            answer_vocab_size=100,
            feature_dim=768,
            hidden_dim=768,
            max_reasoning_hops=3,
            enable_temporal_binding=True,
            enable_adaptive_depth=True,
        ).to(device)
        
        logger.info("✓ LVLM model created")
        
        # Prepare dummy input
        batch_size = 2
        seq_len = 10
        feature_dim = 768
        
        frames = torch.randn(batch_size, seq_len, feature_dim, device=device)
        questions = torch.randint(0, 1000, (batch_size,), device=device)
        frame_lengths = torch.tensor([seq_len, seq_len], device=device)
        
        logger.info(f"Input shapes - frames: {frames.shape}, questions: {questions.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(frames, questions, frame_lengths)
        
        logger.info(f"✓ Model forward pass successful")
        logger.info(f"  - output['answer_logits']: {output['answer_logits'].shape}")
        logger.info(f"  - output['temporal_spans']: {output['temporal_spans'].shape}")
        logger.info(f"  - output['depth_used']: {output['depth_used']}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_temporal_binding():
    """Test temporal binding module specifically."""
    logger.info("=" * 60)
    logger.info("Testing Temporal Binding Module...")
    logger.info("=" * 60)
    
    try:
        from models import TemporalBindingModule
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = TemporalBindingModule(
            feature_dim=768,
            num_layers=2,
            hidden_dim=256,
        ).to(device)
        
        frames = torch.randn(2, 16, 768, device=device)
        memory_nodes, assignments = model(frames)
        
        logger.info(f"✓ Temporal binding successful")
        logger.info(f"  - Input frames: {frames.shape}")
        logger.info(f"  - Memory nodes: {memory_nodes.shape}")
        logger.info(f"  - Compression ratio: {frames.shape[1] / memory_nodes.shape[1]:.2f}x")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Temporal binding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adaptive_depth():
    """Test adaptive depth controller."""
    logger.info("=" * 60)
    logger.info("Testing Adaptive Depth Controller...")
    logger.info("=" * 60)
    
    try:
        from models import AdaptiveDepthController
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        controller = AdaptiveDepthController(
            hidden_dim=768,
            max_hops=5,
            entropy_threshold=0.5,
        ).to(device)
        
        reasoning_state = torch.randn(2, 768, device=device)
        answer_logits = torch.randn(2, 100, device=device)
        
        stop_prob, should_stop, info = controller(reasoning_state, answer_logits, 0)
        
        logger.info(f"✓ Adaptive depth control working")
        logger.info(f"  - Stop probability: {stop_prob.mean().item():.4f}")
        logger.info(f"  - Should stop: {should_stop}")
        logger.info(f"  - Entropy: {info['entropy']:.4f}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Adaptive depth test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading."""
    logger.info("=" * 60)
    logger.info("Testing Configuration Loading...")
    logger.info("=" * 60)
    
    try:
        from utils import load_config
        
        config_path = Path(__file__).parent / "configs" / "experiment.yaml"
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return True  # Skip test
        
        config = load_config(str(config_path))
        logger.info(f"✓ Config loaded successfully")
        logger.info(f"  - Project name: {config.get('project_name')}")
        logger.info(f"  - Batch size: {config.get('training', {}).get('batch_size')}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    logger.info("\n" + "=" * 60)
    logger.info("LVLM Project Verification")
    logger.info("=" * 60 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Config Loading", test_config_loading),
        ("Temporal Binding", test_temporal_binding),
        ("Adaptive Depth", test_adaptive_depth),
        ("Model Forward Pass", test_model_forward_pass),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status} - {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    logger.info("=" * 60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
