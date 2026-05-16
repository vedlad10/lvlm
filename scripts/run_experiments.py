#!/usr/bin/env python
"""Experiment runner orchestrator for LVLM research modules.

Automatically runs a sequence of experiments (baseline, router, router+guidance, etc.)
with consistent settings and logs results.

Usage (Local):
    python scripts/run_experiments.py --dataset clip --epochs 10 --batch-size 32

Usage (Lightning AI):
    lightning run model scripts/run_experiments.py --dataset clip --epochs 20
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates sequence of experiments"""
    
    EXPERIMENTS = {
        "baseline": {
            "config": "configs/research_modules_baseline.yaml",
            "description": "Baseline (no research modules)",
            "name": "exp_baseline",
        },
        "router_only": {
            "config": "configs/research_modules_router_only.yaml",
            "description": "AdaptiveVisualTokenRouter only",
            "name": "exp_router_only",
        },
        "router_guided": {
            "config": "configs/research_modules_router_guided.yaml",
            "description": "Router + InstructionGuidedAggregator",
            "name": "exp_router_guided",
        },
        "with_grounding": {
            "config": "configs/research_modules_with_grounding.yaml",
            "description": "Router + Guidance + Grounding Head",
            "name": "exp_with_grounding",
        },
    }
    
    def __init__(
        self,
        dataset: str = "clip",
        epochs: int = 10,
        batch_size: int = 32,
        experiments: List[str] = None,
        use_lightning: bool = False,
        no_wandb: bool = False,
    ):
        """Initialize experiment runner.
        
        Args:
            dataset: Dataset to use
            epochs: Number of training epochs
            batch_size: Training batch size
            experiments: List of experiment names to run (default: all)
            use_lightning: Use lightning run command
            no_wandb: Disable WandB logging
        """
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.experiments = experiments or list(self.EXPERIMENTS.keys())
        self.use_lightning = use_lightning
        self.no_wandb = no_wandb
        self.results = {}
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    
    def run_experiment(self, exp_key: str) -> bool:
        """Run single experiment"""
        exp = self.EXPERIMENTS[exp_key]
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Starting: {exp['description']}")
        logger.info(f"Name: {exp['name']}")
        logger.info(f"Config: {exp['config']}")
        logger.info(f"{'='*70}\n")
        
        # Build command
        if self.use_lightning:
            # Use python -m pytorch_lightning instead of lightning CLI
            cmd = ["python", "experiments/train_lightning.py"]
        else:
            cmd = ["python", "experiments/train_lightning.py"]
        
        cmd.extend([
            "--config", "configs/experiment.yaml",
            "--research-config", exp["config"],
            "--dataset", self.dataset,
            "--epochs", str(self.epochs),
            "--batch-size", str(self.batch_size),
            "--name", exp["name"],
        ])
        
        if self.no_wandb:
            cmd.append("--no-wandb")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                cwd=Path(__file__).parent.parent,
            )
            
            logger.info(f"✓ {exp['description']} completed successfully\n")
            self.results[exp_key] = {"status": "completed", "exit_code": 0}
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {exp['description']} failed with exit code {e.returncode}\n")
            self.results[exp_key] = {"status": "failed", "exit_code": e.returncode}
            return False
    
    def run_all(self) -> int:
        """Run all selected experiments"""
        logger.info(f"\n{'='*70}")
        logger.info("LVLM Research Modules Experiment Runner")
        logger.info(f"{'='*70}")
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Experiments: {', '.join(self.experiments)}")
        logger.info(f"{'='*70}\n")
        
        start_time = datetime.now()
        completed = 0
        failed = 0
        
        for exp_key in self.experiments:
            if exp_key not in self.EXPERIMENTS:
                logger.warning(f"Unknown experiment: {exp_key}")
                continue
            
            if self.run_experiment(exp_key):
                completed += 1
            else:
                failed += 1
        
        # Summary
        elapsed = datetime.now() - start_time
        logger.info(f"\n{'='*70}")
        logger.info("EXPERIMENT SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Completed: {completed}/{len(self.experiments)}")
        logger.info(f"Failed: {failed}/{len(self.experiments)}")
        logger.info(f"Total time: {elapsed}")
        logger.info(f"{'='*70}\n")
        
        # Save results
        self.save_results(elapsed)
        
        return 0 if failed == 0 else 1
    
    def save_results(self, total_time):
        """Save experiment results"""
        results_file = Path(__file__).parent.parent / "EXPERIMENT_RESULTS.json"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_time": str(total_time),
            "dataset": self.dataset,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "experiments": self.results,
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run LVLM research module experiments"
    )
    parser.add_argument("--dataset", default="clip", help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["baseline", "router_only", "router_guided"],
        help="Experiments to run",
    )
    parser.add_argument("--lightning", action="store_true", help="Use lightning run")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        experiments=args.experiments,
        use_lightning=args.lightning,
        no_wandb=args.no_wandb,
    )
    
    return runner.run_all()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
