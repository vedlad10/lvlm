#!/usr/bin/env python
"""Generate research module config variations for experiments.

This script creates experiment configs automatically:
- research_modules_baseline.yaml (all disabled)
- research_modules_router_only.yaml (router enabled)
- research_modules_router_guided.yaml (router + instruction)
- research_modules_with_grounding.yaml (router + grounding loss)

Usage:
    python scripts/generate_configs.py

Output:
    configs/research_modules_baseline.yaml
    configs/research_modules_router_only.yaml
    configs/research_modules_router_guided.yaml
    configs/research_modules_with_grounding.yaml
"""

import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_base_config():
    """Load base research_modules.yaml"""
    config_path = Path(__file__).parent.parent / "configs" / "research_modules.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config, name):
    """Save config to file"""
    output_path = Path(__file__).parent.parent / "configs" / f"research_modules_{name}.yaml"
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"✓ Created {output_path.name}")


def create_baseline():
    """All research modules disabled"""
    config = load_base_config()
    config["research"]["adaptive_router"]["enabled"] = False
    config["research"]["instruction_guided_aggregation"]["enabled"] = False
    config["research"]["grounding"]["enabled"] = False
    config["research"]["temporal_fusion"]["enabled"] = False
    config["research"]["memory_projection"]["enabled"] = False
    config["research"]["dynamic_token_generator"]["enabled"] = False
    config["research"]["unsupported_claim_penalty"]["enabled"] = False
    save_config(config, "baseline")


def create_router_only():
    """Only AdaptiveVisualTokenRouter enabled"""
    config = load_base_config()
    config["research"]["adaptive_router"]["enabled"] = True
    config["research"]["instruction_guided_aggregation"]["enabled"] = False
    config["research"]["grounding"]["enabled"] = False
    config["research"]["temporal_fusion"]["enabled"] = False
    config["research"]["memory_projection"]["enabled"] = False
    config["research"]["dynamic_token_generator"]["enabled"] = False
    config["research"]["unsupported_claim_penalty"]["enabled"] = False
    save_config(config, "router_only")


def create_router_guided():
    """Router + InstructionGuidedVisualAggregator"""
    config = load_base_config()
    config["research"]["adaptive_router"]["enabled"] = True
    config["research"]["instruction_guided_aggregation"]["enabled"] = True
    config["research"]["grounding"]["enabled"] = False
    config["research"]["temporal_fusion"]["enabled"] = False
    config["research"]["memory_projection"]["enabled"] = False
    config["research"]["dynamic_token_generator"]["enabled"] = False
    config["research"]["unsupported_claim_penalty"]["enabled"] = False
    save_config(config, "router_guided")


def create_with_grounding():
    """Router + InstructionGuidedVisualAggregator + VisualTextGroundingHead"""
    config = load_base_config()
    config["research"]["adaptive_router"]["enabled"] = True
    config["research"]["instruction_guided_aggregation"]["enabled"] = True
    config["research"]["grounding"]["enabled"] = True
    config["research"]["temporal_fusion"]["enabled"] = False
    config["research"]["memory_projection"]["enabled"] = False
    config["research"]["dynamic_token_generator"]["enabled"] = False
    config["research"]["unsupported_claim_penalty"]["enabled"] = False
    save_config(config, "with_grounding")


def main():
    """Generate all configs"""
    print("Generating research module config variations...\n")
    
    try:
        create_baseline()
        create_router_only()
        create_router_guided()
        create_with_grounding()
        
        print("\n✓ All configs generated successfully!")
        print("\nExperiment configs created:")
        print("  1. research_modules_baseline.yaml      (All disabled)")
        print("  2. research_modules_router_only.yaml   (Router only)")
        print("  3. research_modules_router_guided.yaml (Router + Guidance)")
        print("  4. research_modules_with_grounding.yaml (Router + Guidance + Grounding)\n")
        
        return 0
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
