# LVLM Research Modules Integration Guide

**Last Updated:** May 15, 2026  
**Timeline:** 2-3 Days to Complete  
**Target:** Integrate research modules, run baseline + ablation experiments, start paper

---

## 📋 Quick Summary

You have **7 research modules** for visual token optimization. The plan:

| Day | Focus | Modules | Output |
|-----|-------|---------|--------|
| **Day 1** | Core Integration | Router (PRIMARY) | Code ready, smoke tests pass |
| **Day 2** | Ablation Experiments | Router + Instruction Guidance | Metrics & comparisons |
| **Day 3** | Analysis & Paper Prep | All (if time) | Results tables, paper outline |

---

## 🗂️ Files Created/Modified

### New Files
```
models/research_modules.py              ← 7 research modules + docstrings
configs/research_modules.yaml           ← Config for enabling/disabling modules
experiments/train_lightning.py          ← Lightning AI trainer (replaces train.py)
experiments/test_research_modules.py   ← Smoke tests for all modules
```

### Updated Files (no changes needed yet)
```
models/llava_lvlm.py                   ← Will integrate router in projection layer
experiments/train.py                   ← Keep for backward compatibility
configs/experiment.yaml                ← Add loss weights for research modules
```

---

## 🎯 Module Hierarchy & Usage

### PRIMARY (Implement Today - Day 1)
**AdaptiveVisualTokenRouter** — Learns which visual tokens are important
- Reduces from fixed 32 tokens → variable 8-32 tokens per question
- Uses query embeddings to decide token importance
- Expected: 10-20% inference speedup, <1% accuracy drop
- **Config flag:** `research.adaptive_router.enabled`

### SECONDARY (Day 1-2)
**InstructionGuidedVisualAggregator** — Task-specific token selection
- Reads question → decides which visual regions matter
- OCR tasks attend to text regions, counting to objects, etc.
- Cascade with router for better efficiency
- **Config flag:** `research.instruction_guided_aggregation.enabled`

### OPTIONAL (Day 2+)
**VisualTextGroundingHead** — Reduce hallucinations via contrastive loss
- Pulls answer embeddings closer to visual tokens
- Optional auxiliary loss (lambda=0.05)
- **Config flag:** `research.grounding.enabled`

### SKIP FOR NOW (Post-Experiments)
- **TemporalFrameFusion** — Only for video datasets
- **MemoryAwareProjection** — Complex; test after core works
- **DynamicVisualTokenGenerator** — Alternative to router (choose one)
- **UnsupportedClaimPenalty** — Requires negative example pairs

---

## 🚀 Day-by-Day Execution Plan

### DAY 1: Integration & Validation (8 hours)

#### Hour 1-2: Verify All Files & Run Smoke Tests
```bash
# Check files exist
ls models/research_modules.py
ls configs/research_modules.yaml
ls experiments/train_lightning.py
ls experiments/test_research_modules.py

# Run smoke tests (should all pass)
cd experiments
python test_research_modules.py
```

**Expected Output:**
```
Running research module smoke tests...

✓ AdaptiveVisualTokenRouter passed
✓ TemporalFrameFusion passed
✓ InstructionGuidedVisualAggregator passed
✓ MemoryAwareProjection passed
✓ DynamicVisualTokenGenerator passed
✓ VisualTextGroundingHead passed
✓ UnsupportedClaimPenalty passed

✓ All smoke tests passed!
```

#### Hour 2-3: Update Model Imports
Edit `models/__init__.py` to expose research modules:
```python
from .research_modules import (
    AdaptiveVisualTokenRouter,
    InstructionGuidedVisualAggregator,
    VisualTextGroundingHead,
    # ... other modules
)
```

#### Hour 3-4: Integrate Router into Projection Layer
In `models/llava_projection.py`, modify forward pass:
```python
def forward(self, clip_embeds, query_embed=None):
    # Existing projection
    visual_tokens = self.projection_mlp(clip_embeds)  # [batch, 32, dim]
    
    # Optional: Apply router if enabled
    if self.use_router and query_embed is not None:
        routed, mask, stats = self.router(visual_tokens, query_embed)
        # routed: [batch, variable_k, dim]
        # mask: [batch, variable_k]
        self.router_stats = stats
        return routed, mask
    else:
        return visual_tokens  # [batch, 32, dim], no mask
```

#### Hour 4-5: Create Config Variations
```bash
# Copy and modify research_modules.yaml
cp configs/research_modules.yaml configs/research_modules_baseline.yaml
cp configs/research_modules.yaml configs/research_modules_router_only.yaml
cp configs/research_modules.yaml configs/research_modules_router_guided.yaml
```

In `research_modules_baseline.yaml`: Set all `enabled: false`  
In `research_modules_router_only.yaml`: Set `adaptive_router.enabled: true`  
In `research_modules_router_guided.yaml`: Set both router and instruction_guided to `true`

#### Hour 5-6: Create Lightning AI Project
```bash
# Install Lightning CLI
pip install lightning

# Create project on Lightning AI
lightning run model experiments/train_lightning.py \
    --config ../configs/experiment.yaml \
    --research-config ../configs/research_modules_router_only.yaml \
    --epochs 1 --batch-size 4

# This will run 1 test epoch locally to verify setup
```

#### Hour 6-8: Documentation & Backup
- Create experiment log file
- Document any integration issues
- Commit code to version control
- **Deliverable:** All code integrated, smoke tests pass, 1 epoch runs

---

### DAY 2: Ablation Experiments (8 hours)

#### Hour 1-3: Run Baseline (Router Disabled)
```bash
lightning run model experiments/train_lightning.py \
    --config ../configs/experiment.yaml \
    --research-config ../configs/research_modules_baseline.yaml \
    --epochs 10 --batch-size 32 \
    --name exp_baseline
```
**Metrics to track:**
- Training loss
- Validation loss
- Inference speed (tokens/sec)
- Effective batch throughput

#### Hour 3-6: Run Router Ablation
```bash
lightning run model experiments/train_lightning.py \
    --config ../configs/experiment.yaml \
    --research-config ../configs/research_modules_router_only.yaml \
    --epochs 10 --batch-size 32 \
    --name exp_router_only
```
**Compare vs baseline:**
- Speedup: (baseline_time / router_time)
- Token count: mean(kept_tokens) vs 32
- Accuracy impact: |val_loss_baseline - val_loss_router|

#### Hour 6-8: Run Router + Instruction Guidance
```bash
lightning run model experiments/train_lightning.py \
    --config ../configs/experiment.yaml \
    --research-config ../configs/research_modules_router_guided.yaml \
    --epochs 10 --batch-size 32 \
    --name exp_router_guided
```
**Compare vs both:**
- Speedup vs baseline
- Accuracy vs baseline
- Better than router alone?

**Deliverable:** 3 experiment runs with metrics, comparison table

---

### DAY 3: Analysis & Paper Preparation (4 hours)

#### Hour 1-2: Generate Results & Plots
```bash
# Use Lightning Studio to export results
# Create comparison table (CSV):
# exp_name | avg_loss | speedup | tokens_kept | accuracy

# Generate plots:
# - Loss curves (3 experiments)
# - Token count distribution
# - Speed vs accuracy tradeoff
```

#### Hour 2-3: Prepare Paper Outline
Create `PAPER_OUTLINE.md`:
```markdown
# Paper Title: Efficient Vision-Language Understanding via Adaptive Token Routing

## Sections
1. **Introduction** — Motivation for visual token efficiency
2. **Related Work** — Token pruning, vision-language models
3. **Method**
   - AdaptiveVisualTokenRouter architecture
   - Query-conditioned importance scoring
4. **Experiments**
   - Baseline vs Router vs Router+Guidance
   - Speedup measurements
   - Accuracy preservation
5. **Results** — Tables & plots from Day 2
6. **Conclusion** — Future work
```

#### Hour 3-4: Finalize Configs & Docs
- Document each research module
- Create experiment matrix script
- Write WandB run descriptions
- Prepare for larger-scale experiments

**Deliverable:** Paper outline, results table, plot figures

---

## 🔧 Integration Checklist

### Before Day 1
- [ ] Research modules created (`models/research_modules.py`)
- [ ] Config file created (`configs/research_modules.yaml`)
- [ ] Lightning trainer created (`experiments/train_lightning.py`)
- [ ] Smoke tests created (`experiments/test_research_modules.py`)

### During Day 1
- [ ] Smoke tests pass (verify with `python test_research_modules.py`)
- [ ] Models imported in `__init__.py`
- [ ] Router integrated into projection layer
- [ ] 4 config variations created (baseline, router_only, router_guided, etc.)
- [ ] Lightning AI project created
- [ ] 1 epoch test run completes successfully
- [ ] Code committed to version control

### During Day 2
- [ ] Baseline experiment (10 epochs) completes
- [ ] Router experiment (10 epochs) completes
- [ ] Router+Guidance experiment (10 epochs) completes
- [ ] Metrics extracted and compared
- [ ] Speedup measurements recorded

### During Day 3
- [ ] Results tables generated
- [ ] Plots created
- [ ] Paper outline written
- [ ] Experiment configs finalized
- [ ] Ready for larger-scale experiments

---

## 📊 Experiment Tracking Template

Create `EXPERIMENT_LOG.md`:

```markdown
# Experiment Log

## Baseline (Router Disabled)
- Run date: [DATE]
- Epochs: 10
- Avg training loss: [X]
- Avg validation loss: [X]
- Inference speed: [X] tokens/sec
- Notes: Reference point

## Router Only
- Run date: [DATE]
- Epochs: 10
- Avg training loss: [X]
- Avg validation loss: [X]
- Inference speed: [X] tokens/sec
- Mean tokens kept: [X] / 32
- Speedup vs baseline: [X]%
- Notes: [observations]

## Router + Instruction Guidance
- Run date: [DATE]
- Epochs: 10
- Avg training loss: [X]
- Avg validation loss: [X]
- Inference speed: [X] tokens/sec
- Notes: [observations]
```

---

## ⚡ Quick Start (Run This Now)

```bash
# 1. Verify files created
cd ~/Desktop/lvlm2
ls models/research_modules.py && echo "✓ Research modules"
ls configs/research_modules.yaml && echo "✓ Config"
ls experiments/train_lightning.py && echo "✓ Lightning trainer"
ls experiments/test_research_modules.py && echo "✓ Smoke tests"

# 2. Run smoke tests
cd experiments
python test_research_modules.py

# 3. If all pass, you're ready for Day 1!
```

---

## 🎓 Research Module Descriptions

### AdaptiveVisualTokenRouter
**Purpose:** Learn which visual tokens are important per question  
**Key Idea:** Use question embeddings to score tokens, select top-k (8-32)  
**Paper analogy:** "Hard attention with learned importance"  
**Config:** `min_tokens=8, max_tokens=32, query_conditioned=true`

### InstructionGuidedVisualAggregator
**Purpose:** Task-specific visual token selection  
**Key Idea:** Different questions need different visual evidence  
**Paper analogy:** "Conditional visual attention based on question type"  
**Config:** `output_tokens=32, num_heads=8`

### VisualTextGroundingHead
**Purpose:** Reduce hallucinations via visual-text consistency  
**Key Idea:** Contrastive loss pulls answer embeddings toward visual tokens  
**Paper analogy:** "Cross-modal grounding for factuality"  
**Config:** `lambda_grounding=0.05, temperature=0.07`

### TemporalFrameFusion
**Purpose:** Fuse video frames with temporal context  
**Key Idea:** Transformer-based temporal encoder with learnable queries  
**Paper analogy:** "Temporal aggregation for video understanding"  
**Config:** `num_heads=8, output_tokens=32, max_frames=16`

### MemoryAwareProjection
**Purpose:** Carry context across image chunks  
**Key Idea:** Recurrent memory slots updated with attention  
**Paper analogy:** "Episodic memory for multi-image reasoning"  
**Config:** `memory_slots=8, num_heads=8`

### DynamicVisualTokenGenerator
**Purpose:** Alternative learnable sparsity mechanism  
**Key Idea:** Generate tokens with learned gate, output mask  
**Paper analogy:** "End-to-end sparsity learning"  
**Config:** `min_tokens=8, max_tokens=32` (Choose ONE: router or this)

### UnsupportedClaimPenalty
**Purpose:** Evidence scoring for fact grounding  
**Key Idea:** Lightweight evidence scorer between visual and answer  
**Paper analogy:** "Fact verification via visual grounding"  
**Config:** `lambda_penalty=0.05` (Requires negative examples)

---

## 🚨 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "Module not found" | Run `pip install -e .` in project root |
| Smoke tests fail | Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"` |
| Lightning trainer crashes | Install latest: `pip install --upgrade pytorch-lightning` |
| GPU out of memory | Reduce batch size: `--batch-size 16` or `--batch-size 8` |
| WandB disabled | Add `--no-wandb` flag or run `wandb login` |

---

## 📚 Next Steps After Day 3

1. **Larger Datasets:** Run on full TVQA/ActivityNet datasets
2. **More Ablations:** Test different router configurations (min_tokens, max_tokens)
3. **Grounding Loss:** Add VisualTextGroundingHead and measure hallucination reduction
4. **Paper Writing:** Use results to write full paper (methods, experiments, analysis)
5. **Submission:** Target ICCV, NeurIPS, or ICLR

---

## 📞 Support

If stuck:
1. Check the smoke test: `python test_research_modules.py`
2. Review config: `cat configs/research_modules.yaml`
3. Check logs: `tail -100 checkpoints/exp/lightning_logs/version_0/events*`
4. Look at Lightning AI docs: https://lightning.ai/

---

**You're all set! Start with Hour 1 of Day 1 and follow the checklist. Good luck! 🚀**
