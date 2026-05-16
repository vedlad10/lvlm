# 🚀 QUICK START GUIDE - LVLM Research Integration

**Status:** ✅ Phase 1 Complete | ⏳ Ready for Phase 2  
**Current Time Estimate:** 2 hours to complete Phase 2 (all remaining tasks)

---

## ⚡ 5-Minute Setup (Do This NOW)

```bash
cd c:\Users\DELL\Desktop\lvlm2

# 1. Generate config variations (creates 4 config files automatically)
python scripts/generate_configs.py

# Expected output:
# ✓ Created research_modules_baseline.yaml
# ✓ Created research_modules_router_only.yaml  
# ✓ Created research_modules_router_guided.yaml
# ✓ Created research_modules_with_grounding.yaml
# ✓ All configs generated successfully!

# 2. Verify files were created
ls configs/research_modules_*.yaml
# Should list 5 files: baseline, router_only, router_guided, with_grounding, + original
```

---

## 📋 What You Have Now

### ✅ Research Modules (7 total, all tested)
```
✓ AdaptiveVisualTokenRouter          - Select 8-32 tokens based on importance
✓ InstructionGuidedVisualAggregator  - Query-aware token selection
✓ VisualTextGroundingHead            - Reduce hallucinations
✓ TemporalFrameFusion                - Video frame fusion
✓ MemoryAwareProjection              - Context carry-over
✓ DynamicVisualTokenGenerator        - Alternative sparsity
✓ UnsupportedClaimPenalty            - Evidence scoring
```

### ✅ Training Infrastructure
```
✓ train_lightning.py                 - PyTorch Lightning trainer (GPU-distributed ready)
✓ test_research_modules.py           - Smoke tests (all passing ✓)
✓ scripts/generate_configs.py        - Auto-generate config variations
✓ scripts/run_experiments.py         - Orchestrate ablation experiments
```

### ✅ Documentation
```
✓ RESEARCH_MODULES_INTEGRATION.md    - Complete 3-day plan (400+ lines)
✓ EXECUTION_SUMMARY.md               - Status overview
✓ QUICK_START.md                     - This file
```

---

## 🎯 Next 30 Minutes (Phase 2 Start)

### Step 1: Update imports (2 minutes)
Edit `models/__init__.py`, add at the end:

```python
# Research Modules
from .research_modules import (
    AdaptiveVisualTokenRouter,
    InstructionGuidedVisualAggregator,
    VisualTextGroundingHead,
    UnsupportedClaimPenalty,
    MemoryAwareProjection,
    TemporalFrameFusion,
    DynamicVisualTokenGenerator,
    RoutingStats,
)
```

### Step 2: Generate configs (1 minute)
```bash
python scripts/generate_configs.py
```

### Step 3: Verify setup (2 minutes)
```bash
# Activate environment
& c:\Users\DELL\Desktop\lvlm2\.venv\Scripts\Activate.ps1

# Run smoke tests again
python experiments/test_research_modules.py

# Should show: ✓ All smoke tests passed!
```

---

## 🔥 3-Day Research Roadmap

### **DAY 1: Today (6-8 hours remaining)**
- ✅ Research modules created
- ✅ Smoke tests passing  
- ⏳ **TODO:** Complete imports, generate configs, test Lightning trainer (30 min total)
- ⏳ **TODO:** Set up Lightning AI (30 min)
- **Target:** Lightning trainer runs 1 epoch successfully

### **DAY 2: Tomorrow (8 hours)**
```bash
# Run 3 experiments automatically
python scripts/run_experiments.py \
    --dataset clip \
    --epochs 10 \
    --batch-size 32 \
    --experiments baseline router_only router_guided
```
**Generates:** 3 experiment runs with metrics, logs, and checkpoints

### **DAY 3: Day After Tomorrow (4 hours)**
- Export results tables
- Create comparison plots
- Write paper outline
- **Done!** Ready to write paper + start larger experiments

---

## 📊 Experiment Variations (Automatic)

Run any of these commands to test different configurations:

```bash
# Baseline (no optimization)
python experiments/train_lightning.py \
    --config configs/experiment.yaml \
    --research-config configs/research_modules_baseline.yaml \
    --epochs 10

# Router only (most efficient)
python experiments/train_lightning.py \
    --config configs/experiment.yaml \
    --research-config configs/research_modules_router_only.yaml \
    --epochs 10

# Router + Instruction Guidance (best accuracy)
python experiments/train_lightning.py \
    --config configs/experiment.yaml \
    --research-config configs/research_modules_router_guided.yaml \
    --epochs 10

# All features (router + guidance + grounding)
python experiments/train_lightning.py \
    --config configs/experiment.yaml \
    --research-config configs/research_modules_with_grounding.yaml \
    --epochs 10
```

---

## 📈 Expected Results (Research Metrics)

| Experiment | Tokens Kept | Speedup | Accuracy | Use Case |
|------------|-------------|---------|----------|----------|
| Baseline | 32 (100%) | 1.0x | Reference | Control |
| Router Only | ~20 (60%) | 1.15x | -0.5% to +0.5% | Efficiency |
| Router + Guidance | ~16 (50%) | 1.25x | +0.5% to +1.5% | Balanced |
| With Grounding | ~16 (50%) | 1.25x | +1% to +2% | Best Accuracy |

---

## 🎓 Module Selection Guide

### Use AdaptiveVisualTokenRouter if:
- ✓ You want inference speedup (10-20% expected)
- ✓ You have enough visual redundancy in images
- ✓ Accuracy preservation is critical (<1% drop acceptable)
- → **Recommended: YES (start here)**

### Also enable InstructionGuidedVisualAggregator if:
- ✓ Accuracy improvements matter more than speed
- ✓ Tasks are diverse (OCR, counting, scene understanding)
- ✓ You want query-aware token selection
- → **Recommended: YES (if router alone is not enough)**

### Also enable VisualTextGroundingHead if:
- ✓ Hallucination reduction is critical
- ✓ You need better grounding/factuality
- ✓ Model is sometimes making up details
- → **Recommended: Try Day 2**

---

## 🚨 Troubleshooting

**Q: Smoke tests fail with CUDA error**  
A: Install/update PyTorch: `pip install --upgrade torch torchvision`

**Q: Lightning trainer won't start**  
A: Install Lightning: `pip install --upgrade pytorch-lightning`

**Q: Config generation script fails**  
A: Install PyYAML: `pip install pyyaml`

**Q: WandB errors**  
A: Add `--no-wandb` flag to disable, or `wandb login` to enable

**Q: GPU out of memory**  
A: Reduce batch size: `--batch-size 16` or `--batch-size 8`

---

## 📁 File Locations

```
lvlm2/
├── models/
│   └── research_modules.py              ← 7 research modules
├── configs/
│   ├── research_modules.yaml            ← Base config
│   ├── research_modules_baseline.yaml   ← Generated
│   ├── research_modules_router_only.yaml ← Generated
│   └── research_modules_router_guided.yaml ← Generated
├── experiments/
│   ├── train_lightning.py               ← Lightning trainer
│   └── test_research_modules.py         ← Smoke tests ✓
├── scripts/
│   ├── generate_configs.py              ← Config generator
│   └── run_experiments.py               ← Experiment runner
├── EXECUTION_SUMMARY.md                 ← Current status
└── RESEARCH_MODULES_INTEGRATION.md      ← Full guide
```

---

## 🎯 Your Next 3 Actions (Right Now!)

### 1️⃣ Generate configs (1 minute)
```bash
python scripts/generate_configs.py
```

### 2️⃣ Update imports (2 minutes)
Edit `models/__init__.py` - add research module imports

### 3️⃣ Verify all works (2 minutes)
```bash
python experiments/test_research_modules.py
```

**After these 5 minutes: You're ready for Phase 2! 🚀**

---

## 📊 Quick Metrics Reference

### Token Efficiency
- Baseline: 32 tokens (100%)
- With Router: ~20 tokens (60%) = 40% reduction
- With Router+Guidance: ~16 tokens (50%) = 50% reduction

### Speed Improvement
- Fewer tokens → Faster inference
- 32 tokens: 1.0x (baseline)
- 20 tokens: 1.15x (15% faster)
- 16 tokens: 1.25x (25% faster)

### Accuracy Impact
- Target: <1% accuracy drop
- Realistic: -0.5% to +1% (varies by dataset)
- Win: If speedup > accuracy drop

---

## 🎓 Paper Section You Can Write Now

*"We investigated adaptive visual token routing as an efficiency mechanism for vision-language models. We propose AdaptiveVisualTokenRouter, which learns to select the K most important visual tokens based on question embeddings, reducing from 32 tokens to K ∈ [8, 32]. Experiments show..."*

---

## ✨ Final Checklist Before Phase 2

- [ ] Configs generated (4 files created)
- [ ] Imports updated in `models/__init__.py`
- [ ] Smoke tests passing
- [ ] Lightning AI installed
- [ ] Ready to start experiments

**All checked? → You're ready for Day 2 ablations! 🚀**

---

**Questions?** See [RESEARCH_MODULES_INTEGRATION.md](RESEARCH_MODULES_INTEGRATION.md) for detailed guidance.  
**Stuck?** Check Troubleshooting section above.  
**Timeline?** 2-3 days to complete, then write paper!

**Go get 'em! 💪**
