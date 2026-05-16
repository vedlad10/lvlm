# Quick Reference - Common Commands

## One-Command Training (Recommended)

**Windows**:
```bash
run_training_pipeline.bat tvqa 20 32 1e-4
```

**Linux/Mac**:
```bash
bash run_training_pipeline.sh tvqa 20 32 1e-4
```

---

## Individual Commands

### Setup (One-time)
```bash
# Extract and cache features
cd experiments
python extract_features.py --config ../configs/experiment.yaml --dataset tvqa --mock
```

### Training
```bash
# Train for 20 epochs
python train.py --config ../configs/experiment.yaml --dataset tvqa --epochs 20

# Resume from checkpoint
python train.py --dataset tvqa --epochs 20 --resume checkpoints/checkpoint_best.pt

# Custom learning rate
python train.py --dataset tvqa --epochs 20 --lr 5e-5

# Small batch for testing
python train.py --dataset tvqa --epochs 5 --batch_size 8
```

### Evaluation
```bash
# Full evaluation
python eval.py --checkpoint checkpoints/checkpoint_best.pt --dataset tvqa

# Save results to file
python eval.py --checkpoint checkpoints/checkpoint_best.pt --dataset tvqa --output results/eval.json

# ActivityNet-QA
python eval.py --checkpoint checkpoints/checkpoint_best.pt --dataset activitynet_qa
```

### Ablation Studies
```bash
# Run all ablations
python eval_ablations.py --config ../configs/experiment.yaml --checkpoint_dir ./checkpoints --output results/ablations.json

# ActivityNet-QA ablations
python eval_ablations.py --dataset activitynet_qa --output results/ablations_activitynet.json
```

### Visualization
```bash
# Generate all plots
python visualize_results.py --ablation_results results/ablations.json --output_dir results/plots

# Specific metrics only
python visualize_results.py --ablation_results results/ablations.json --metrics accuracy temporal_iou_mean
```

---

## Directory Structure for Results

After running pipeline:
```
results/
├── eval_results.json           # Main evaluation metrics
├── ablation_results.json       # Ablation study results
└── plots/
    ├── accuracy_speed_tradeoff.png
    ├── ablation_accuracy.png
    ├── ablation_temporal_iou_mean.png
    ├── ablation_avg_depth.png
    ├── depth_distribution.png
    └── summary_table.png

checkpoints/
├── checkpoint_epoch_00.pt      # Checkpoint after epoch 1
├── checkpoint_epoch_01.pt      # Checkpoint after epoch 2
├── ...
└── checkpoint_best.pt          # Best model
```

---

## Dataset Selection

### TVQA
- 152k QA pairs
- 21.8k video clips
- Avg video length: 85 seconds
- Good for quick testing

```bash
python train.py --dataset tvqa --epochs 20
```

### ActivityNet-QA
- 58k QA pairs
- Longer videos (avg 2 min)
- More challenging temporal reasoning
- Use for thorough evaluation

```bash
python train.py --dataset activitynet_qa --epochs 15
```

---

## Hyperparameter Variations

### Quick Test (2 minutes)
```bash
python train.py --dataset tvqa --epochs 1 --batch_size 8
```

### Standard Training (1-2 hours)
```bash
python train.py --dataset tvqa --epochs 20 --batch_size 32 --lr 1e-4
```

### Careful Fine-tuning (3-4 hours)
```bash
python train.py --dataset tvqa --epochs 20 --batch_size 16 --lr 5e-5
```

### Large Batch (needs good GPU)
```bash
python train.py --dataset tvqa --epochs 10 --batch_size 64 --lr 2e-4
```

---

## Performance Monitoring

### Training
- Check `checkpoints/` for model saves
- Look at loss convergence
- Monitor GPU memory usage

### Evaluation
- Check `results/eval_results.json` for accuracy
- Look for high temporal IoU (>0.5)
- Inference speed should be >100 samples/sec

### Ablations
- Compare accuracy deltas vs full model
- Full model should be best overall
- Individual components should show contribution

---

## Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch_size: `--batch_size 16` |
| Slow data loading | Increase workers in config or use `--mock` |
| NaN loss | Reduce learning rate: `--lr 1e-5` |
| Model not improving | Try smaller learning rate or more epochs |
| Features not found | Run `python extract_features.py` first |
| Dataset not found | Use `--mock` flag or download datasets |

---

## Expected Times

| Operation | Time (GPU V100) |
|-----------|-----------------|
| Feature extraction (mock) | 1 min |
| Feature extraction (real) | 30 min |
| Train 1 epoch (TVQA) | 1-2 min |
| Train 1 epoch (ActivityNet-QA) | 3-5 min |
| Evaluation (full test set) | 2-3 min |
| Ablation studies (8 configs) | 10-15 min |
| Visualization generation | <1 min |
| **Full Pipeline** | **20-40 min** |

---

## Output Interpretation

### Accuracy
- 50%: Random guess (baseline)
- 65%+: Good performance
- 75%+: Excellent performance
- 85%+: State-of-the-art

### Temporal IoU
- 0.4: Moderate temporal grounding
- 0.5+: Good temporal accuracy
- 0.65+: Excellent temporal localization

### Reasoning Depth
- ~3.0: Model uses ~3 reasoning hops on average
- Decreasing during training: Adaptive depth learning
- Should be lower than max_hops (usually 5)

### Speed (samples/sec)
- GPU inference: 100-200 samples/sec
- CPU inference: 5-20 samples/sec
- Faster with fixed depth ablation

---

## Next Steps After Training

1. **Review Results**
   - Check `results/ablations.json` for all metrics
   - Look at visualization plots in `results/plots/`
   - Compare accuracy vs speed Pareto frontier

2. **Fine-tune**
   - If accuracy is low, increase epochs or reduce learning rate
   - If loss doesn't decrease, check data loading
   - If temporal IoU is low, check temporal grounding weight

3. **Prepare Paper**
   - Use plots from `results/plots/`
   - Include metrics from `results/eval_results.json`
   - Show ablation contributions table

4. **Advanced Training**
   - Try different datasets
   - Experiment with hyperparameters
   - Enable multi-GPU training

---

## File Locations

```
Project Root: /path/to/lvlm2

Models:
- models/lvlm.py
- models/temporal_binding.py
- models/chimrt.py
- models/adaptive_depth.py
- models/multimodal_vdb.py

Config:
- configs/experiment.yaml

Training:
- experiments/train.py
- experiments/eval.py
- experiments/eval_ablations.py
- experiments/extract_features.py
- experiments/visualize_results.py

Results:
- experiments/results/
- experiments/checkpoints/
```

---

## Key Formulas

**Loss Function**:
```
L_total = w_ans * L_ans + w_temp * L_temp + w_contra * L_contra + w_depth * L_depth

Where:
  L_ans = cross_entropy(pred_answer, true_answer)
  L_temp = smooth_l1(pred_spans, true_spans)
  L_contra = NT_Xent(question, memory)
  L_depth = max(0, entropy - ε) * stop_logit
```

**Temporal IoU**:
```
IoU = intersection / union
    = (end_pred - start_pred) ∩ (end_true - start_true) / 
      (end_pred - start_pred) ∪ (end_true - start_true)
```

**Adaptive Depth**:
```
EntropyCond = entropy(reasoning_state) < entropy_threshold
GateCond = sigmoid(w^T * reasoning_state) > 0.5
Stop = EntropyCond AND GateCond
```

---

**Last Updated**: March 2026
**Status**: ✅ Complete and Production-Ready
