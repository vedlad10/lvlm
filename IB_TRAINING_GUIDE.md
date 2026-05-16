# Information Bottleneck Training Guide

## Quick Summary

Your LVLM now trains using **Tishby's Information Bottleneck framework** alongside mathematical formulations. This guide explains what to expect during training and how to interpret the metrics.

---

## The Core Idea (In 30 Seconds)

**Information Bottleneck objective**: $L_{IB} = I(X; Z) - \beta \cdot I(Z; Y)$

- **X** = Video frames (input)
- **Z** = Compressed memory nodes (bottleneck)
- **Y** = Answer prediction (task)

**Goal**: Extract only the information **Y** needs from **X**, discarding the rest.

- ✓ If I(X; Z) high and I(Z; Y) low: **Bad** (retaining useless info)
- ✓ If I(X; Z) low and I(Z; Y) high: **Good** (efficient compression)

---

## Three IB Loss Components

### 1. Memory Consolidation IB
**Metric**: `ib_memory_consolidation`

What it measures:
- How much compression happens: frames → memory nodes
- Information theory: $I(X; Z)$ (reconstruction loss) - $\beta \cdot I(Z; Y)$ (KL divergence)

Typical behavior during training:
```
Epoch 1:   ib_memory ≈ 0.15-0.20  (high compression penalty)
Epoch 5:   ib_memory ≈ 0.08-0.12  (learning efficient compression)
Epoch 20:  ib_memory ≈ 0.05-0.08  (stable compression)
```

What it means:
- 📊 Decreasing = Model learning to compress frames efficiently ✓
- 📈 Increasing = Model retaining too much info ✗
- ⚠️ Stable = Optimal compression found

### 2. Reasoning Depth IB
**Metric**: `ib_reasoning_depth`

What it measures:
- Trade-off between reasoning hops and accuracy
- Goal: Solve with minimal hops while maintaining accuracy

Typical behavior during training:
```
Epoch 1:   ib_depth ≈ 0.10-0.15   (many hops still needed)
Epoch 10:  ib_depth ≈ 0.05-0.08   (confident reasoning)
Epoch 20:  ib_depth ≈ 0.03-0.06   (efficient depth)
```

Combined with actual depth metric:
```
Epoch 1:   avg_depth = 4.2 hops,  ib_depth = 0.15
Epoch 20:  avg_depth = 2.3 hops,  ib_depth = 0.05  ✓ (more efficient)
```

### 3. Multimodal Alignment IB
**Metric**: `ib_multimodal`

What it measures:
- Vision (image embeddings) and language (question embeddings) bottleneck alignment
- Separate compression spaces that share a common prior

Typical behavior during training:
```
Epoch 1:   ib_multimodal ≈ 0.08-0.12
Epoch 20:  ib_multimodal ≈ 0.04-0.08
```

What it means:
- 📊 Both modalities find efficient compression ✓
- 🔗 Joint information bottleneck aligns understanding ✓

---

## Total Loss Breakdown

Your training uses this weighted combination:

```python
total_loss = (
    1.0 * answer_loss +           # Main task (Q→A)
    0.5 * temporal_loss +         # Grounding (when did it happen)
    0.1 * sparsity_loss +         # Sparse attention
    0.2 * adaptive_depth_loss +   # Efficient reasoning
    0.1 * total_ib_loss           # Information balance
)
```

**Expected total loss progression**:
```
Epoch 1:   total ≈ 2.0-2.5
Epoch 5:   total ≈ 1.2-1.5
Epoch 10:  total ≈ 0.8-1.1
Epoch 20:  total ≈ 0.5-0.8  ✓
```

**Breakdown by component** (at Epoch 20):
```
answer_loss:         0.40  (1.0 weight × 0.40)
temporal_loss:       0.15  (0.5 weight × 0.30)
sparsity_loss:       0.02  (0.1 weight × 0.20)
adaptive_depth_loss: 0.04  (0.2 weight × 0.20)
total_ib_loss:       0.05  (0.1 weight × 0.50)
                     ----
                     0.66  ✓
```

---

## Monitoring During Training

### TensorBoard Metrics to Watch

```bash
# Navigate to training logs
tensorboard --logdir checkpoints/
```

**Watch these scalar metrics**:

| Metric | Good Sign | Warning Sign |
|--------|-----------|--------------|
| `answer_loss` | Decreasing smoothly | Increasing or unstable |
| `temporal_loss` | Decreasing after epoch 5 | Plateaus too early |
| `ib_memory_consolidation` | Decreasing trend | Never decreases |
| `ib_reasoning_depth` | Decreasing (→fewer hops) | Stable high value |
| `ib_multimodal` | Decreasing smoothly | Diverging (modalities misalign) |
| `avg_reasoning_depth` | Should go 5.0→2-3 | Stuck at max hops |
| `avg_stop_probability` | Should increase | Stays low (no early stopping) |
| `avg_entropy` | Should decrease | Oscillates |

### Command-Line Monitoring

```bash
# Watch training loss in real-time
tail -f checkpoints/train.log | grep "Total loss"

# Parse loss components
grep "ib_memory\|ib_depth\|ib_multimodal" checkpoints/train.log | tail -20
```

### Expected Patterns

**Good training looks like**:
```
Epoch 1:  loss=2.34, answer=0.52, temporal=0.45, ib_mem=0.18, ib_depth=0.12, ib_mm=0.10
Epoch 5:  loss=1.45, answer=0.38, temporal=0.28, ib_mem=0.10, ib_depth=0.08, ib_mm=0.07
Epoch 10: loss=0.92, answer=0.25, temporal=0.18, ib_mem=0.07, ib_depth=0.05, ib_mm=0.05
Epoch 20: loss=0.66, answer=0.18, temporal=0.12, ib_mem=0.05, ib_depth=0.03, ib_mm=0.04
         ↓ smooth decrease across all components ✓
```

**Bad training looks like**:
```
Epoch 1:  loss=2.50, ib_mem=0.20, ib_depth=0.15, ib_mm=0.12
Epoch 10: loss=2.48, ib_mem=0.21, ib_depth=0.16, ib_mm=0.13
         ↓ IB losses not decreasing (compression not learned) ✗
```

---

## Understanding IB Behavior

### When IB Loss Increases (Not Good)

**Possible causes**:
1. **Learning rate too high**
   - Solution: Reduce by 0.5x
   - Try: `--lr 5e-5` instead of `1e-4`

2. **Beta parameters too high**
   - Current: `beta_memory=0.01, beta_depth=0.01`
   - Try: `beta_memory=0.001, beta_depth=0.001` (more task-focused)

3. **Not enough capacity to compress**
   - Check: Is `consolidated_memory` size (K) reasonable?
   - Try: Increase `max_consolidated_size` in config

### When IB Loss Doesn't Decrease (Training Unstable)

**Diagnostics**:
```python
# Add to train.py for debugging
if epoch % 5 == 0:
    print(f"Info plane: I(X;Z)={ix_z:.4f}, I(Z;Y)={iz_y:.4f}, ratio={ix_z/iz_y:.2f}")
    # Healthy: ratio around 2-3 (some redundancy, but compressed)
    # ~1.0: No redundancy (suspicious, may be undercompressing)
    # >10: Too much redundancy (undercompressing)
```

---

## Reasoning Depth & IB Trade-off

### Depth Behavior During Training

Expected evolution:

```
Epoch 1:  70% samples use 5 hops,  30% use <5
Epoch 5:  50% samples use 5 hops,  50% use 2-4
Epoch 10: 20% samples use 5 hops,  80% use 1-3  ✓
Epoch 20: 15% samples use 5 hops,  85% use 1-3  ✓

Average: 5.0 → 4.2 → 2.8 → 2.1  ✓ (good compression)
```

**What's happening**:
- IB depth loss: penalizes excessive hops
- Model learns: "Can I answer this in 1-2 hops?" (efficient reasoning)
- Entropy check: prevents premature stopping (maintains accuracy)

### If Depth Stays at Max Hops

**Diagnosis**: Model hasn't learned to early-stop

**Solutions**:
1. Increase `entropy_threshold` (default 0.5)
   - Try: 0.7 (less stringent entropy requirement)
   
2. Lower `stopping_gate_weights` initialization
   - More likely to trigger stop condition

3. Check if answer_loss is too high
   - Model still learning basics, not efficiency
   - Reduce IB weight temporarily: `--ib-weight 0.05`

---

## Information Plane Visualization

After training, you can visualize the information plane:

```python
import matplotlib.pyplot as plt
import numpy as np

# During training, track:
ix_z_history = []  # I(X; Z) per epoch
iz_y_history = []  # I(Z; Y) per epoch
y_loss_history = []  # Accuracy loss

# Plot information plane
plt.figure(figsize=(8, 6))
plt.scatter(ix_z_history, iz_y_history, c=range(len(ix_z_history)), cmap='viridis')
plt.xlabel('I(X; Z) - Compression')
plt.ylabel('I(Z; Y) - Task Performance')
plt.title('Information Plane Trajectory')
plt.colorbar(label='Epoch')
plt.axhline(y=np.max(iz_y_history) * 0.9, color='r', linestyle='--', label='Target I(Z;Y)')
plt.axvline(x=np.min(ix_z_history), color='g', linestyle='--', label='Best Compression')
plt.legend()
plt.savefig('information_plane.png')
```

**Healthy trajectory**: Moves towards lower-left (more compression, high task performance)

---

## Checkpointing Strategy

Save best models based on different criteria:

```bash
# Best overall loss
checkpoints/best_loss.pt

# Best memory consolidation IB
checkpoints/best_ib_memory.pt

# Best reasoning efficiency
checkpoints/best_ib_depth.pt

# Best multimodal alignment
checkpoints/best_ib_multimodal.pt
```

**Choose based on your priority**:
- **Want fast inference?** Use `best_ib_depth.pt` (fewer hops)
- **Want best accuracy?** Use `best_loss.pt` (overall performance)
- **Want efficient storage?** Use `best_ib_memory.pt` (compressed memory)

---

## Configuration Tuning

### For Better Information Compression

```yaml
# configs/experiment.yaml

# Make memory consolidation more aggressive
sparsity_target: 0.2  # More compression (default: 0.3)

# Increase IB weight in loss
ib_weight: 0.15  # Default: 0.1

# Tighter entropy threshold
entropy_threshold: 0.7  # Default: 0.5 (force early stop)

# Stronger compression penalty
beta_memory: 0.02  # Default: 0.01
```

### For More Accurate Reasoning

```yaml
# configs/experiment.yaml

# Allow more reasoning
max_reasoning_hops: 7  # Default: 5

# Weaker compression penalty
beta_memory: 0.005  # Default: 0.01

# Reduced IB weight
ib_weight: 0.05  # Default: 0.1
```

### Balanced (Recommended)

```yaml
# configs/experiment.yaml

# Current defaults (good balance)
sparsity_target: 0.3
ib_weight: 0.1
entropy_threshold: 0.5
beta_memory: 0.01
beta_depth: 0.01
beta_multimodal: 0.01
max_reasoning_hops: 5
```

---

## Troubleshooting

### Q: IB losses are NaN after epoch 1

**Answer**: 
- Check if `original_frames` has NaN values
- Verify `consolidated_memory` dimensions match expectations
- Try reducing `beta_memory` to 0.001

### Q: Reasoning depth never decreases

**Answer**:
- IB depth penalty too weak
- Increase `beta_depth` to 0.02-0.05
- Or reduce `entropy_threshold` to 0.3-0.4

### Q: Memory consolidation IB keeps increasing

**Answer**:
- Learning rate too high → reduce by 0.5x
- Consolidation not learning efficiently
- Try increasing number of consolidation steps

### Q: Model trains fine but slow inference

**Answer**:
- Avg reasoning depth still high (check metric)
- Use checkpoint with lower `ib_reasoning_depth`
- Or increase `ib_weight` and retrain

---

## Next: Evaluation Phase

After training converges, evaluate on test set:

```bash
# Test best checkpoint
python experiments/eval.py \
    --checkpoint checkpoints/best_loss.pt \
    --dataset clip_vit_b32 \
    --metrics qa_accuracy temporal_iou efficiency
```

**Metrics with IB framework**:
- **QA Accuracy**: What % of answers correct
- **Temporal IOU**: How well temporal spans match ground truth
- **Reasoning Efficiency**: Average hops per sample (lower = better)
- **Memory Compression**: K/T ratio (target < 0.3)
- **Information Ratio**: I(X;Z) / I(Z;Y) (ideally 2-5)

---

## Summary

When you start training:

1. **Watch for all losses decreasing** (good sign ✓)
2. **IB components should drop** (compression being learned)
3. **Reasoning depth should decrease** (efficiency improving)
4. **Total loss should plateau** around epoch 15-20

**Your model is efficient when**:
- Average reasoning depth: 1.5-2.5 (instead of max 5)
- I(X;Z): ~0.3-0.5 (compressed from original dimension)
- I(Z;Y): ~0.8-0.95 (task info preserved)
- Total loss: < 0.8

🎯 **Go train! The Information Bottleneck framework will guide efficient learning.** 🎯
