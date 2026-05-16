#!/usr/bin/env bash
# LVLM Complete Training Pipeline

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}LVLM Complete Training Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"

# Configuration
DATASET=${1:-"tvqa"}  # tvqa or activitynet_qa
EPOCHS=${2:-"10"}
BATCH_SIZE=${3:-"32"}
LR=${4:-"1e-4"}

echo -e "${YELLOW}Configuration:${NC}"
echo "  Dataset: $DATASET"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LR"
echo ""

# Change to experiments directory
cd "$(dirname "$0")/experiments" || exit 1

# Step 1: Extract Features
echo -e "${BLUE}Step 1: Extract Features${NC}"
echo "Running: python extract_features.py --dataset $DATASET --mock"
python extract_features.py \
  --config ../configs/experiment.yaml \
  --dataset "$DATASET" \
  --mock \
  2>&1

if [ $? -ne 0 ]; then
  echo -e "${YELLOW}Warning: Feature extraction had issues${NC}"
else
  echo -e "${GREEN}✓ Features extracted${NC}"
fi
echo ""

# Step 2: Train Model
echo -e "${BLUE}Step 2: Train Model${NC}"
mkdir -p ./checkpoints
echo "Running: python train.py --dataset $DATASET --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR"
python train.py \
  --config ../configs/experiment.yaml \
  --dataset "$DATASET" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --output_dir ./checkpoints \
  2>&1

if [ $? -ne 0 ]; then
  echo -e "${YELLOW}Training failed or completed with errors${NC}"
  exit 1
fi
echo -e "${GREEN}✓ Training complete${NC}"
echo ""

# Step 3: Evaluate Model
echo -e "${BLUE}Step 3: Evaluate Model${NC}"
mkdir -p ./results
CHECKPOINT="./checkpoints/checkpoint_best.pt"
if [ ! -f "$CHECKPOINT" ]; then
  echo -e "${YELLOW}Warning: Best checkpoint not found, using latest checkpoint${NC}"
  CHECKPOINT=$(ls -t ./checkpoints/checkpoint*.pt 2>/dev/null | head -1)
fi

echo "Running: python eval.py --checkpoint $CHECKPOINT --dataset $DATASET"
python eval.py \
  --config ../configs/experiment.yaml \
  --checkpoint "$CHECKPOINT" \
  --dataset "$DATASET" \
  --split test \
  --output ./results/eval_results.json \
  2>&1

if [ $? -ne 0 ]; then
  echo -e "${YELLOW}Evaluation had issues${NC}"
else
  echo -e "${GREEN}✓ Evaluation complete${NC}"
fi
echo ""

# Step 4: Run Ablations
echo -e "${BLUE}Step 4: Run Ablation Studies${NC}"
echo "Running: python eval_ablations.py"
python eval_ablations.py \
  --config ../configs/experiment.yaml \
  --checkpoint_dir ./checkpoints \
  --dataset "$DATASET" \
  --output ./results/ablation_results.json \
  2>&1

if [ $? -ne 0 ]; then
  echo -e "${YELLOW}Ablation studies had issues${NC}"
else
  echo -e "${GREEN}✓ Ablation studies complete${NC}"
fi
echo ""

# Step 5: Visualize Results
echo -e "${BLUE}Step 5: Generate Visualizations${NC}"
if [ -f "./results/ablation_results.json" ]; then
  echo "Running: python visualize_results.py"
  python visualize_results.py \
    --ablation_results ./results/ablation_results.json \
    --output_dir ./results/plots \
    2>&1
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Visualizations generated${NC}"
  fi
else
  echo -e "${YELLOW}Ablation results not found, skipping visualizations${NC}"
fi
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Pipeline Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Output files:${NC}"
echo "  Model checkpoint: ./checkpoints/checkpoint_best.pt"
echo "  Evaluation results: ./results/eval_results.json"
echo "  Ablation results: ./results/ablation_results.json"
echo "  Visualizations: ./results/plots/"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Review results in ./results/"
echo "  2. Check visualizations in ./results/plots/"
echo "  3. Fine-tune hyperparameters and re-train"
echo "  4. Prepare paper with results"
echo ""
