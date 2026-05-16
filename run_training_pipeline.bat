@echo off
REM LVLM Complete Training Pipeline (Windows)

setlocal enabledelayedexpansion

echo.
echo ========================================
echo LVLM Complete Training Pipeline
echo ========================================
echo.

REM Configuration
set DATASET=%1
if "!DATASET!"=="" set DATASET=tvqa

set EPOCHS=%2
if "!EPOCHS!"=="" set EPOCHS=10

set BATCH_SIZE=%3
if "!BATCH_SIZE!"=="" set BATCH_SIZE=32

set LR=%4
if "!LR!"=="" set LR=1e-4

echo Configuration:
echo   Dataset: !DATASET!
echo   Epochs: !EPOCHS!
echo   Batch Size: !BATCH_SIZE!
echo   Learning Rate: !LR!
echo.

REM Change to experiments directory
cd /d "%~dp0experiments"
if !errorlevel! neq 0 (
    echo Error: Could not change to experiments directory
    exit /b 1
)

REM Step 1: Extract Features
echo.
echo [Step 1] Extract Features
echo Running: python extract_features.py --dataset !DATASET! --mock
python extract_features.py ^
    --config ../configs/experiment.yaml ^
    --dataset !DATASET! ^
    --mock

if !errorlevel! neq 0 (
    echo Warning: Feature extraction had issues
) else (
    echo Features extracted successfully
)

REM Step 2: Train Model
echo.
echo [Step 2] Train Model
if not exist checkpoints mkdir checkpoints
echo Running: python train.py --dataset !DATASET! --epochs !EPOCHS! --batch_size !BATCH_SIZE! --lr !LR!
python train.py ^
    --config ../configs/experiment.yaml ^
    --dataset !DATASET! ^
    --epochs !EPOCHS! ^
    --batch_size !BATCH_SIZE! ^
    --lr !LR! ^
    --output_dir ./checkpoints

if !errorlevel! neq 0 (
    echo Training failed or completed with errors
    exit /b 1
)
echo Training complete

REM Step 3: Evaluate Model
echo.
echo [Step 3] Evaluate Model
if not exist results mkdir results

set CHECKPOINT=./checkpoints/checkpoint_best.pt
if not exist !CHECKPOINT! (
    echo Warning: Best checkpoint not found
    for /f "tokens=*" %%A in ('dir /b /od checkpoints\checkpoint*.pt 2^>nul ^| tasklist /FI "IMAGENAME eq sort*" ^>nul 2^>^&1 ^&^& findstr ".*"') do (
        set CHECKPOINT=./checkpoints/%%A
    )
)

echo Running: python eval.py --checkpoint !CHECKPOINT! --dataset !DATASET!
python eval.py ^
    --config ../configs/experiment.yaml ^
    --checkpoint !CHECKPOINT! ^
    --dataset !DATASET! ^
    --split test ^
    --output ./results/eval_results.json

if !errorlevel! neq 0 (
    echo Warning: Evaluation had issues
) else (
    echo Evaluation complete
)

REM Step 4: Run Ablations
echo.
echo [Step 4] Run Ablation Studies
echo Running: python eval_ablations.py
python eval_ablations.py ^
    --config ../configs/experiment.yaml ^
    --checkpoint_dir ./checkpoints ^
    --dataset !DATASET! ^
    --output ./results/ablation_results.json

if !errorlevel! neq 0 (
    echo Warning: Ablation studies had issues
) else (
    echo Ablation studies complete
)

REM Step 5: Visualize Results
echo.
echo [Step 5] Generate Visualizations
if exist results\ablation_results.json (
    echo Running: python visualize_results.py
    python visualize_results.py ^
        --ablation_results ./results/ablation_results.json ^
        --output_dir ./results/plots
    
    if !errorlevel! equ 0 (
        echo Visualizations generated
    )
) else (
    echo Warning: Ablation results not found, skipping visualizations
)

REM Summary
echo.
echo ========================================
echo Pipeline Complete!
echo ========================================
echo.
echo Output files:
echo   Model checkpoint: .\checkpoints\checkpoint_best.pt
echo   Evaluation results: .\results\eval_results.json
echo   Ablation results: .\results\ablation_results.json
echo   Visualizations: .\results\plots\
echo.
echo Next steps:
echo   1. Review results in .\results\
echo   2. Check visualizations in .\results\plots\
echo   3. Fine-tune hyperparameters and re-train
echo   4. Prepare paper with results
echo.
pause
