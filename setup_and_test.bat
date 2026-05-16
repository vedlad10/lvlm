@echo off
REM Setup and test LVLM code on Windows

echo.
echo ============================================================================
echo LVLM Code Testing & Setup
echo ============================================================================
echo.

REM Activate virtual environment
echo [1/5] Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install core dependencies
echo [2/5] Installing core dependencies (this may take 5-10 minutes)...
pip install -q torch torchvision transformers>=4.35.0 peft>=0.7.0 pillow numpy tqdm 2>nul

REM Install CLIP from git
echo [3/5] Installing CLIP...
pip install -q "git+https://github.com/openai/CLIP.git" 2>nul

REM Run validation test
echo [4/5] Running validation tests...
python test_lvlm_local.py

REM Check results
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================================
    echo SUCCESS! All validation tests passed
    echo ============================================================================
    echo.
    echo Next steps:
    echo 1. Prepare your real LLaVA dataset
    echo 2. Update config in train_llava.py or use the Colab notebook
    echo 3. Run: python experiments/train_llava.py
    echo.
) else (
    echo.
    echo VALIDATION FAILED! Check errors above.
    echo.
    exit /b 1
)

pause
