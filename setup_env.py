"""
Auto-setup and test LVLM environment
Installs dependencies and runs validation tests
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{description}...", end=" ", flush=True)
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            print("✓")
            return True
        else:
            print("✗")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    print("\n" + "="*80)
    print("LVLM SETUP & VALIDATION")
    print("="*80)
    
    # Get Python executable from current environment
    python_exe = sys.executable
    venv_pip = os.path.join(os.path.dirname(python_exe), "pip.exe")
    
    print(f"\nPython: {python_exe}")
    print(f"Pip: {venv_pip}")
    
    # Step 1: Install core dependencies
    print("\n[1/4] Installing Core Dependencies")
    deps = [
        "torch torchvision",
        "transformers peft",
        "pillow numpy tqdm",
    ]
    
    for dep in deps:
        if not run_command(f'"{venv_pip}" install -q {dep}', f"  Installing {dep.split()[0]}"):
            print(f"Failed to install {dep}")
            return False
    
    # Step 2: Install CLIP
    print("\n[2/4] Installing CLIP")
    if not run_command(
        f'"{venv_pip}" install -q "git+https://github.com/openai/CLIP.git@main"',
        "  Installing CLIP"
    ):
        print("Note: CLIP installation may require git. Continuing anyway...")
    
    # Step 3: Run validation test
    print("\n[3/4] Running Validation Tests")
    test_cmd = f'"{python_exe}" test_lvlm_local.py'
    if not run_command(test_cmd, "  Running validation"):
        print("Validation tests failed!")
        return False
    
    # Step 4: Summary
    print("\n[4/4] Setup Complete")
    print("="*80)
    print("✓ All setup and validation tests passed!")
    print("="*80)
    print("\nYou can now:")
    print("  1. Prepare real LLaVA dataset")
    print("  2. Run: python experiments/train_llava.py")
    print("  3. Or use: notebooks/06_llava_training_colab.ipynb on Colab")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
