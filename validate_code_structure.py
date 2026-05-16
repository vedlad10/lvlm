"""
Quick syntax and structure validation for LVLM code (no PyTorch required)
"""

import os
import sys
import ast
import json
from pathlib import Path

print("="*80)
print("LVLM CODE SYNTAX & STRUCTURE VALIDATION")
print("="*80)

# ============================================================================
# 1. Check Python syntax of all modules
# ============================================================================
print("\n[1/4] Checking Python Syntax...")

python_files = [
    "models/llava_clip_encoder.py",
    "models/llava_projection.py",
    "models/llava_lvlm.py",
    "data/llava_dataset.py",
    "experiments/train_llava.py",
    "test_lvlm_local.py",
]

syntax_errors = []
for file_path in python_files:
    full_path = Path(file_path)
    if not full_path.exists():
        print(f"  ✗ File not found: {file_path}")
        syntax_errors.append(file_path)
        continue
    
    try:
        with open(full_path, 'r') as f:
            code = f.read()
        ast.parse(code)
        print(f"  ✓ {file_path}")
    except SyntaxError as e:
        print(f"  ✗ {file_path}: {e}")
        syntax_errors.append(file_path)

if syntax_errors:
    print(f"\n✗ Syntax errors found in: {syntax_errors}")
    sys.exit(1)

# ============================================================================
# 2. Check file structure and imports
# ============================================================================
print("\n[2/4] Checking Module Structure...")

modules_to_check = {
    "models/llava_clip_encoder.py": ["FrozenCLIPEncoder"],
    "models/llava_projection.py": ["VisualProjection"],
    "models/llava_lvlm.py": ["LVLM", "build_lora_llama"],
    "data/llava_dataset.py": ["LLaVADataset", "llava_collate_fn"],
    "experiments/train_llava.py": ["TrainingConfig", "train_lvlm", "build_model"],
}

structure_errors = []
for file_path, expected_classes in modules_to_check.items():
    full_path = Path(file_path)
    if not full_path.exists():
        structure_errors.append(f"{file_path}: FILE NOT FOUND")
        print(f"  ✗ {file_path}")
        continue
    
    try:
        with open(full_path, 'r') as f:
            tree = ast.parse(f.read())
        
        defined = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                defined.add(node.name)
        
        missing = [c for c in expected_classes if c not in defined]
        if missing:
            structure_errors.append(f"{file_path}: Missing {missing}")
            print(f"  ✗ {file_path}: Missing {missing}")
        else:
            found = ", ".join(expected_classes)
            print(f"  ✓ {file_path}: {found}")
    except Exception as e:
        structure_errors.append(f"{file_path}: {e}")
        print(f"  ✗ {file_path}: {e}")

if structure_errors:
    print(f"\n✗ Structure errors found:")
    for err in structure_errors:
        print(f"   - {err}")
    sys.exit(1)

# ============================================================================
# 3. Check dataset and configuration files
# ============================================================================
print("\n[3/4] Checking Data & Config Files...")

required_files = [
    ("configs/experiment.yaml", "YAML config"),
    ("notebooks/06_llava_training_colab.ipynb", "Jupyter notebook"),
    ("LVLM_INTEGRATION_GUIDE.md", "Integration guide"),
    ("requirements.txt", "Requirements"),
]

for file_path, description in required_files:
    if Path(file_path).exists():
        file_size = Path(file_path).stat().st_size
        if file_size > 0:
            print(f"  ✓ {description}: {file_path} ({file_size} bytes)")
        else:
            print(f"  ✗ {description}: {file_path} (empty)")
    else:
        print(f"  ⚠ {description}: {file_path} (not found)")

# ============================================================================
# 4. Verify directory structure
# ============================================================================
print("\n[4/4] Checking Directory Structure...")

required_dirs = [
    "models",
    "data",
    "experiments",
    "notebooks",
    "configs",
]

for dir_name in required_dirs:
    if Path(dir_name).is_dir():
        file_count = len(list(Path(dir_name).glob("*.py"))) + len(list(Path(dir_name).glob("*.ipynb"))) + len(list(Path(dir_name).glob("*.yaml")))
        print(f"  ✓ {dir_name}/ ({file_count} files)")
    else:
        print(f"  ✗ {dir_name}/ (not found)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("✓ CODE VALIDATION PASSED!")
print("="*80)

print("""
All source files have valid Python syntax and expected class/function definitions.

Next steps:
1. Install PyTorch dependencies:
   pip install torch torchvision transformers peft pillow numpy tqdm "git+https://github.com/openai/CLIP.git"

2. Run functional tests:
   python test_lvlm_local.py

3. Prepare your dataset:
   - Create LLaVA-format JSON with image paths and conversations
   - Structure: [{"image": "path.jpg", "conversations": [{"from": "human"/"gpt", "value": "text"}]}]

4. Start training:
   - Local: python experiments/train_llava.py
   - Colab: Use notebooks/06_llava_training_colab.ipynb

5. Research paper:
   - Refer to LVLM_INTEGRATION_GUIDE.md for math and paper structure
   - Collect results and qualitative examples during training
""")
