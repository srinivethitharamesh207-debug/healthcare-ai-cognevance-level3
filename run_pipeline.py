"""
run_pipeline.py
---------------
Master script to run the complete ML pipeline end-to-end.

Steps:
    1. Generate synthetic healthcare dataset
    2. Run preprocessing and feature engineering
    3. Train all 5 models and compare
    4. Print final summary

Usage:
    python run_pipeline.py
"""

import subprocess
import sys
import os

def run(script, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print('='*60)
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"[ERROR] {script} failed.")
        sys.exit(1)
    print(f"[OK] {label} complete.")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    run('scripts/generate_dataset.py', 'Step 1/3: Dataset Generation')
    run('scripts/preprocessing.py',   'Step 2/3: Preprocessing & Feature Engineering')
    run('scripts/train_models.py',    'Step 3/3: Model Training & Evaluation')

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print('='*60)
    print("  Reports   : reports/")
    print("  Models    : models/")
    print("  Dashboard : open dashboard/index.html in browser")
    print("  API       : uvicorn api.main:app --reload --port 8000")
    print('='*60)
