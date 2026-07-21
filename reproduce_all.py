import subprocess
import sys
import os
import time
import glob

def clean_environment():
    print("=" * 60)
    print("CLEANING INTERMEDIATE FILES")
    print("=" * 60)
    
    patterns_to_clean = [
        "data/processed/paper/*.csv",
        "data/processed/paper/*.pkl",
        "models/*.json",
        "reports/figures/paper/*.png"
    ]
    
    deleted_files = []
    for pattern in patterns_to_clean:
        files = glob.glob(pattern)
        for f in files:
            try:
                os.remove(f)
                deleted_files.append(f)
            except Exception as e:
                print(f"Failed to delete {f}: {e}")
                
    if deleted_files:
        print(f"Deleted {len(deleted_files)} files:")
        for f in deleted_files:
            print(f"  - {f}")
    else:
        print("No intermediate files found to delete.")
    print("\n")

def run_step(step_num, total_steps, name, script_path):
    print(f"[{step_num}/{total_steps}] {name}...")
    start_time = time.time()
    
    try:
        # Run without capture_output to stream stdout/stderr in real-time
        subprocess.run([sys.executable, script_path], check=True)
        elapsed = time.time() - start_time
        print(f"[{step_num}/{total_steps}] {name}... done ({elapsed:.1f}s)\n")
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n[{step_num}/{total_steps}] {name}... FAILED ({elapsed:.1f}s)")
        print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
        sys.exit(1)

def main():
    print("=" * 60)
    print("SOLAR PV FAULT DIAGNOSIS - REPRODUCTION PIPELINE")
    print("=" * 60)
    
    # Ensure PYTHONPATH includes the project root
    os.environ['PYTHONPATH'] = os.path.abspath(os.path.dirname(__file__))
    
    clean_environment()
    
    pipeline = [
        ("Preprocessing", "src/preprocessing.py"),
        ("Expected Power Model", "src/expected_power_model.py"),
        ("Taxonomy (Rule-based)", "src/taxonomy.py"),
        ("Masked Loss Calculation", "src/masked_loss.py"),
        ("Carbon Quantification", "src/carbon_quantification.py"),
        ("Baselines & Fault Injection", "src/baselines.py"),
        ("Generate All Figures", "figures/generate_all_figures.py")
    ]
    
    total = len(pipeline)
    
    for idx, (name, path) in enumerate(pipeline, 1):
        if not os.path.exists(path):
            print(f"\nERROR: Script '{path}' not found!")
            sys.exit(1)
            
        run_step(idx, total, name, path)
        
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("All datasets, models, and figures have been generated.")
    print("=" * 60)

if __name__ == "__main__":
    main()
