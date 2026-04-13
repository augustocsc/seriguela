import os
import glob
import shutil
import pathlib

def merge_results():
    print("="*70)
    print("MERGING VALID RESULTS FOR FINAL ANALYSIS")
    print("="*70)
    
    # Paths
    test5_dir = "results/pre_phase__t5"
    test6_base = "/content/drive/MyDrive/seriguela_results" # Colab Drive
    local_test6_base = "../../results" # Local fallback
    
    output_dir = "results/pre_phase__t5_and_t6_merged"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}\n")
    
    files_copied = 0
    
    # 1. Copy valid Test 5 (Pure GRPO and Pure PPO only)
    print("--- 1. Extracting valid Baselines from Test 5 ---")
    if os.path.exists(test5_dir):
        t5_files = glob.glob(f"{test5_dir}/aggregate_pure_*.json")
        for f in t5_files:
            filename = os.path.basename(f)
            dest = os.path.join(output_dir, filename)
            shutil.copy2(f, dest)
            files_copied += 1
        print(f"Copied {len(t5_files)} Pure baseline files from Test 5.")
    else:
        print(f"Warning: Test 5 directory not found at {test5_dir}")
        
    # 2. Find latest Test 6 folder
    print("\n--- 2. Extracting fixed BoN from Test 6 ---")
    
    search_dirs = [test6_base, local_test6_base, "results"]
    t6_folder = None
    
    for base in search_dirs:
        if os.path.exists(base):
            entries = glob.glob(os.path.join(base, "pre_phase__t6_*"))
            folders = sorted([e for e in entries if os.path.isdir(e)], reverse=True)
            if folders:
                t6_folder = folders[0]
                break
                
    if t6_folder:
        print(f"Found latest Test 6 folder: {t6_folder}")
        t6_files = glob.glob(os.path.join(t6_folder, "aggregate_*.json"))
        for f in t6_files:
            filename = os.path.basename(f)
            dest = os.path.join(output_dir, filename)
            shutil.copy2(f, dest)
            files_copied += 1
        print(f"Copied {len(t6_files)} BoN files from Test 6.")
    else:
        print("Warning: Could not find any Test 6 folders.")
        
    print("\n" + "="*70)
    print(f"MERGE COMPLETE: {files_copied} total files ready in '{output_dir}'")
    print("You can now run your analysis scripts pointing to this folder.")
    print("="*70)

if __name__ == "__main__":
    merge_results()
