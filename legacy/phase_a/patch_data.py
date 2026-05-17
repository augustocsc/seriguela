import os
import csv
import json
import yaml
from pathlib import Path

def extract_missing_params():
    target_dir = Path("c:/Users/madeinweb/seriguela/2_training/reinforcement/phase_a_extracted/wandb")
    csv_file = "phase_a_all_results.csv"
    output_csv = "phase_a_results_patched.csv"
    
    print("Loading original CSV...")
    df = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            df.append(row)
            
    print(f"Loaded {len(df)} rows. Scanning wandb directories to patch missing 'reward' and 'penalty'...")
    
    # Create mapping of run_id (the random string) to index
    run_id_map = {}
    for i, row in enumerate(df):
        run_id = row['run_name'].split('-')[-1]
        run_id_map[run_id] = i

    patched_count = 0
    
    # Iterate through wandb dirs
    for run_dir in target_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("run-"):
            continue
            
        run_id = run_dir.name.split('-')[-1]
        if run_id not in run_id_map:
            continue
            
        config_path = run_dir / "files" / "config.yaml"
        if not config_path.exists():
            continue
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                # Basic parsing to avoid slow full YAML load if possible, 
                # but YAML is safer given the structure
                config = yaml.safe_load(f)
                
            # The args are nested weirdly in wandb config.yaml
            args = config.get('_wandb', {}).get('value', {}).get('e', {})
            # Get the first key under 'e' (writerId)
            if not args:
                 continue
            
            writer_key = list(args.keys())[0]
            cmd_args = args[writer_key].get('args', [])
            
            reward = "unknown"
            penalty = "unknown"
            
            for i in range(len(cmd_args)):
                if cmd_args[i] == '--reward' and i + 1 < len(cmd_args):
                    reward = cmd_args[i+1]
                elif cmd_args[i] == '--penalty' and i + 1 < len(cmd_args):
                    penalty = cmd_args[i+1]
                    
            idx = run_id_map[run_id]
            df[idx]['reward'] = reward
            df[idx]['penalty'] = penalty
            patched_count += 1
            
        except Exception as e:
            print(f"Error parsing {config_path}: {e}")
            
    print(f"Patched {patched_count} runs. Saving to {output_csv}...")
    
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        if df:
            writer = csv.DictWriter(f, fieldnames=df[0].keys())
            writer.writeheader()
            writer.writerows(df)
            
    print("Done! Now running deep analysis on the patched data.")
    
if __name__ == '__main__':
    extract_missing_params()
