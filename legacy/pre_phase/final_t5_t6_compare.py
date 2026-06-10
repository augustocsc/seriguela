import json
import glob
import os

def main():
    merged_dir = "results/pre_phase__t5_and_t6_merged"
    
    print(f"{'Problem'.ljust(10)} | {'Seed'.ljust(4)} | {'BoN-GRPO'.ljust(15)} | {'Pure-GRPO'.ljust(15)} | {'BoN-PPO'.ljust(15)} | {'Pure-PPO'.ljust(15)}")
    print("-" * 85)
    
    results_map = {}
    
    # Load all files
    for f in glob.glob(os.path.join(merged_dir, "*.json")):
        with open(f, 'r') as file:
            data = json.load(file)
            algo = data.get('algorithm', '')
            prob = data.get('problem', '')
            seed = str(data.get('seeds', [''])[0])
            r2 = data.get('max_train_r2', 0)
            
            if prob not in results_map:
                results_map[prob] = {}
            if seed not in results_map[prob]:
                results_map[prob][seed] = {}
                
            results_map[prob][seed][algo] = r2

    probs = sorted(results_map.keys())
    for prob in probs:
        seeds = sorted(results_map[prob].keys())
        for seed in seeds:
            vals = results_map[prob][seed]
            
            def fmt(val):
                if val is None: return "N/A"
                if val < -100: return "Collapse"
                if val < 0: return f"{val:.2f}"
                return f"{val:.4f}"
            
            b_grpo = fmt(vals.get('bon_grpo'))
            p_grpo = fmt(vals.get('pure_grpo'))
            b_ppo = fmt(vals.get('bon_ppo'))
            p_ppo = fmt(vals.get('pure_ppo'))
            
            # Highlight winners with an asterisk
            if vals.get('bon_grpo', 0) > vals.get('pure_grpo', 0) and vals.get('bon_grpo', 0) > 0:
                b_grpo += "*"
            elif vals.get('pure_grpo', 0) > vals.get('bon_grpo', 0) and vals.get('pure_grpo', 0) > 0:
                p_grpo += "*"
                
            if vals.get('bon_ppo', 0) > vals.get('pure_ppo', 0) and vals.get('bon_ppo', 0) > 0:
                b_ppo += "*"
            elif vals.get('pure_ppo', 0) > vals.get('bon_ppo', 0) and vals.get('pure_ppo', 0) > 0:
                p_ppo += "*"
                
            print(f"{prob.ljust(10)} | {seed.ljust(4)} | {b_grpo.ljust(15)} | {p_grpo.ljust(15)} | {b_ppo.ljust(15)} | {p_ppo.ljust(15)}")
            
if __name__ == "__main__":
    main()
