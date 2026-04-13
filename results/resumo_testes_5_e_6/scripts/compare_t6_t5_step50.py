import json
import glob
import os

def get_test5_step50_r2(algo, problem, seed):
    # Pure variants from Test 5
    pure_algo = algo.replace("bon_", "pure_")
    files = glob.glob(f"results/pre_phase__t5/aggregate_{pure_algo}_{problem}_seed{seed}.json")
    if not files:
        return None
    
    with open(files[0], 'r') as f:
        data = json.load(f)
        
    history = data.get('individual_results', [{}])[0].get('history', [])
    for step_data in history:
        if step_data.get('step') == 3:
            return step_data.get('best_r2', 0)
    return None

def main():
    print(f"{'Algorithm'.ljust(15)} | {'Problem'.ljust(10)} | {'Seed'.ljust(4)} | {'T6 BoN R2 (3.2k)'.ljust(20)} | {'T5 Pure R2 (3.0k)'.ljust(25)} | {'Winner'.ljust(10)}")
    print("-" * 100)
    
    files = glob.glob('results/pre_phase__t6_20260310_023310-teste/*.json')
    files.sort()
    
    for f in files:
        with open(f, 'r') as file:
            data = json.load(file)
            algo = data.get('algorithm', '')
            prob = data.get('problem', '')
            seed = str(data.get('seeds', [''])[0])
            
            # T6 Result (BoN, Batch 64, Step 50)
            bon_r2 = data.get('max_train_r2', 0)
            
            # T5 Result (Pure, Batch 1024, Step 50)
            pure_r2 = get_test5_step50_r2(algo, prob, seed)
            
            formatted_bon = f"{bon_r2:.4f}" if bon_r2 > 0 else "0.0"
            if pure_r2 is not None:
                formatted_pure = f"{pure_r2:.4f}" if pure_r2 > 0 else "0.0"
                winner = "BoN" if bon_r2 > pure_r2 else "Pure"
                if bon_r2 == pure_r2: winner = "Tie"
            else:
                formatted_pure = "N/A"
                winner = "N/A"
                
            print(f"{algo.ljust(15)} | {prob.ljust(10)} | {seed.ljust(4)} | {formatted_bon.ljust(20)} | {formatted_pure.ljust(25)} | {winner.ljust(10)}")

if __name__ == '__main__':
    main()
