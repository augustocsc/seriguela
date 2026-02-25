import pandas as pd

def check_balance():
    df = pd.read_csv('phase_a_results_patched.csv')
    df = df[df['algorithm'] != 'best_of_n'].copy()
    
    print("="*60)
    print("SERIGUELA PHASE A - DATA BALANCE CHECK")
    print("="*60)
    
    # 1. Check marginal distributions of categorical variables
    params = ['model', 'problem', 'algorithm', 'reward', 'penalty', 'temperature', 'prompt', 'noise']
    
    print("\n--- MARGINAL DISTRIBUTIONS ---")
    for p in params:
        counts = df[p].value_counts()
        print(f"\nParameter: {p.upper()}")
        for val, count in counts.items():
            print(f"  {val:<20} {count:>6} runs ({count/len(df)*100:.1f}%)")
            
    # 2. Check full grid factorial balance
    print("\n--- FACTORIAL GRID BALANCE ---")
    # Expected number of total combinations
    unique_vals = {p: df[p].unique() for p in params}
    expected_combinations = 1
    for p, vals in unique_vals.items():
        expected_combinations *= len(vals)
        
    actual_combinations = len(df.groupby(params))
    
    print(f"Expected unique hyperparameter combinations (grid size): {expected_combinations}")
    print(f"Actual unique combinations present in dataset:           {actual_combinations}")
    
    if expected_combinations != actual_combinations:
        print("\nWARNING: The dataset is NOT a complete factorial grid!")
        print(f"Missing {expected_combinations - actual_combinations} combinations.")
    else:
        print("\nThe dataset covers the full factorial grid.")
        
    # 3. Check seed balance per configuration
    print("\n--- SEED BALANCE ---")
    seed_counts = df.groupby(params).size()
    sc_dist = seed_counts.value_counts().sort_index()
    
    print("Number of seeds | Number of configurations")
    print("-" * 45)
    for num_seeds, num_configs in sc_dist.items():
        print(f"       {num_seeds:<2}       |       {num_configs}")
        
    print(f"\nTotal RL Runs: {len(df)}")
    
if __name__ == '__main__':
    check_balance()
