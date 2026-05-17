import pandas as pd
import numpy as np

def run_analysis():
    # Load dataset
    df = pd.read_csv('phase_a_results_patched.csv')
    df = df.fillna(0.0)
    
    # Exclude best_of_n which crashed
    df_rl = df[df['algorithm'] != 'best_of_n'].copy()
    
    # Fix negative R2 values for average calculations (clamp to 0)
    df_rl['best_r2_clamped'] = df_rl['best_r2'].clip(lower=0)
    df_rl['final_r2_clamped'] = df_rl['final_r2'].clip(lower=0)
    
    # Calculate stability: Final R2 - Best R2 (How much did it collapse at the end)
    df_rl['collapse'] = df_rl['best_r2_clamped'] - df_rl['final_r2_clamped']
    
    # Count replicas (group by all hyperparameters)
    group_cols = ['model', 'problem', 'algorithm', 'reward', 'penalty', 'temperature', 'prompt', 'noise']
    df_agg = df_rl.groupby(group_cols).agg(
        num_seeds=('best_r2', 'count'),
        mean_best_r2=('best_r2_clamped', 'mean'),
        std_best_r2=('best_r2_clamped', 'std'),
        max_best_r2=('best_r2_clamped', 'max'),
        mean_collapse=('collapse', 'mean'),
        gte99_runs=('best_r2', lambda x: (x >= 0.99).sum())
    ).reset_index()

    print("="*80)
    print("SERIGUELA PHASE A DEEP DIVE RESEARCH ANALYSIS")
    print("="*80)
    
    print("\n1. REPLICATIONS & DATA INTEGRITY")
    print(f"Total RL runs: {len(df_rl)}")
    print(f"Unique Hyperparameter Configurations: {len(df_agg)}")
    print(f"Average seeds per configuration: {df_agg['num_seeds'].mean():.2f}")
    
    print("\n2. PPO vs GRPO (Including BoN variants)")
    ppo_mask = df_rl['algorithm'].str.contains('ppo')
    grpo_mask = df_rl['algorithm'].str.contains('grpo')
    
    print(f"{'Metric':<20} | {'PPO Family':<15} | {'GRPO Family':<15}")
    print("-"*55)
    print(f"{'Mean Best R²':<20} | {df_rl.loc[ppo_mask, 'best_r2_clamped'].mean():<15.4f} | {df_rl.loc[grpo_mask, 'best_r2_clamped'].mean():<15.4f}")
    print(f"{'Runs R² ≥ 0.99':<20} | {df_rl.loc[ppo_mask, 'best_r2'].apply(lambda x: x >= 0.99).sum():<15} | {df_rl.loc[grpo_mask, 'best_r2'].apply(lambda x: x >= 0.99).sum():<15}")
    print(f"{'Mean Collapse':<20} | {df_rl.loc[ppo_mask, 'collapse'].mean():<15.4f} | {df_rl.loc[grpo_mask, 'collapse'].mean():<15.4f}")
    
    print("\n3. REWARD ALGORITHM ABLATION (What is the best reward function?)")
    # If reward is empty string or nan in some rows, fill it
    df_rl['reward'] = df_rl['reward'].replace(0.0, 'unknown').replace('', 'unknown')
    reward_perf = df_rl.groupby('reward')['best_r2_clamped'].agg(['mean', 'max', 'count', lambda x: (x >= 0.99).sum()]).rename(columns={'<lambda_0>': '>=0.99'})
    print(reward_perf.to_string())

    print("\n4. PENALTY STRATEGY ABLATION (Binary vs Gradient)")
    df_rl['penalty'] = df_rl['penalty'].replace(0.0, 'unknown').replace('', 'unknown')
    penalty_perf = df_rl.groupby('penalty')['best_r2_clamped'].agg(['mean', 'max', 'count', lambda x: (x >= 0.99).sum()]).rename(columns={'<lambda_0>': '>=0.99'})
    print(penalty_perf.to_string())

    print("\n5. LEARNING STABILITY (Did models improve or behavior was random?)")
    # A model that improves usually has final_r2 close to best_r2. 
    # If final_r2 is 0 or much lower than best_r2, it means policy collapsed.
    collapsed_runs = (df_rl['collapse'] > 0.5).sum()
    stable_runs = (df_rl['collapse'] < 0.1).sum()
    print(f"Highly stable runs (Final R² within 0.1 of Best): {stable_runs} ({stable_runs/len(df_rl)*100:.1f}%)")
    print(f"Collapsed runs (Final R² dropped >0.5 from Best): {collapsed_runs} ({collapsed_runs/len(df_rl)*100:.1f}%)")
    
    print("\nStability by Algorithm (Mean collapse - lower is more stable):")
    print(df_rl.groupby('algorithm')['collapse'].mean().sort_values().to_string())

    print("\n6. HIGHEST PERFORMING ROBUST CONFIGURATIONS")
    print("(Averaged across seeds, requiring at least 2 seeds to consider variance)")
    robust_configs = df_agg[df_agg['num_seeds'] >= 2].sort_values('mean_best_r2', ascending=False).head(10)
    for i, row in robust_configs.iterrows():
        print(f"\nModel: {row['model']}, Problem: {row['problem']}")
        print(f"Algo: {row['algorithm']}, Reward: {row['reward']}, Pen: {row['penalty']}, Temp: {row['temperature']}, Prompt: {row['prompt']}")
        print(f"Mean R²: {row['mean_best_r2']:.4f} ± {row['std_best_r2']:.4f}, Runs: {row['num_seeds']}")
        
    print("\n7. PROMPT ANALYSIS (Interactions with Problems and Algorithms)")
    print("--- Prompt vs Problem (Mean Best R²) ---")
    prompt_prob_pivot = df_rl.pivot_table(index='prompt', columns='problem', values='best_r2_clamped', aggfunc='mean')
    print(prompt_prob_pivot.to_string())
    
    print("\n--- Prompt vs Algorithm (Mean Best R²) ---")
    prompt_algo_pivot = df_rl.pivot_table(index='prompt', columns='algorithm', values='best_r2_clamped', aggfunc='mean')
    print(prompt_algo_pivot.to_string())
    
if __name__ == '__main__':
    run_analysis()
