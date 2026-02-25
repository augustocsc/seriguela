$ErrorActionPreference = "Stop"

$models = @("augustocsc/gpt2_base_infix_682k", "augustocsc/gpt2_base_prefix_682k")
$problems = @("nguyen_1", "nguyen_9")
$algorithms = @("best_of_n", "bon_grpo", "pure_ppo")
$seeds = @(42, 43, 44, 45, 46)

$total = $models.Length * $problems.Length * $algorithms.Length * $seeds.Length
$count = 0

Write-Host "=============================================="
Write-Host "SERIGUELA PHASE A-v2 RIGOROUS MASTER'S RUNNER"
Write-Host "Total Configurations: $total"
Write-Host "=============================================="

foreach ($model in $models) {
    foreach ($problem in $problems) {
        foreach ($algo in $algorithms) {
            foreach ($seed in $seeds) {
                $count++
                Write-Host "`n[$count/$total] Running: $algo | $problem | $model | seed:$seed"
                
                # We enforce the robust grid parameters discovered in deep analysis
                # reward=sr_ic, penalty=gradient, temp=linear_annealing, prompt=standard
                
                python ..\run_experiment.py `
                    --algorithm $algo `
                    --model $model `
                    --problem $problem `
                    --reward sr_ic `
                    --penalty gradient `
                    --temperature linear_annealing `
                    --prompt_type standard `
                    --max_steps 5000 `
                    --batch_size 32 `
                    --seeds $seed `
                    --noise_type none `
                    --use_wandb `
                    --wandb_project seriguela_phase_a_v2_rigorous
                    
                if ($LASTEXITCODE -ne 0) {
                    Write-Warning "Run failed! $algo on $problem with seed $seed"
                    # We don't stop the whole script to allow other runs to complete
                }
            }
        }
    }
}

Write-Host "=============================================="
Write-Host "PHASE A-v2 RIGOROUS GRID COMPLETED"
Write-Host "=============================================="
