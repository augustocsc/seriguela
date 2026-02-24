# Check Missing Configs - Phase A

Scripts to verify which configs are missing from the completed Phase A experiments.

## Quick Start

### Option 1: Check all 6 instances automatically

```bash
cd 2_training/reinforcement
python check_all_missing.py
```

This will:
1. Start all 6 instances
2. Check each instance for missing configs
3. Save results to `missing_*.json` files
4. Stop all instances
5. Print summary

### Option 2: Check one instance manually

```bash
# Start instance
aws ec2 start-instances --instance-ids i-0ab8277c5128ef303

# Wait for it to start, get IP
aws ec2 describe-instances --instance-ids i-0ab8277c5128ef303 --query "Reservations[*].Instances[*].PublicIpAddress" --output text

# Check configs (replace IP)
python check_missing_configs.py --ssh ubuntu@IP_ADDRESS --output missing_base_infix_n1.json

# Stop instance
aws ec2 stop-instances --instance-ids i-0ab8277c5128ef303
```

### Option 3: Check local wandb directory

If you've downloaded the wandb folder locally:

```bash
python check_missing_configs.py --local /path/to/wandb --output missing_local.json
```

## Instance IDs

| Instance ID | Model-Problem | Name |
|-------------|---------------|------|
| i-0ab8277c5128ef303 | base_infix × nguyen_1 | rem_base_infix_n1 |
| i-0dcb39ad7278622ec | base_infix × nguyen_5 | rem_base_infix_n5 |
| i-00d7e518d26082914 | base_infix × nguyen_9 | rem_base_infix_n9 |
| i-0aeeb70b76c5dc7d8 | base_prefix × nguyen_1 | rem_base_prefix_n1 |
| i-073564e75558da6f3 | base_prefix × nguyen_5 | rem_base_prefix_n5 |
| i-09aadd345995e5611 | base_prefix × nguyen_9 | rem_base_prefix_n9 |

## Output Format

The script generates JSON files with missing configs in the same format as `remaining_base_configs.json`:

```json
[
  ["base_infix", "nguyen_1", "pure_ppo", "sr_ic", "gradient", "fixed_0.7", "oracle", 0.05],
  ["base_infix", "nguyen_1", "pure_ppo", "sr_ic", "gradient", "fixed_0.7", "oracle", 0.1],
  ...
]
```

Each config tuple contains:
1. Model (base_infix or base_prefix)
2. Problem (nguyen_1, nguyen_5, or nguyen_9)
3. Algorithm (best_of_n, bon_ppo, bon_grpo, pure_ppo, pure_grpo)
4. Reward (length_penalized, r2_clipped, sr_ic)
5. Penalty (binary, gradient)
6. Temperature (cosine_annealing, fixed_0.7, fixed_0.9, linear_annealing)
7. Prompt (distractor, oracle, standard)
8. Noise level (0.0, 0.01, 0.05, 0.1)

## Known Results

From previous check (base_infix × nguyen_1):
- Expected: 1,398 configs
- Completed: 1,368 configs
- Missing: 30 configs (all `pure_ppo` + `sr_ic` + `gradient`)

## Running Missing Configs

If you find missing configs, you can run them with:

```bash
# Create a JSON file with just the missing configs
# Then use run_remaining_experiment.py

python run_remaining_experiment.py \
  --model base_infix \
  --problem nguyen_1 \
  --remaining_file missing_base_infix_n1.json \
  --max_steps 5000 \
  --batch_size 32
```

## Notes

- The scripts require AWS CLI configured
- SSH key path defaults to `C:/Users/madeinweb/chave-gpu.pem`
- Instances cost ~$1/hour each when running
- The check takes ~5 minutes total (mostly waiting for instances to start)
