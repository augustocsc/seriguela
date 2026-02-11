#!/bin/bash
# Launch all 3 prefix models (Base, Medium, Large) in parallel on AWS
# Usage: ./launch_all_prefix_models.sh

set -e

echo "============================================"
echo "Launching 3 Prefix Models in Parallel"
echo "============================================"

# Load credentials from ~/.tokens.txt
if [ -f ~/.tokens.txt ]; then
  WANDB_KEY=$(grep wandb ~/.tokens.txt | cut -d= -f2 | tr -d ' ')
  HF_TOKEN=$(grep huggingface ~/.tokens.txt | cut -d= -f2 | tr -d ' ')
else
  echo "Error: ~/.tokens.txt not found"
  echo "Create it with:"
  echo "  huggingface = hf_..."
  echo "  wandb = wandb_..."
  exit 1
fi

if [ -z "$HF_TOKEN" ] || [ -z "$WANDB_KEY" ]; then
  echo "Error: Tokens not found in ~/.tokens.txt"
  exit 1
fi

echo "✓ Credentials loaded from ~/.tokens.txt"
echo ""

# Launch Base (g5.xlarge)
echo "[1/3] Launching Base (124M) on g5.xlarge..."
bash scripts/aws/launch_base_prefix_training.sh \
  --wandb-key "$WANDB_KEY" \
  --hf-token "$HF_TOKEN" \
  > launch_base_prefix.log 2>&1 &
BASE_PID=$!
echo "  → Base launched (PID: $BASE_PID)"
echo "  → Log: launch_base_prefix.log"

sleep 10  # Stagger launches to avoid AWS rate limits

# Launch Medium (g5.xlarge)
echo ""
echo "[2/3] Launching Medium (355M) on g5.xlarge..."
bash scripts/aws/launch_medium_prefix_training.sh \
  --wandb-key "$WANDB_KEY" \
  --hf-token "$HF_TOKEN" \
  > launch_medium_prefix.log 2>&1 &
MEDIUM_PID=$!
echo "  → Medium launched (PID: $MEDIUM_PID)"
echo "  → Log: launch_medium_prefix.log"

sleep 10

# Launch Large (g5.2xlarge)
echo ""
echo "[3/3] Launching Large (774M) on g5.2xlarge..."
bash scripts/aws/launch_large_prefix_training.sh \
  --wandb-key "$WANDB_KEY" \
  --hf-token "$HF_TOKEN" \
  > launch_large_prefix.log 2>&1 &
LARGE_PID=$!
echo "  → Large launched (PID: $LARGE_PID)"
echo "  → Log: launch_large_prefix.log"

echo ""
echo "============================================"
echo "All 3 launches initiated!"
echo "============================================"
echo ""
echo "Process IDs:"
echo "  Base   (124M): PID $BASE_PID"
echo "  Medium (355M): PID $MEDIUM_PID"
echo "  Large  (774M): PID $LARGE_PID"
echo ""
echo "Check launch progress:"
echo "  tail -f launch_base_prefix.log"
echo "  tail -f launch_medium_prefix.log"
echo "  tail -f launch_large_prefix.log"
echo ""

# Wait for all launches to complete
echo "Waiting for all launches to complete..."
wait $BASE_PID $MEDIUM_PID $LARGE_PID

echo ""
echo "============================================"
echo "All instances launched successfully!"
echo "============================================"
echo ""

# Show running instances
echo "Running instances:"
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=seriguela-*-prefix-training-*" \
           "Name=instance-state-name,Values=running" \
  --query "Reservations[*].Instances[*].[InstanceId,InstanceType,PublicIpAddress,Tags[?Key=='Name'].Value|[0]]" \
  --output table

echo ""
echo "============================================"
echo "Monitoring Training"
echo "============================================"
echo ""
echo "Wandb Dashboard:"
echo "  https://wandb.ai/YOUR_USERNAME/seriguela"
echo ""
echo "SSH into instances (get IPs from table above):"
echo "  ssh -i ~/.ssh/chave-gpu.pem ubuntu@<IP>"
echo "  tail -f ~/training_base_prefix.log"
echo "  tail -f ~/training_medium_prefix.log"
echo "  tail -f ~/training_large_prefix.log"
echo ""
echo "Check completion:"
echo "  ssh ubuntu@<IP> 'test -f ~/.training_complete && echo DONE || echo Running'"
echo ""
echo "============================================"
echo "⚠️  CRITICAL: Stop Instances When Done!"
echo "============================================"
echo ""
echo "To stop ALL prefix training instances:"
echo "  aws ec2 stop-instances --instance-ids \$(aws ec2 describe-instances \\"
echo "    --filters 'Name=tag:Name,Values=seriguela-*-prefix-training-*' \\"
echo "              'Name=instance-state-name,Values=running' \\"
echo "    --query 'Reservations[*].Instances[*].InstanceId' --output text)"
echo ""
echo "To download models:"
echo "  scp -i ~/.ssh/chave-gpu.pem -r ubuntu@<IP>:~/seriguela/output/gpt2_base_prefix_682k ./"
echo "  scp -i ~/.ssh/chave-gpu.pem -r ubuntu@<IP>:~/seriguela/output/gpt2_medium_prefix_682k ./"
echo "  scp -i ~/.ssh/chave-gpu.pem -r ubuntu@<IP>:~/seriguela/output/gpt2_large_prefix_682k ./"
echo ""
echo "============================================"
echo "Estimated Costs:"
echo "  Base   (g5.xlarge):  2-3h × $1.006/h = $2-3"
echo "  Medium (g5.xlarge):  3-4h × $1.006/h = $3-4"
echo "  Large  (g5.2xlarge): 4-5h × $1.212/h = $5-6"
echo "  Total (parallel): ~$10-13 USD"
echo "============================================"
