#!/bin/bash
# Monitor GPT-2 Medium training in real-time

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get last instance info
INFO_FILE="${HOME}/.seriguela/medium_instance_info.txt"

if [ ! -f "$INFO_FILE" ]; then
    echo "No medium training instance found!"
    echo "Launch one first with: bash scripts/aws/launch_medium_training.sh"
    exit 1
fi

INSTANCE_ID=$(grep "Instance ID:" "$INFO_FILE" | cut -d' ' -f3)
PUBLIC_IP=$(grep "Public IP:" "$INFO_FILE" | cut -d' ' -f3)
KEY_NAME=$(grep "Key Name:" "$INFO_FILE" | cut -d' ' -f3)

echo "=========================================="
echo "Monitoring GPT-2 Medium Training"
echo "=========================================="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo ""

# Check if instance is running
STATUS=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query "Reservations[0].Instances[0].State.Name" \
    --output text 2>/dev/null)

if [ "$STATUS" != "running" ]; then
    echo "Instance is not running (status: $STATUS)"
    exit 1
fi

echo -e "${GREEN}Instance is running${NC}"
echo ""

# Check if training is complete
echo "Checking training status..."
COMPLETE=$(ssh -i ~/.ssh/${KEY_NAME}.pem -o StrictHostKeyChecking=no ubuntu@${PUBLIC_IP} \
    'test -f ~/.training_complete && echo "yes" || echo "no"' 2>/dev/null)

if [ "$COMPLETE" == "yes" ]; then
    echo -e "${GREEN}Training is COMPLETE!${NC}"
    echo ""
    echo "Results:"
    ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} 'cat ~/training_results.txt'
    echo ""
    echo "Download model with:"
    echo "  scp -i ~/.ssh/${KEY_NAME}.pem -r ubuntu@${PUBLIC_IP}:~/seriguela/output/gpt2_medium_700K_json ./"
    exit 0
fi

echo -e "${YELLOW}Training in progress...${NC}"
echo ""
echo "Options:"
echo "  1. Watch live logs (Ctrl+C to exit)"
echo "  2. Check last 50 lines"
echo "  3. Check GPU usage"
echo "  4. Exit"
echo ""
read -p "Select option [1-4]: " OPTION

case $OPTION in
    1)
        echo ""
        echo "Following training logs (Ctrl+C to stop)..."
        echo ""
        ssh -i ~/.ssh/${KEY_NAME}.pem -t ubuntu@${PUBLIC_IP} \
            'tail -f /home/ubuntu/training_medium.log'
        ;;
    2)
        echo ""
        ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} \
            'tail -n 50 /home/ubuntu/training_medium.log'
        ;;
    3)
        echo ""
        ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} \
            'nvidia-smi'
        ;;
    4)
        exit 0
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac
