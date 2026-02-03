#!/bin/bash
# Auto-launch medium training using saved credentials or prompt

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# Check for credentials in order of preference:
# 1. Command line args
# 2. Environment variables
# 3. Credentials file
# 4. Prompt user

CRED_FILE="${HOME}/.seriguela/credentials"

# Try to load from credentials file
if [ -f "$CRED_FILE" ]; then
    print_status "Loading credentials from $CRED_FILE"
    source "$CRED_FILE"
fi

# Get WANDB_KEY
if [ -z "$WANDB_KEY" ] && [ -n "$WANDB_API_KEY" ]; then
    WANDB_KEY="$WANDB_API_KEY"
fi

if [ -z "$WANDB_KEY" ]; then
    print_warning "Wandb API key not found"
    echo -n "Enter your Wandb API key: "
    read -s WANDB_KEY
    echo ""
fi

# Get HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    print_warning "HuggingFace token not found (optional)"
    echo -n "Enter your HuggingFace token (or press Enter to skip): "
    read -s HF_TOKEN
    echo ""
fi

# Launch training
print_status "Launching GPT-2 Medium training on AWS..."

bash "$(dirname "$0")/launch_medium_training.sh" \
    --wandb-key "$WANDB_KEY" \
    --hf-token "$HF_TOKEN"

# Offer to save credentials
if [ ! -f "$CRED_FILE" ]; then
    echo ""
    echo -n "Save credentials for next time? [y/N]: "
    read SAVE_CREDS
    if [ "$SAVE_CREDS" = "y" ] || [ "$SAVE_CREDS" = "Y" ]; then
        mkdir -p "$(dirname "$CRED_FILE")"
        cat > "$CRED_FILE" << EOF
# Seriguela API credentials
export WANDB_API_KEY="$WANDB_KEY"
export HF_TOKEN="$HF_TOKEN"
EOF
        chmod 600 "$CRED_FILE"
        print_status "Credentials saved to $CRED_FILE"
    fi
fi
