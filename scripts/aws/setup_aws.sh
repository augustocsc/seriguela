#!/bin/bash
# Setup script for AWS g5.xlarge instance (Deep Learning AMI Ubuntu)
# Project: Seriguela - GPT-2 Fine-tuning for Symbolic Regression
# Optimized for faster setup

set -e

echo "=========================================="
echo "Seriguela AWS Setup Script (Optimized)"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
REPO_URL="https://github.com/augustocsc/seriguela.git"
REPO_DIR="$HOME/seriguela"
PYTHON_VERSION="python3"

# Check GPU
print_status "Checking GPU..."
if ! nvidia-smi &>/dev/null; then
    print_error "GPU not detected!"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Install system dependencies (minimal)
print_status "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip git htop

# Clone or update repository
if [ -d "$REPO_DIR" ]; then
    print_status "Updating repository..."
    cd "$REPO_DIR" && git pull
else
    print_status "Cloning repository..."
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# Setup virtual environment
print_status "Setting up virtual environment..."
$PYTHON_VERSION -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies in one step
print_status "Installing all dependencies (this may take a few minutes)..."
pip install --upgrade pip -q
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 -q

# Verify installation
print_status "Verifying installation..."
python -c "
import torch
import transformers
import peft
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
"

echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next: Configure tokens in .env file:"
echo "  echo 'HF_TOKEN=your_token' > .env"
echo "  echo 'WANDB_API_KEY=your_key' >> .env"
echo ""
echo "Then run training:"
echo "  source venv/bin/activate"
echo "  bash scripts/aws/run_all_training.sh --test-only"
echo ""
