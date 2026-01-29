#!/bin/bash
# Setup script for AWS g5.xlarge instance (Deep Learning AMI Ubuntu)
# Project: Seriguela - GPT-2 Fine-tuning for Symbolic Regression

set -e  # Exit on error

echo "=========================================="
echo "Seriguela AWS Setup Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. System update
print_status "Updating system packages..."
sudo apt-get update -y

# 2. Install essential packages
print_status "Installing essential packages..."
sudo apt-get install -y git htop tmux nvtop

# 3. Check CUDA and GPU
print_status "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    print_status "GPU detected successfully!"
else
    print_error "nvidia-smi not found. GPU might not be available."
    exit 1
fi

# 4. Check Python
print_status "Checking Python version..."
python3 --version

# 5. Clone repository (if not already cloned)
REPO_DIR="$HOME/seriguela"
if [ -d "$REPO_DIR" ]; then
    print_status "Repository already exists at $REPO_DIR"
    cd "$REPO_DIR"
    git pull
else
    print_status "Cloning repository..."
    git clone https://github.com/augustocsc/seriguela.git "$REPO_DIR"
    cd "$REPO_DIR"
fi

# 6. Create virtual environment
print_status "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created."
else
    print_status "Virtual environment already exists."
fi

# 7. Activate virtual environment and install dependencies
print_status "Installing Python dependencies..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (adjust version if needed)
print_status "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project requirements
print_status "Installing project requirements..."
pip install -r requirements.txt

# Install additional dependencies for training
print_status "Installing additional training dependencies..."
pip install wandb accelerate bitsandbytes scipy

# 8. Verify installation
print_status "Verifying PyTorch CUDA installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 9. Setup environment variables reminder
echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Set your environment variables:"
echo "   export HF_TOKEN='your_huggingface_token'"
echo "   export WANDB_API_KEY='your_wandb_api_key'"
echo ""
echo "2. Or create a .env file in the project root:"
echo "   echo 'HF_TOKEN=your_token' > .env"
echo "   echo 'WANDB_API_KEY=your_key' >> .env"
echo ""
echo "3. Login to wandb (optional):"
echo "   wandb login"
echo ""
echo "4. Run a test training:"
echo "   cd $REPO_DIR"
echo "   source venv/bin/activate"
echo "   python scripts/train.py --help"
echo ""
echo "5. Or run the full training workflow:"
echo "   bash scripts/aws/run_all_training.sh"
echo ""
