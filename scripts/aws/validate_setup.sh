#!/bin/bash
# Validate Seriguela Training Setup
# This script validates that everything is configured correctly before training
# Usage: ./validate_setup.sh

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✅${NC} $1"; }
print_error() { echo -e "${RED}❌${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠️${NC}  $1"; }
print_header() { echo -e "\n${BLUE}========== $1 ==========${NC}"; }

ERRORS=0

print_header "Seriguela Setup Validation"

# Change to project directory
if [ -d "/home/ubuntu/seriguela" ]; then
    cd /home/ubuntu/seriguela
elif [ -d "$(pwd)/seriguela" ]; then
    cd seriguela
else
    cd .
fi

print_header "1. Python Environment"

# Check Python version
if python3 --version &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python installed: $PYTHON_VERSION"
else
    print_error "Python not found"
    ERRORS=$((ERRORS + 1))
fi

# Check venv
if [ -d "venv" ]; then
    print_success "Virtual environment exists"
    source venv/bin/activate
else
    print_error "Virtual environment not found"
    ERRORS=$((ERRORS + 1))
fi

# Check pip
if pip --version &> /dev/null; then
    PIP_VERSION=$(pip --version | cut -d' ' -f2)
    print_success "pip version: $PIP_VERSION"
else
    print_error "pip not found"
    ERRORS=$((ERRORS + 1))
fi

print_header "2. Python Packages"

# Check critical packages
PACKAGES=(
    "transformers:Hugging Face Transformers"
    "torch:PyTorch"
    "wandb:Weights & Biases"
    "peft:Parameter-Efficient Fine-Tuning"
    "datasets:Hugging Face Datasets"
)

for pkg_info in "${PACKAGES[@]}"; do
    IFS=':' read -r pkg_name pkg_desc <<< "$pkg_info"

    if python3 -c "import $pkg_name" &> /dev/null; then
        VERSION=$(python3 -c "import $pkg_name; print($pkg_name.__version__)" 2>/dev/null || echo "unknown")
        print_success "$pkg_desc ($pkg_name) - version $VERSION"
    else
        print_error "$pkg_desc ($pkg_name) not installed"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check Wandb version specifically
WANDB_VERSION=$(python3 -c "import wandb; print(wandb.__version__)" 2>/dev/null || echo "0.0.0")
REQUIRED_VERSION="0.24.0"

if python3 << VERSIONCHECK
import sys
from packaging import version
current = version.parse("$WANDB_VERSION")
required = version.parse("$REQUIRED_VERSION")
sys.exit(0 if current >= required else 1)
VERSIONCHECK
then
    print_success "Wandb version $WANDB_VERSION (>= $REQUIRED_VERSION required)"
else
    print_warning "Wandb version $WANDB_VERSION is older than recommended $REQUIRED_VERSION"
    print_warning "New API key format (wandb_v1_...) requires Wandb >= 0.24.0"
fi

print_header "3. Environment Variables"

# Load .env if exists
if [ -f ".env" ]; then
    source <(grep -v '^#' .env | sed 's/^/export /')
    print_success ".env file loaded"
else
    print_warning ".env file not found"
fi

# Check HF_TOKEN
if [ -n "$HF_TOKEN" ]; then
    TOKEN_LEN=${#HF_TOKEN}
    print_success "HF_TOKEN set ($TOKEN_LEN characters)"
else
    print_warning "HF_TOKEN not set (model won't be pushed to Hub)"
fi

# Check WANDB_API_KEY
if [ -n "$WANDB_API_KEY" ]; then
    KEY_LEN=${#WANDB_API_KEY}
    print_success "WANDB_API_KEY set ($KEY_LEN characters)"
else
    print_error "WANDB_API_KEY not set"
    ERRORS=$((ERRORS + 1))
fi

print_header "4. GPU / CUDA"

# Check nvidia-smi
if nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    print_success "GPU detected: $GPU_NAME ($GPU_MEMORY)"
else
    print_error "GPU not detected (nvidia-smi failed)"
    ERRORS=$((ERRORS + 1))
fi

# Check CUDA
if python3 -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
    CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
    print_success "CUDA available: version $CUDA_VERSION ($GPU_COUNT GPU(s))"
else
    print_error "CUDA not available in PyTorch"
    ERRORS=$((ERRORS + 1))
fi

print_header "5. Wandb Authentication"

if [ -n "$WANDB_API_KEY" ]; then
    if python3 << WANDBCHECK
import wandb
import sys
try:
    result = wandb.login(key="$WANDB_API_KEY", relogin=True)
    if result:
        print("Login successful")
        sys.exit(0)
    else:
        print("Login failed")
        sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
WANDBCHECK
    then
        print_success "Wandb authentication successful"

        # Get user info
        WANDB_USER=$(python3 << 'GETUSER'
import wandb
try:
    api = wandb.Api()
    print(api.viewer.get("username", "unknown"))
except:
    print("unknown")
GETUSER
)
        print_success "Logged in as: $WANDB_USER"
    else
        print_error "Wandb authentication failed"
        ERRORS=$((ERRORS + 1))
    fi
else
    print_warning "Skipping Wandb auth (no API key)"
fi

print_header "6. HuggingFace Authentication"

if [ -n "$HF_TOKEN" ]; then
    if python3 << HFCHECK
from huggingface_hub import HfApi
import sys
try:
    api = HfApi(token="$HF_TOKEN")
    user = api.whoami()
    print(f"Login successful: {user.get('name', 'unknown')}")
    sys.exit(0)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
HFCHECK
    then
        print_success "HuggingFace authentication successful"
    else
        print_error "HuggingFace authentication failed"
        ERRORS=$((ERRORS + 1))
    fi
else
    print_warning "Skipping HF auth (no token)"
fi

print_header "7. Dataset Access"

# Test dataset loading
if python3 << DATASETCHECK
from datasets import load_dataset
import sys
try:
    # Quick test load (just get info, don't download)
    ds = load_dataset("augustocsc/sintetico_natural", split="train", streaming=True)
    print("Dataset accessible")
    sys.exit(0)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
DATASETCHECK
then
    print_success "Dataset accessible: augustocsc/sintetico_natural"
else
    print_warning "Could not verify dataset access (may require authentication)"
fi

print_header "8. Scripts"

SCRIPTS=(
    "scripts/train.py"
    "scripts/evaluate.py"
    "scripts/generate.py"
    "scripts/aws/monitor_training_auto.sh"
    "scripts/aws/analyze_model.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        print_success "$script exists"
    else
        print_warning "$script not found"
    fi
done

# Final summary
print_header "Validation Summary"
echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                      ║${NC}"
    echo -e "${GREEN}║    ✅ ALL VALIDATIONS PASSED ✅     ║${NC}"
    echo -e "${GREEN}║                                      ║${NC}"
    echo -e "${GREEN}║     Ready for training! 🚀           ║${NC}"
    echo -e "${GREEN}║                                      ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"
    echo ""
    echo "You can now run:"
    echo "  python scripts/train.py --help"
    echo "  bash scripts/aws/run_all_training.sh"
    echo ""
    exit 0
else
    echo -e "${RED}╔══════════════════════════════════════╗${NC}"
    echo -e "${RED}║                                      ║${NC}"
    echo -e "${RED}║    ❌ VALIDATION FAILED ❌           ║${NC}"
    echo -e "${RED}║                                      ║${NC}"
    echo -e "${RED}║   $ERRORS error(s) found              ║${NC}"
    echo -e "${RED}║                                      ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════╝${NC}"
    echo ""
    echo "Please fix the errors above before training."
    echo ""
    exit 1
fi
