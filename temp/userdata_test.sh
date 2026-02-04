#!/bin/bash
set -x
exec > >(tee /home/ubuntu/test_setup.log)
exec 2>&1

echo "=== Teste básico de setup ==="
cd /home/ubuntu

# Teste GPU
nvidia-smi

# Teste Python
python3 --version

# Teste pip
pip3 --version

echo "✅ Setup básico OK"
echo "COMPLETE" > /home/ubuntu/.setup_complete
