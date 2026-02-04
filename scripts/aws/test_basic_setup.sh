#!/bin/bash
# Script de teste básico para verificar setup AWS e chaves
# Lança instância, roda teste simples, e para a instância

set -e

echo "=========================================="
echo "TESTE BÁSICO DE SETUP AWS"
echo "=========================================="
echo

# Carregar tokens
WANDB_KEY=$(cat ~/.tokens.txt | grep wandb | cut -d= -f2 | tr -d ' ')
HF_TOKEN=$(cat ~/.tokens.txt | grep huggingface | cut -d= -f2 | tr -d ' ')

if [ -z "$WANDB_KEY" ] || [ -z "$HF_TOKEN" ]; then
    echo "❌ ERRO: Tokens não encontrados em ~/.tokens.txt"
    exit 1
fi

echo "✅ Tokens carregados"
echo

# Criar user-data script simples
cat > /tmp/userdata_test.sh << 'EOF'
#!/bin/bash
set -x
exec > >(tee /home/ubuntu/test_setup.log)
exec 2>&1

echo "=== Iniciando teste básico ==="
cd /home/ubuntu

# Clone repo
git clone https://github.com/augustocsc/seriguela.git
cd seriguela
git checkout experiment/ppo-symbolic-regression

# Setup Python
python3 -m venv .venv
source .venv/bin/activate

# Install minimal dependencies
pip install -q transformers torch --index-url https://download.pytorch.org/whl/cu121
pip install -q peft datasets

# Teste simples: gerar 5 expressões com modelo base
cat > test_generate.py << 'PYTEST'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

prompt = '{"vars": ["x"], "ops": ["+", "*"], "cons": "C", "expr": "'
print(f"\nPrompt: {prompt}")

inputs = tokenizer(prompt, return_tensors="pt")

print("\nGenerating 5 expressions...")
for i in range(5):
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        expr = text.split('"expr": "')[1].split('"')[0]
        print(f"  {i+1}. {expr}")
    except:
        print(f"  {i+1}. [failed to parse]")

print("\n✅ Teste básico concluído com sucesso!")
echo "SETUP_TEST_SUCCESS" > /home/ubuntu/.test_complete
PYTEST

python test_generate.py

echo "=== Teste básico finalizado ==="
echo "COMPLETE" > /home/ubuntu/.setup_complete
EOF

# Substituir variáveis
sed -i "s/YOUR_WANDB_KEY/$WANDB_KEY/g" /tmp/userdata_test.sh
sed -i "s/YOUR_HF_TOKEN/$HF_TOKEN/g" /tmp/userdata_test.sh

# Lançar instância
echo "🚀 Lançando instância g5.xlarge para teste..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id ami-0e2c8caa4b6378d8c \
    --instance-type g5.xlarge \
    --key-name chave-gpu \
    --security-group-ids sg-0deaa73e23482e3f6 \
    --iam-instance-profile Name=ecsInstanceRole \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=50,VolumeType=gp3}' \
    --user-data file:///tmp/userdata_test.sh \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=seriguela-test-setup}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "✅ Instância criada: $INSTANCE_ID"
echo

# Aguardar instância estar running
echo "⏳ Aguardando instância ficar running..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID
echo "✅ Instância running"
echo

# Pegar IP público
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "📍 IP público: $PUBLIC_IP"
echo

# Aguardar SSH estar disponível
echo "⏳ Aguardando SSH ficar disponível..."
for i in {1..30}; do
    if ssh -i ~/chave-gpu.pem -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$PUBLIC_IP "echo ok" 2>/dev/null; then
        echo "✅ SSH disponível"
        break
    fi
    echo "  Tentativa $i/30..."
    sleep 10
done

echo

# Aguardar teste completar (max 10 minutos)
echo "⏳ Aguardando teste completar (max 10 minutos)..."
for i in {1..60}; do
    if ssh -i ~/chave-gpu.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP "test -f .test_complete" 2>/dev/null; then
        echo "✅ Teste completado!"
        break
    fi
    echo "  Aguardando... ($i/60)"
    sleep 10
done

echo

# Mostrar log
echo "=========================================="
echo "LOG DO TESTE:"
echo "=========================================="
ssh -i ~/chave-gpu.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP "tail -50 test_setup.log"
echo

# Parar instância
echo "🛑 Parando instância..."
aws ec2 stop-instances --instance-ids $INSTANCE_ID
echo "✅ Instância $INSTANCE_ID parada"
echo

echo "=========================================="
echo "✅ TESTE BÁSICO CONCLUÍDO!"
echo "=========================================="
echo
echo "Instância: $INSTANCE_ID (STOPPED)"
echo "IP usado: $PUBLIC_IP"
echo
echo "Para deletar a instância:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
