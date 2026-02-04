# Credentials Setup Guide

Este arquivo documenta os caminhos e configurações de credenciais necessárias para o projeto Seriguela.

## API Tokens

**Localização**: `C:\Users\madeinweb\.tokens.txt`

Formato do arquivo:
```
huggingface = hf_...
wandb = wandb_v1_...
```

### HuggingFace Token
- **Variável de ambiente**: `HF_TOKEN`
- **Uso**: Download de datasets e upload de modelos
- **Obter em**: https://huggingface.co/settings/tokens

### Weights & Biases API Key
- **Variável de ambiente**: `WANDB_API_KEY`
- **Uso**: Experiment tracking
- **Obter em**: https://wandb.ai/authorize

## SSH Keys (AWS)

**Localização**: `C:\Users\madeinweb\chave-gpu.pem`

### Configuração da chave:
```bash
# Linux/macOS
chmod 400 ~/chave-gpu.pem

# Uso
ssh -i ~/chave-gpu.pem ubuntu@<IP-DA-INSTANCIA>
```

**Importante**: Esta chave é usada para acessar todas as instâncias AWS do projeto.

## AWS CLI Configuration

As credenciais AWS são configuradas via `aws configure` e armazenadas em:
- **Windows**: `C:\Users\madeinweb\.aws\credentials`
- **Linux/macOS**: `~/.aws/credentials`

### Configuração do Security Group
- **Security Group ID**: `sg-0deaa73e23482e3f6`
- **IPs autorizados** (SSH port 22):
  - 143.106.58.120/32
  - 179.160.37.193/32

Para adicionar seu IP atual:
```bash
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
  --group-id sg-0deaa73e23482e3f6 \
  --protocol tcp --port 22 --cidr $MY_IP/32
```

## Scripts que usam credenciais

### Training Scripts
- `scripts/train_with_json.py` - Requer HF_TOKEN e WANDB_API_KEY
- `scripts/train.py` - Requer HF_TOKEN e WANDB_API_KEY
- `scripts/*_symbolic.py` (REINFORCE, GRPO, PPO) - Requer WANDB_API_KEY

### AWS Launch Scripts
- `scripts/aws/launch_medium_training.sh` - Requer --hf-token e --wandb-key
- `scripts/aws/launch_large_training.sh` - Requer --hf-token e --wandb-key
- Todos usam `chave-gpu.pem` para SSH

### Data Scripts
- `scripts/data/prepare_experiment_data.py` - Requer HF_TOKEN se baixando do Hub

## Carregando credenciais automaticamente

Os scripts Python podem carregar do arquivo `.tokens.txt`:

```python
import os

def load_credentials():
    """Load credentials from ~/.tokens.txt"""
    tokens_file = os.path.expanduser('~/.tokens.txt')
    if os.path.exists(tokens_file):
        with open(tokens_file) as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if key == 'huggingface':
                        os.environ['HF_TOKEN'] = value
                    elif key == 'wandb':
                        os.environ['WANDB_API_KEY'] = value
```

## Segurança

**NUNCA commitar credenciais!** Os seguintes padrões estão no `.gitignore`:
- `*.pem`
- `*.key`
- `.env`
- `aws/.env`
- `aws/keys/*`
- `.tokens.txt` (implícito por estar em home)

## Checklist de Setup

- [ ] Criar `~/.tokens.txt` com tokens HuggingFace e Wandb
- [ ] Baixar `chave-gpu.pem` e colocar em `~/`
- [ ] Configurar `chmod 400 ~/chave-gpu.pem` (Linux/macOS)
- [ ] Configurar AWS CLI com `aws configure`
- [ ] Adicionar IP ao security group se necessário
- [ ] Testar conexão: `ssh -i ~/chave-gpu.pem ubuntu@<IP-TESTE>`
- [ ] Testar AWS CLI: `aws ec2 describe-instances`

## Problemas Comuns

### SSH Permission Denied
- Verificar se está usando a chave correta: `chave-gpu.pem`
- Verificar permissões: `ls -la ~/chave-gpu.pem` (deve ser 400 ou 600)
- Verificar se IP está no security group

### HuggingFace Download Fails
- Verificar se `HF_TOKEN` está configurado
- Testar token: `huggingface-cli whoami`

### Wandb Login Fails
- Verificar se `WANDB_API_KEY` está configurado
- Testar: `wandb login`
- Atualizar wandb: `pip install --upgrade 'wandb>=0.24.1'`
