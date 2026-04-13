# Treinar GPT-2 Medium (355M) na AWS

Guia rápido para treinar o modelo maior e comparar com o base (124M).

## 1. Lançar Instância AWS

### Windows:
```bash
launch_medium_aws.bat
```

### Linux/Mac:
```bash
bash scripts/aws/launch_medium_training.sh \
  --wandb-key YOUR_WANDB_KEY \
  --hf-token YOUR_HF_TOKEN
```

Isso vai:
- Criar instância g5.xlarge (NVIDIA A10G GPU, 24GB VRAM)
- Instalar dependências automaticamente
- Começar treinamento do GPT-2 Medium imediatamente
- Treinar por 3 épocas (~2-3 horas)

## 2. Monitorar Treinamento

```bash
# Ver logs em tempo real
bash scripts/aws/monitor_medium_training.sh

# Ou conectar manualmente
ssh -i ~/.ssh/KEY.pem ubuntu@IP
tail -f /home/ubuntu/training_medium.log
```

## 3. Quando Completar

### Baixar Modelo:
```bash
scp -i ~/.ssh/KEY.pem -r ubuntu@IP:~/seriguela/output/gpt2_medium_700K_json ./
```

### Comparar com Base:
```bash
python scripts/compare_trained_models.py \
  --model_base augustocsc/Se124M_700K_infix_v3_json \
  --model_medium ./gpt2_medium_700K_json \
  --epochs 10
```

Isso vai mostrar:
- % de expressões válidas (base vs medium)
- % de uso de potências (x²)
- % de trig aninhadas (sin(x²))
- Profundidade média de composição
- Melhor R² no Nguyen-5

## 4. Parar Instância

```bash
# Pegar Instance ID
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=seriguela-medium-training" \
  --query "Reservations[*].Instances[*].[InstanceId,State.Name]" \
  --output table

# Parar instância
aws ec2 stop-instances --instance-ids i-xxxxx

# Ou terminar (deletar)
aws ec2 terminate-instances --instance-ids i-xxxxx
```

## Custos Estimados

- **g5.xlarge**: ~$1.00/hora
- **3 épocas**: ~2-3 horas
- **Total**: ~$2-3 USD

Não esqueça de parar a instância quando terminar!

## Troubleshooting

### Instância não inicia:
```bash
aws ec2 describe-instances --instance-ids i-xxxxx
```

### Training falha:
```bash
ssh ubuntu@IP
cat /var/log/user-data.log
```

### GPU não detectada:
```bash
ssh ubuntu@IP
nvidia-smi
```

## Próximos Passos

Depois de comparar base vs medium, você pode:

1. **Testar GPT-2 Large (774M)** - Mudar `--model_size gpt2-large` (requer g5.2xlarge)
2. **Treinar mais épocas** - Aumentar `--num_train_epochs`
3. **Fazer fine-tuning com RL** - Usar o modelo medium como base para REINFORCE/PPO
