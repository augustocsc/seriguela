#!/bin/bash
#
# Para uma instância AWS específica do Augusto (com validação de prefixo)
# Uso: ./stop_instance.sh INSTANCE_ID
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

if [ $# -ne 1 ]; then
  echo -e "${RED}Erro: Instance ID não fornecido${NC}"
  echo "Uso: $0 INSTANCE_ID"
  echo ""
  echo "Para ver suas instâncias:"
  echo "  bash scripts/aws/list_augusto_instances.sh"
  exit 1
fi

INSTANCE_ID="$1"

echo -e "${BLUE}Parando instância: $INSTANCE_ID${NC}"
echo ""

# Validar que a instância existe e tem prefixo "augusto-"
INSTANCE_INFO=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query "Reservations[0].Instances[0].[State.Name,Tags[?Key=='Name'].Value|[0],InstanceType]" \
  --output text 2>/dev/null)

if [ -z "$INSTANCE_INFO" ]; then
  echo -e "${RED}Erro: Instância $INSTANCE_ID não encontrada${NC}"
  exit 1
fi

# Parse info
STATE=$(echo "$INSTANCE_INFO" | cut -f1)
NAME=$(echo "$INSTANCE_INFO" | cut -f2)
TYPE=$(echo "$INSTANCE_INFO" | cut -f3)

echo "Nome: $NAME"
echo "Tipo: $TYPE"
echo "Estado: $STATE"
echo ""

# Validar prefixo "augusto-"
if [[ ! "$NAME" =~ ^augusto- ]]; then
  echo -e "${RED}⚠️  AVISO: Esta instância NÃO tem prefixo 'augusto-'!${NC}"
  echo -e "${RED}⚠️  Pode pertencer a outra pessoa.${NC}"
  echo ""
  echo -e "${YELLOW}Tem certeza que deseja parar esta instância?${NC}"
  read -p "Digite 'CONFIRMO' (maiúsculo) para continuar: " CONFIRM

  if [ "$CONFIRM" != "CONFIRMO" ]; then
    echo -e "${GREEN}Operação cancelada. Instância segura!${NC}"
    exit 0
  fi
fi

# Verificar se já está parada
if [ "$STATE" == "stopped" ]; then
  echo -e "${GREEN}✓ Instância já está parada.${NC}"
  exit 0
fi

# Verificar se está em transição
if [ "$STATE" == "stopping" ]; then
  echo -e "${YELLOW}Instância já está em processo de parar...${NC}"
  echo "Aguardando..."
  aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID"
  echo -e "${GREEN}✓ Instância parada!${NC}"
  exit 0
fi

# Parar a instância
if [ "$STATE" == "running" ]; then
  echo -e "${BLUE}Enviando comando para parar...${NC}"
  aws ec2 stop-instances --instance-ids "$INSTANCE_ID" > /dev/null

  echo -e "${BLUE}Aguardando instância parar...${NC}"
  aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID"

  echo ""
  echo -e "${GREEN}======================================${NC}"
  echo -e "${GREEN}✓ Instância parada com sucesso!${NC}"
  echo -e "${GREEN}✓ Custos pararam de acumular.${NC}"
  echo -e "${GREEN}======================================${NC}"
else
  echo -e "${YELLOW}Instância está no estado: $STATE${NC}"
  echo -e "${YELLOW}Não foi possível parar.${NC}"
  exit 1
fi
