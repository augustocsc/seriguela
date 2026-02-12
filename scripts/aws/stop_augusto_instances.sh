#!/bin/bash
#
# Para APENAS instâncias AWS com prefixo "augusto-" que estão rodando
# Uso: ./stop_augusto_instances.sh
#       ./stop_augusto_instances.sh --confirm  (sem confirmação interativa)
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
AUTO_CONFIRM=false
if [[ "$1" == "--confirm" ]]; then
  AUTO_CONFIRM=true
fi

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Parar Instâncias AWS do Augusto${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Busca instâncias com prefixo "augusto-" que estão RUNNING
RUNNING_INSTANCES=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=augusto-*" "Name=instance-state-name,Values=running" \
  --query "Reservations[*].Instances[*].[InstanceId,Tags[?Key=='Name'].Value|[0],InstanceType]" \
  --output text)

if [ -z "$RUNNING_INSTANCES" ]; then
  echo -e "${GREEN}✓ Nenhuma instância com prefixo 'augusto-' está rodando.${NC}"
  echo -e "${GREEN}✓ Tudo OK! Sem custos em andamento.${NC}"
  echo ""
  exit 0
fi

# Mostra instâncias que serão paradas
echo -e "${YELLOW}As seguintes instâncias serão PARADAS:${NC}"
echo ""
echo -e "${GREEN}Instance ID${NC}\t\t${GREEN}Name${NC}\t\t\t\t${GREEN}Type${NC}"
echo "------------------------------------------------------------"

INSTANCE_IDS=()
while IFS=$'\t' read -r INSTANCE_ID NAME TYPE; do
  echo -e "$INSTANCE_ID\t$NAME\t$TYPE"
  INSTANCE_IDS+=("$INSTANCE_ID")
done <<< "$RUNNING_INSTANCES"

echo ""
echo -e "${BLUE}Total: ${#INSTANCE_IDS[@]} instância(s)${NC}"
echo ""

# Confirmação
if [ "$AUTO_CONFIRM" = false ]; then
  echo -e "${YELLOW}Tem certeza que deseja parar TODAS essas instâncias?${NC}"
  read -p "Digite 'SIM' (maiúsculo) para confirmar: " CONFIRM

  if [ "$CONFIRM" != "SIM" ]; then
    echo -e "${RED}Operação cancelada.${NC}"
    exit 0
  fi
fi

# Para as instâncias
echo ""
echo -e "${BLUE}Parando instâncias...${NC}"

aws ec2 stop-instances --instance-ids "${INSTANCE_IDS[@]}" > /dev/null

echo -e "${GREEN}✓ Comando de parar enviado para ${#INSTANCE_IDS[@]} instância(s)${NC}"
echo ""

# Aguarda confirmação
echo -e "${BLUE}Aguardando instâncias pararem...${NC}"

for INSTANCE_ID in "${INSTANCE_IDS[@]}"; do
  echo -n "  $INSTANCE_ID ... "
  aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" 2>/dev/null && echo -e "${GREEN}parada${NC}" || echo -e "${YELLOW}timeout${NC}"
done

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}✓ Todas as instâncias foram paradas!${NC}"
echo -e "${GREEN}✓ Custos pararam de acumular.${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Verificação final
echo -e "${BLUE}Verificação final:${NC}"
bash scripts/aws/list_augusto_instances.sh
