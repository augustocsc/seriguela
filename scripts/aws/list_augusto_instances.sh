#!/bin/bash
#
# Lista apenas instâncias AWS com prefixo "augusto-"
# Uso: ./list_augusto_instances.sh
#

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Instâncias AWS do Augusto${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Lista todas as instâncias com prefixo "augusto-"
INSTANCES=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=augusto-*" \
  --query "Reservations[*].Instances[*].[InstanceId,InstanceType,State.Name,PublicIpAddress,Tags[?Key=='Name'].Value|[0]]" \
  --output text)

if [ -z "$INSTANCES" ]; then
  echo -e "${YELLOW}Nenhuma instância com prefixo 'augusto-' encontrada.${NC}"
  echo ""
  exit 0
fi

# Formatação
echo -e "${GREEN}Instance ID${NC}\t\t${GREEN}Type${NC}\t\t${GREEN}State${NC}\t\t${GREEN}IP${NC}\t\t${GREEN}Name${NC}"
echo "------------------------------------------------------------"

# Conta instâncias por estado
RUNNING=0
STOPPED=0
OTHER=0

while IFS=$'\t' read -r INSTANCE_ID TYPE STATE IP NAME; do
  # Colorir baseado no estado
  case "$STATE" in
    running)
      STATE_COLOR="${GREEN}"
      RUNNING=$((RUNNING + 1))
      ;;
    stopped)
      STATE_COLOR="${YELLOW}"
      STOPPED=$((STOPPED + 1))
      ;;
    *)
      STATE_COLOR="${RED}"
      OTHER=$((OTHER + 1))
      ;;
  esac

  echo -e "$INSTANCE_ID\t$TYPE\t${STATE_COLOR}$STATE${NC}\t\t$IP\t$NAME"
done <<< "$INSTANCES"

echo ""
echo -e "${BLUE}Resumo:${NC}"
echo -e "  ${GREEN}Running:${NC} $RUNNING"
echo -e "  ${YELLOW}Stopped:${NC} $STOPPED"
if [ $OTHER -gt 0 ]; then
  echo -e "  ${RED}Other:${NC} $OTHER"
fi
echo ""

# Alerta se houver instâncias rodando
if [ $RUNNING -gt 0 ]; then
  echo -e "${YELLOW}⚠️  Você tem $RUNNING instância(s) rodando!${NC}"
  echo -e "${YELLOW}Para parar todas as suas instâncias, use:${NC}"
  echo -e "  ${BLUE}bash scripts/aws/stop_augusto_instances.sh${NC}"
  echo ""
fi
