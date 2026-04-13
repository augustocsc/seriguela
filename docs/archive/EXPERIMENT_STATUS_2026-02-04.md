# Experiment Status - Model Scaling Evaluation
**Data**: 2026-02-04
**Horário**: 02:25 (hora local)
**Status**: ✅ **EXPERIMENTOS RODANDO EM 3 INSTÂNCIAS AWS**

---

## 🎯 Objetivo do Experimento

Avaliar o impacto do tamanho do modelo (Base 124M, Medium 355M, Large 774M) na qualidade e complexidade de expressões matemáticas geradas para regressão simbólica.

### Hipóteses Testadas

1. **H1**: Modelos maiores geram mais expressões válidas (>80%)
2. **H2**: Modelos maiores geram expressões mais complexas
3. **H3**: Modelos maiores têm melhor performance em benchmarks Nguyen
4. **H4**: Modelos maiores geram mais diversidade de expressões

---

## 🖥️ Infraestrutura AWS Ativa

### **Instância 1** - Avaliação de Qualidade Básica
- **IP**: 3.90.154.4 (seriguela-eval-basic)
- **Tipo**: g5.xlarge (NVIDIA A10G GPU, 24GB VRAM)
- **Tarefa**: Avaliar qualidade de geração de expressões
- **Status**:
  - ✅ Base (124M): 500 samples - **RODANDO** (38% completo às 02:19)
  - ✅ Medium (355M): 500 samples - **RODANDO**
  - ✅ Large (774M): 500 samples - **RODANDO**
- **Tempo estimado**: ~40 minutos (conclusão ~03:05)
- **Script corrigido**: ✅ Expression() constructor ao invés de parse_infix()

### **Instância 2** - Nguyen Benchmarks 1-6
- **IP**: 23.20.79.242 (seriguela-nguyen-1-6)
- **Tipo**: g5.xlarge (NVIDIA A10G GPU, 24GB VRAM)
- **Tarefa**: Avaliar em benchmarks Nguyen 1-6
- **Status**:
  - ✅ **18 experimentos**: 3 modelos × 6 benchmarks
  - Progresso: Benchmark 5/18 (28%) às 02:19
  - Velocidade: ~162s por benchmark
- **Tempo estimado**: ~50 minutos (conclusão ~03:10)

### **Instância 3** - Nguyen Benchmarks 7-12
- **IP**: 54.84.126.145 (seriguela-nguyen-7-12)
- **Tipo**: g5.xlarge (NVIDIA A10G GPU, 24GB VRAM)
- **Tarefa**: Avaliar em benchmarks Nguyen 7-12
- **Status**:
  - ✅ **18 experimentos**: 3 modelos × 6 benchmarks
  - Progresso: Benchmark 5/18 (28%) às 02:19
  - Velocidade: ~162s por benchmark
- **Tempo estimado**: ~50 minutos (conclusão ~03:10)

---

## 📊 Trabalho Total em Execução

### Avaliações Básicas (Instância 1)
- **Base**: 500 samples de qualidade
- **Medium**: 500 samples de qualidade
- **Large**: 500 samples de qualidade
- **Total**: 1500 gerações

### Benchmarks Nguyen (Instâncias 2 e 3)
- **Nguyen 1-6**: 3 modelos × 6 benchmarks × 200 samples = 3600 gerações
- **Nguyen 7-12**: 3 modelos × 6 benchmarks × 200 samples = 3600 gerações
- **Total**: 7200 gerações
- **Algoritmo**: Supervised generation (sem RL para acelerar)

### **TOTAL GERAL**: 8700 gerações de expressões!

---

## ⏱️ Cronograma

| Fase | Horário Início | Horário Esperado Conclusão | Status |
|------|---------------|---------------------------|--------|
| Lançamento instâncias | 01:45 | 01:50 | ✅ Completo |
| Upload modelos (155MB) | 01:50 | 02:10 | ✅ Completo |
| Correção de script | 02:10 | 02:15 | ✅ Completo |
| Avaliações básicas | 02:16 | 03:05 | 🔄 Rodando |
| Nguyen 1-6 | 02:05 | 03:10 | 🔄 Rodando |
| Nguyen 7-12 | 02:09 | 03:10 | 🔄 Rodando |
| Download resultados | 03:10 | 03:15 | ⏳ Pendente |
| Parar instâncias | 03:15 | 03:15 | ⏳ Pendente |
| Análise e relatório | 03:15 | 04:00 | ⏳ Pendente |

---

## 💰 Custo Estimado

- **3 instâncias** g5.xlarge
- **Tempo de execução**: ~1.5 horas cada
- **Custo**: 3 × 1.5h × $1.03/h = **~$4.50 USD total**

---

## 🔧 Problemas Encontrados e Resolvidos

### Problema 1: Script de avaliação com erro
- **Descrição**: `Expression.parse_infix()` não existe
- **Causa**: Método correto é usar o construtor `Expression(expr_str, is_prefix=False)`
- **Solução**: Script corrigido e re-uploaded para todas instâncias
- **Status**: ✅ Resolvido

### Problema 2: Dataset HuggingFace incompatível
- **Descrição**: Script original tentava carregar dataset específico
- **Solução**: Criado `evaluate_quality_simple.py` que gera prompts aleatórios
- **Status**: ✅ Resolvido

---

## 📁 Localização dos Resultados

### AWS (durante execução)
```
Instância 1: ~/seriguela/results/quality/
  - gpt2_base_700K_json_metrics.json
  - gpt2_base_700K_json_results.json
  - gpt2_medium_700K_json_metrics.json
  - gpt2_medium_700K_json_results.json
  - gpt2_large_700K_json_metrics.json
  - gpt2_large_700K_json_results.json

Instância 2: ~/seriguela/results/nguyen/
  - base_nguyen1_supervised.json ... base_nguyen6_supervised.json
  - medium_nguyen1_supervised.json ... medium_nguyen6_supervised.json
  - large_nguyen1_supervised.json ... large_nguyen6_supervised.json

Instância 3: ~/seriguela/results/nguyen/
  - base_nguyen7_supervised.json ... base_nguyen12_supervised.json
  - medium_nguyen7_supervised.json ... medium_nguyen12_supervised.json
  - large_nguyen7_supervised.json ... large_nguyen12_supervised.json
```

### Local (após download)
```
./results_experiment/
  ├── quality/
  │   ├── base_metrics.json
  │   ├── medium_metrics.json
  │   └── large_metrics.json
  └── nguyen/
      ├── base_nguyen1.json ... base_nguyen12.json
      ├── medium_nguyen1.json ... medium_nguyen12.json
      └── large_nguyen1.json ... large_nguyen12.json
```

---

## 🔍 Monitoramento

### Script de Monitoramento
```bash
bash monitor_all_experiments.sh
```

Mostra:
- Processos rodando em cada instância
- Progresso atual (%) de cada tarefa
- Últimas linhas dos logs
- Arquivos de resultado disponíveis

### SSH Manual
```bash
# Instância 1 (Quality)
ssh -i ~/chave-gpu.pem ubuntu@3.90.154.4
tail -f ~/eval_base_quality.log

# Instância 2 (Nguyen 1-6)
ssh -i ~/chave-gpu.pem ubuntu@23.20.79.242
tail -f ~/nguyen_1_6.log

# Instância 3 (Nguyen 7-12)
ssh -i ~/chave-gpu.pem ubuntu@54.84.126.145
tail -f ~/nguyen_7_12.log
```

---

## 📝 Próximos Passos (quando completar)

### 1. Download dos Resultados
```bash
# Criar diretório local
mkdir -p ./results_experiment/quality
mkdir -p ./results_experiment/nguyen

# Baixar da instância 1
scp -i ~/chave-gpu.pem ubuntu@3.90.154.4:~/seriguela/results/quality/*.json ./results_experiment/quality/

# Baixar da instância 2
scp -i ~/chave-gpu.pem ubuntu@23.20.79.242:~/seriguela/results/nguyen/*nguyen[1-6]*.json ./results_experiment/nguyen/

# Baixar da instância 3
scp -i ~/chave-gpu.pem ubuntu@54.84.126.145:~/seriguela/results/nguyen/*nguyen[7-9]*.json ./results_experiment/nguyen/
scp -i ~/chave-gpu.pem ubuntu@54.84.126.145:~/seriguela/results/nguyen/*nguyen1[0-2]*.json ./results_experiment/nguyen/
```

### 2. Parar Instâncias (CRÍTICO!)
```bash
aws ec2 stop-instances --instance-ids i-020af019c407e77da i-04c4eabae4a555af1 i-091e1500599aa6bd3
```

### 3. Análise e Agregação
```bash
python scripts/aggregate_experiment_results.py --quality_dir ./results_experiment/quality --nguyen_dir ./results_experiment/nguyen
```

### 4. Gerar Relatório Científico
- Comparar valid rates entre modelos
- Analisar diversidade de expressões
- Avaliar performance nos benchmarks Nguyen
- Testes estatísticos (t-test, ANOVA)
- Gerar visualizações (box plots, heatmaps)

---

## 🎓 Publicação

**Resultados esperados**:
- Paper com metodologia completa
- Model cards no HuggingFace
- Código e scripts no GitHub
- Apresentação em conferência

---

## 🔑 Credenciais e Configurações

- **AWS Key**: chave-gpu-nova (key-0779bf2fa7b802515)
- **HuggingFace Token**: ~/.tokens.txt
- **Wandb API Key**: ~/.tokens.txt
- **Security Group**: sg-0deaa73e23482e3f6
- **Region**: us-east-1

---

## ✅ Checklist de Finalização

- [ ] Aguardar conclusão de todos experimentos (~03:10)
- [ ] Verificar logs para erros
- [ ] Baixar todos resultados
- [ ] **PARAR INSTÂNCIAS AWS IMEDIATAMENTE**
- [ ] Verificar integridade dos arquivos JSON
- [ ] Agregar métricas em tabelas
- [ ] Gerar visualizações
- [ ] Escrever relatório científico
- [ ] Atualizar TRAINING_LOG com resultados finais
- [ ] Criar model cards
- [ ] (Opcional) Upload modelos para HuggingFace

---

**Última atualização**: 2026-02-04 02:25
**Próxima verificação**: 2026-02-04 03:10 (após conclusão estimada)
