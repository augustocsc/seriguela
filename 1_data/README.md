# 1_data/ - Preparação de Dados

Este diretório contém todos os dados utilizados no projeto, organizados por estágio de processamento e tipo.

## Estrutura

```
1_data/
├── raw/                    # Dados originais sem processamento
├── processed/              # Dados processados e prontos para treino
└── benchmarks/             # Benchmarks para avaliação
    ├── nguyen/            # Nguyen benchmarks 1-12 (atual)
    ├── feynman/           # Feynman equations (futuro)
    └── strogatz/          # Strogatz benchmarks (futuro)
```

## Fontes de Dados

### Dados de Treinamento
- **Fonte**: HuggingFace Hub (`augustocsc/sintetico_natural`)
- **Tamanho**: 700K expressões matemáticas sintéticas
- **Formato**: JSON estruturado
- **Localização**: `processed/`

### Benchmarks Disponíveis

#### Nguyen Benchmarks (1-12)
Benchmarks padrão para symbolic regression:
- **Nguyen-1**: x³ + x² + x
- **Nguyen-2**: x⁴ + x³ + x² + x
- **Nguyen-3**: x⁵ + x⁴ + x³ + x² + x
- **Nguyen-4**: x⁶ + x⁵ + x⁴ + x³ + x² + x
- **Nguyen-5**: sin(x²)·cos(x) - 1
- **Nguyen-6**: sin(x) + sin(x + x²)
- **Nguyen-7**: log(x + 1) + log(x² + 1)
- **Nguyen-8**: √x
- **Nguyen-9**: sin(x) + sin(y²)
- **Nguyen-10**: 2·sin(x)·cos(y)
- **Nguyen-11**: x^y
- **Nguyen-12**: x⁴ - x³ + y²/2 - y

**Localização**: `benchmarks/nguyen/`

## Próximos Benchmarks (Planejados)

### Feynman Equations
Equações da física de Feynman - 120+ fórmulas
- Complexidade maior que Nguyen
- Multi-variáveis (até 10+)
- Constantes físicas

### Strogatz Benchmarks
Sistemas dinâmicos e equações diferenciais
- Osciladores
- Sistemas caóticos
- Modelos populacionais

## Uso

### Preparar Dados de Treinamento

```bash
# A partir do diretório raiz
cd 2_training/supervised
python train_with_json.py --dataset_path ../../1_data/processed/700K
```

### Adicionar Novo Benchmark

1. Criar diretório: `benchmarks/novo_benchmark/`
2. Adicionar arquivos CSV com formato:
   ```csv
   x,y
   1.0,2.5
   2.0,5.0
   ...
   ```
3. Adicionar metadata em `novo_benchmark/metadata.json`:
   ```json
   {
     "name": "Novo Benchmark",
     "formula": "expressão matemática",
     "variables": ["x", "y"],
     "description": "descrição"
   }
   ```

## Scripts Relacionados

- Processamento: `src/seriguela/data/`
- Avaliação em benchmarks: `3_evaluation/benchmarks/`

## Referências

- Nguyen et al. (2012): "Semantically-based crossover in genetic programming"
- Feynman Lectures on Physics
- Dataset original: https://huggingface.co/datasets/augustocsc/sintetico_natural
