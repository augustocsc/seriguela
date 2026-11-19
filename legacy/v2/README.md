# legacy/v2: o código das Fases 1 a 4 (até 2026-07)

Este é o código que produziu `results/v2/` e o texto atual de `dissertation/`.
Ele fica aqui como registro; **não é mantido e não roda a partir desta pasta**
(os caminhos internos apontam para a raiz antiga do repositório).

Para rodar o código antigo exatamente como era:

```bash
git checkout v2-final
```

## O que tem aqui

| Pasta | Conteúdo |
|---|---|
| `2_training/reinforcement/` | Treinadores PPO/GRPO/BoN da v2, recompensas, buffer |
| `2_training/supervised/` | Scripts do SFT (os 6 modelos estão no HuggingFace) |
| `3_evaluation/` | CLI de avaliação antiga |
| `4_analysis/` | Tabela-mãe, comparação SRBench, rescore pós-bug do expoente-C |
| `experiments/` | Filas YAML, daemon de fila, smoke test |
| `runpod/` | Orquestrador de pod (piloto e Fase 4) |
| `classes/expression.py` | Avaliador de expressões da v2 |
| `1_data/` | Geradores e downloaders antigos de benchmark |

## Por que foi substituído

Ver "Por que a v2 foi refeita" no `CLAUDE.md` e o plano em
`docs/reports/PLANO_V3_2026-11.md`. Em uma frase: a distribuição que amostrava não
era a que o update diferenciava, o otimizador rodava em FP16 com o estado do Adam
zerado, o baseline não era comparável aos braços RL, e a avaliação misturava
treino e teste.
