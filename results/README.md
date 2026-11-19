# results/

| Pasta | O que é | Pode ser citado na defesa? |
|---|---|---|
| `v2/` | Resultados das Fases 1 a 4 (até 2026-07), gerados por `legacy/v2/` | **Não.** Servem de histórico e de material para o capítulo de ameaças à validade |
| `v3/` | Resultados do núcleo novo (`seriguela/`), esquema v3 | Sim, depois dos portões G1–G4 |

## v3: um arquivo por run

```
results/v3/<problema>/<braço>/seed_<s>.json
```

Cada arquivo passa por `seriguela.trainer.validate` antes de ser gravado e traz:
proveniência (commit, versões, hash da configuração e dos dados), contadores de
orçamento, a melhor expressão escolhida **pelo treino** com seu R² de teste e o
veredito simbólico, a política final, os 50 melhores candidatos e a curva por passo.

Um run interrompido por tripwire grava `seed_<s>.failed.json` com o motivo, e nunca
um resultado.

## v2: o que falta trazer para cá

`phase_4/` (1,9 GB, 356 arquivos) nunca entrou no git: está só no checkout principal,
em `results/phase_4/`. Antes de qualquer limpeza de disco, subir para um dataset do
HuggingFace ou copiar para `results/v2/phase_4/`.
