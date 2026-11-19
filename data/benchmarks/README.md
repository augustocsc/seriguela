# data/benchmarks

| Item | Origem | No git? |
|---|---|---|
| `nguyen/` | CSVs gerados na v2 (referência; o núcleo v3 gera os dados por seed em `seriguela/data.py`) | sim |
| `strogatz/*.csv` + `.meta.json` | PMLB oficial (14 problemas) | sim |
| `feynman/*.meta.json` | PMLB oficial (101 problemas) | sim |
| `feynman/*.csv` | PMLB oficial, 100 mil linhas cada | **não**: `seriguela.data.load_problem` baixa sob demanda |
| `feynman_equations.py` | Equações-alvo dos 101 Feynman, nos nomes físicos | sim |
| `srbench_ground-truth_results.feather` | Resultados oficiais do SRBench (14 métodos, 130 problemas, 4 níveis de ruído) | **não** (79 MB): baixar do repositório `cavalab/srbench` |
| `download_pmlb_official.py` | Baixa os CSVs e metadados do PMLB | sim |

## Pendências do protocolo v3

- **Faltam os 19 `feynman_test_*`** do SRBench (os mais difíceis). Baixar metadados
  e equações antes do portão G4, para o painel externo ter os 116 Feynman.
- **4 problemas daqui não estão no SRBench** (`feynman_II_11_17`, `feynman_I_26_2`,
  `feynman_I_30_5`, `feynman_I_34_10`): ficam fora da comparação externa.
- Os domínios Nguyen de `seriguela/data.py` seguem a tabela usada pelo DSR
  (Petersen et al., 2021). Conferir contra o artigo antes de congelar o protocolo.
