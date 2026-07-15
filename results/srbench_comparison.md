# Nosso campeao (Base Infix, bon_ppo, C=1, 1 seed) vs SRBench leaderboard

Ruido 0. `solution`: SRBench=symbolic_solution (sympy); nosso=R²>0.999 (proxy).

_Atualizado em 2026-07-15 apos correcao do bug de expoente-C (ver
dissertation_final/HANDOFF.md §0 e classes/expression.py); numeros de
"solution %" nao mudaram (nenhuma solucao exata dependia do bug), "R²>0.99 %"
e a mediana ajustaram levemente. Regenerar com `python
4_analysis/srbench_comparison.py`._

## Feynman (ruido 0)

| metodo | solution % | R²>0.99 % | R² mediana |
|---|---|---|---|
| AIFeynman | 57.3 | 88.5 | 1.000 |
| AFP_FE | 27.0 | 73.3 | 1.000 |
| GP-GOMEA | 26.8 | 88.0 | 1.000 |
| ITEA | 22.4 | 40.5 | 0.979 |
| AFP | 21.1 | 56.8 | 0.995 |
| DSR | 19.7 | 35.8 | 0.954 |
| Operon | 16.6 | 94.0 | 1.000 |
| gplearn | 16.4 | 50.4 | 0.990 |
| **NOSSO (bon_ppo)** | 16.2 | 26.3 | 0.808 |
| SBP-GP | 12.7 | 94.1 | 1.000 |
| EPLEX | 12.4 | 78.0 | 0.999 |
| BSR | 2.5 | 22.2 | 0.769 |
| FEAT | 0.0 | 61.6 | 0.997 |
| FFX | 0.0 | 42.5 | 0.986 |
| MRGP | 0.0 | 97.7 | 1.000 |

## Strogatz (ruido 0)

| metodo | solution % | R²>0.99 % | R² mediana |
|---|---|---|---|
| GP-GOMEA | 29.5 | 88.4 | 1.000 |
| AIFeynman | 28.2 | 55.7 | 0.994 |
| AFP_FE | 20.0 | 57.1 | 0.996 |
| DSR | 19.6 | 16.1 | 0.866 |
| AFP | 15.2 | 41.1 | 0.983 |
| SBP-GP | 11.6 | 90.2 | 1.000 |
| Operon | 11.4 | 93.6 | 1.000 |
| EPLEX | 8.9 | 38.4 | 0.975 |
| gplearn | 8.9 | 17.0 | 0.889 |
| **NOSSO (bon_ppo)** | 7.1 | 7.1 | 0.852 |
| ITEA | 7.1 | 25.0 | 0.892 |
| FEAT | 0.9 | 66.1 | 0.999 |
| BSR | 0.9 | 47.3 | 0.976 |
| FFX | 0.0 | 44.6 | 0.986 |
| MRGP | 0.0 | 83.0 | 1.000 |
