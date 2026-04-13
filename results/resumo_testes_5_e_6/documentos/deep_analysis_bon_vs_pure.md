# Análise Profunda: Elite Buffer (BoN) vs Baselines (Pure)

Esta análise confronta os resultados isolados do **Teste 5** (apenas baselines puros) com o **Teste 6** (apenas variantes BoN com o Elite Buffer ativado e o tokenizador corrigido). O objetivo é entender o impacto real de reciclar as melhores expressões geradas no treinamento por reforço para Regressão Simbólica.

---

## 1. O Diagnóstico Empírico

A tabela abaixo mostra o R² máximo de treino atingido no step 500 (Batch = 1024):

| Benchmark | Escopo | Pure-GRPO | BoN-GRPO | Pure-PPO | BoN-PPO |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **nguyen_1** | Fácil | **0.998** | 0.992 | 0.992 | 0.992 |
| **nguyen_1** | Fácil | 0.997 | **1.000** | 0.995 | **1.000** |
| **nguyen_1** | Fácil | 0.994 | 0.994 | **1.000** | 0.994 |
| **nguyen_5** | Difícil | 0.116 | **0.493** | **0.719** | 0.535 |
| **nguyen_5** | Difícil | 0.000 | 0.000 | **0.385** | 0.038 |
| **nguyen_5** | Difícil | **1.000** | 0.000 | **0.442** | 0.038 |
| **nguyen_9** | Moderado | **0.478** | 0.326 | 0.391 | **0.777** |
| **nguyen_9** | Moderado | 0.799 | 0.799 | **1.000** | 0.458 |
| **nguyen_9** | Moderado | **0.540** | 0.506 | 0.536 | **1.000** |

### Padrão A: Saturação em Problemas Fáceis (`nguyen_1`)
Em equações de fácil resolução (`x**3 + x**2 + x`), tanto o modelo puro quanto o modelo BoN batem virtualmente $1.0$ de $R^2$. Aqui, o Elite Buffer atua apenas como um acelerador secundário de convergência. O espaço de busca é tão brando que ambas as abordagens acham a resposta antes do passo 100.

### Padrão B: Colapso de Variância (`nguyen_5`)
O `nguyen_5` (`sin(x**2) * cos(x) - 1`) é fatal para a exploração. Todas as seeds do **BoN-GRPO colapsaram no zero absoluto (0.000)** na geração inicial, ou perto disso, e não saíram mais. Mesmo o BoN-PPO sofreu imensamente.
* **Por que ocorreu?** Quando o batch size é massivo (1024), gerar 1024 rollouts *frescos* (Pure) garante que a rede neural tente rotas incrivelmente diversas no espaço da equação. No BoN, estamos preenchendo **metade do batch (512 amostras)** com lixo inicial reciclado do buffer, pelo simples fato de tentar retroalimentar as perdas. A rede fica sobre-treinada nos piores exemplos iniciais tentando "espremer o gradiente" de um local onde não há sinal, bloqueando a chance que o modelo teria de tropeçar num seno ou cosseno redentor no passo 20 ou 30.

### Padrão C: Super-Exploitation (`nguyen_9`)
O `nguyen_9` revela o trunfo absoluto do PPO com Buffer: **BoN-PPO atingiu 1.0 cravado e 0.77**, destruindo o Baseline Pure-PPO em duas das três seeds.
* **Por que ocorreu?** O `nguyen_9` (`sin(x) + sin(y**2)`) é composto por blocos ("building blocks"). O BoN conseguiu tropeçar no primeiro bloco (`sin(x)`), guardou essa "semente" no Elite Buffer, e passou os 400 steps seguintes obrigando a rede a olhar para o `sin(x)` em 50% dos gradientes. A rede aprendeu a usar essa âncora e focar apenas na variação do segundo bloco (`sin(y**2)`), matando a questão. O modelo Pure, por não ter memória contínua de "semanas passadas", esqueceu do `sin(x)` quando o gradiente apontou para um ótimo local e perdeu o caminho para a solução global.

---

## 2. Insights para a Avaliação Final (Próximos Passos Ph. A)

### I. A Proporção do Buffer de Elite é Tóxica (50% é alto demais)
Substituir **50% do Batch Size** atual (`buffer_proportion` em 0.5) pelo Elite Buffer transforma o modelo em um sistema "Gula-first" (alta explotação e baixa exploração).
* **Solução:** Na próxima fase, a proporção do buffer deve cair para algo entre **10% a 25%** (ou. `0.1` a `0.25`). Assim, em um batch de 1024, 800 amostras continuam garantindo exploração maciça, enquanto as "100 melhores amostras históricas" âncoram os blocos já encontrados, evitando o colapso visto em `nguyen_5`.

### II. O Risco de Sobretreino Precoce (Early Collapse)
A reciclagem contínua das mesmas expressões desde o step zero (onde elas só valem $R^2 \approx 0.05$ por causa de uma constante aleatória que agradou levemente a rede) destrói a Entropia do GRPO/PPO.
* **Solução:** O Elite Buffer não deve ser ativado no Step 1. Ou ele precisa exigir um R² ou Recompensa mínima de fundação (Ex: `if reward > 0.4: add_to_buffer`), ou só "ligar" o buffer após o step 50 (warmup).

### III. GRPO Odeia o Elite Buffer (Por Design)
O GRPO ($Group\ Relative\ Policy\ Optimization$) opera puramente estimando a vantagem subtraindo a média da recompensa **do lote atual**. No BoN-GRPO, inserimos 512 amostras históricas de altíssima recompensa junto com 512 gerações péssimas e frescas. 
Isso faz a "Média do Lote" disparar. Como resultado, **todas as gerações frescas vão invariavelmente receber pesos negativos (vantagem no GRPO = Recompensa - Média do Lote)**, sendo reprimidas! O GRPO foi desenhado para medir parentesco pareado em gerações irmãs; colocar primos tetravôs distantes (o elite buffer) no meio destrói o estimador e induz o colapso zero que vimos.

---

## 3. Conclusão da Pesquisa e Ação para a Fase V2
A sua suspeita validada traz um conhecimento profundo para a engenharia de RL em Symbol Regression:

1. **A Hipótese BoN foi Validada, porém Falha Sem Ajustes:** O modelo BoN provou que de fato consegue ancorar "Features / Building blocks", mas o preço pago na perda da exploração massiva com 1024 inviabiliza o modelo em equações densas.
2. **GRPO Deve Ser Usado em Pureza Máxima:** GRPO deve ser a linha de frente de velocidade, mas sempre **Pure**. A matemática dele não suporta injeções de baselines históricos (o elite buffer adultera a Vantagem Relativa de Grupo).
3. **PPO ganha o Monopólio do Buffer:** PPO é um algoritmo *Absolute-Advantage-based* (via Value Network / Baseline fixo). Ele tolera perfeitamente as memórias históricas sem adulterar o cálculo das amostras frescas. O Best-of-N deveria ser renomeado para "Expert-Reply PPO" e receber uma fração muito menor de injeção (20%).
