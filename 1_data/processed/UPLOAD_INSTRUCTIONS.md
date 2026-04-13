# Instruções: Upload Dataset Prefix para HuggingFace

## 📦 Preparação Completa

### ✅ Arquivos Criados

1. **Dataset Convertido**: `./1_data/processed/700K_prefix_682k/`
   - Split `train`: 682,429 exemplos
   - Split `validation`: 75,826 exemplos
   - Total: 758,255 exemplos convertidos

2. **Documentação**:
   - `DATASET_PREFIX_682K_README.md` - README completo para HuggingFace
   - `DATASET_COMPARISON.md` - Comparação entre datasets
   - `UPLOAD_INSTRUCTIONS.md` - Este arquivo

## 🚀 Upload para HuggingFace

### Opção 1: Via Script Automático (Recomendado)

```bash
cd C:/Users/madeinweb/seriguela

# Fazer upload com o script
python scripts/data/convert_infix_to_prefix.py \
  --use_training_split \
  --output_path ./1_data/processed/700K_prefix_682k \
  --upload \
  --repo_id augustocsc/sintetico_natural_prefix_682k
```

**Este comando**:
- ✅ Reutiliza o dataset já convertido (não reconverte)
- ✅ Faz upload para HuggingFace
- ✅ Cria repositório se não existir
- ❌ **Não inclui** README.md automaticamente (precisa adicionar manualmente)

### Opção 2: Via Python API

```python
from datasets import load_from_disk

# Carregar dataset convertido
dataset = load_from_disk('./1_data/processed/700K_prefix_682k')

# Fazer upload
dataset.push_to_hub(
    repo_id='augustocsc/sintetico_natural_prefix_682k',
    private=False  # Tornar público
)

print("✅ Upload completo!")
print("📍 URL: https://huggingface.co/datasets/augustocsc/sintetico_natural_prefix_682k")
```

### Opção 3: Via CLI (Manual)

```bash
# 1. Login
huggingface-cli login

# 2. Criar repositório
huggingface-cli repo create sintetico_natural_prefix_682k \
  --type dataset \
  --organization augustocsc

# 3. Upload dataset
cd 1_data/processed/700K_prefix_682k
git init
git remote add origin https://huggingface.co/datasets/augustocsc/sintetico_natural_prefix_682k
git add .
git commit -m "Initial commit: 682K prefix dataset"
git push -u origin main
```

## 📝 Adicionar README ao HuggingFace

**IMPORTANTE**: O README precisa ser adicionado manualmente após o upload.

### Passo 1: Upload Dataset (via script)

```bash
python scripts/data/convert_infix_to_prefix.py \
  --use_training_split \
  --output_path ./1_data/processed/700K_prefix_682k \
  --upload \
  --repo_id augustocsc/sintetico_natural_prefix_682k
```

### Passo 2: Adicionar README

**Via Web Interface** (Mais fácil):

1. Ir para https://huggingface.co/datasets/augustocsc/sintetico_natural_prefix_682k
2. Clicar em "Files and versions"
3. Clicar em "Add file" → "Create a new file"
4. Nome do arquivo: `README.md`
5. Copiar conteúdo de `1_data/processed/DATASET_PREFIX_682K_README.md`
6. Colar no editor
7. Commit changes

**Via Git** (Para quem prefere linha de comando):

```bash
# Clonar repositório
git clone https://huggingface.co/datasets/augustocsc/sintetico_natural_prefix_682k
cd sintetico_natural_prefix_682k

# Copiar README
cp ../seriguela/1_data/processed/DATASET_PREFIX_682K_README.md ./README.md

# Commit e push
git add README.md
git commit -m "Add comprehensive README"
git push
```

## 🔍 Verificação Pós-Upload

### 1. Verificar Dataset Carregável

```python
from datasets import load_dataset

# Testar carregamento
dataset = load_dataset('augustocsc/sintetico_natural_prefix_682k')

# Verificações
assert 'train' in dataset
assert 'validation' in dataset
assert len(dataset['train']) == 682429
assert len(dataset['validation']) == 75826

print("✅ Dataset carregado corretamente!")
```

### 2. Verificar Conversão

```python
# Pegar exemplo
example = dataset['train'][0]

# Verificar colunas
assert 'i_prompt_n' in example  # Infix original
assert 'p_prompt_n_converted' in example  # Prefix convertido
assert 'conversion_success' in example  # Status

print("✅ Colunas corretas!")
print(f"\nINFIX: {example['i_prompt_n'][:100]}...")
print(f"PREFIX: {example['p_prompt_n_converted'][:100]}...")
```

### 3. Verificar README

1. Ir para https://huggingface.co/datasets/augustocsc/sintetico_natural_prefix_682k
2. Verificar se README está visível
3. Verificar se formatação markdown está correta
4. Verificar se exemplos de código funcionam

## 📢 Anunciar Dataset

### Atualizar Dataset Original (Opcional)

Adicionar aviso no README de `augustocsc/sintetico_natural`:

```markdown
## ⚠️ Aviso Importante sobre Splits

Este dataset tem uma configuração que pode causar confusão:

- **Split padrão**: 12,221 exemplos (apenas `test`)
- **Com `data_dir='700K'`**: 947,876 exemplos (train/val/test)

**Para treinar modelos comparáveis**, use o dataset pré-processado:
👉 [sintetico_natural_prefix_682k](https://huggingface.co/datasets/augustocsc/sintetico_natural_prefix_682k)

Este dataset usa o **mesmo split (90/10, seed=42)** dos modelos publicados.
```

### Atualizar Model Cards

Para cada modelo publicado (ex: `Se124M_700K_infix_v3_json`), adicionar:

```markdown
## Dataset

Este modelo foi treinado com **682,429 exemplos** do dataset:
- [augustocsc/sintetico_natural](https://huggingface.co/datasets/augustocsc/sintetico_natural) (data_dir='700K', split='train')
- Split: 90% treino / 10% validação (seed=42)

Para treinar modelos comparáveis ou em notação prefix, use:
- [augustocsc/sintetico_natural_prefix_682k](https://huggingface.co/datasets/augustocsc/sintetico_natural_prefix_682k)
```

### Criar Post no HuggingFace

```markdown
# Novo Dataset: Sintetico Natural - Prefix Notation (682K)

Criamos um dataset que resolve problemas de reprodutibilidade no dataset original:

🎯 **Características**:
- ✅ 682K exemplos (mesmo split usado no treinamento dos modelos)
- ✅ Notação prefix para comparar com infix
- ✅ Split fixo (seed=42) - reprodutibilidade perfeita
- ✅ 100% de conversões bem-sucedidas

📊 **Splits**:
- Train: 682,429 exemplos
- Validation: 75,826 exemplos

🔗 **Links**:
- Dataset: https://huggingface.co/datasets/augustocsc/sintetico_natural_prefix_682k
- Comparação: [Link para DATASET_COMPARISON.md]

Use este dataset para treinar modelos prefix comparáveis aos modelos infix existentes!
```

## ✅ Checklist Final

### Antes do Upload
- [x] Dataset convertido (682,429 train + 75,826 val)
- [x] Taxa de conversão 100%
- [x] README preparado
- [x] Documentação de comparação criada
- [x] Instruções de upload documentadas

### Durante o Upload
- [ ] Login no HuggingFace: `huggingface-cli login`
- [ ] Upload via script ou API
- [ ] Verificar que repositório foi criado
- [ ] Adicionar README.md ao repositório

### Após o Upload
- [ ] Testar carregamento: `load_dataset('augustocsc/sintetico_natural_prefix_682k')`
- [ ] Verificar número de exemplos (682,429 + 75,826)
- [ ] Verificar que README está visível
- [ ] Testar exemplos de código do README
- [ ] (Opcional) Atualizar README do dataset original
- [ ] (Opcional) Atualizar model cards
- [ ] (Opcional) Criar post de anúncio

## 🚨 Troubleshooting

### Erro: "Repository not found"
```bash
# Criar repositório primeiro
huggingface-cli repo create sintetico_natural_prefix_682k \
  --type dataset \
  --organization augustocsc
```

### Erro: "Authentication required"
```bash
# Fazer login
huggingface-cli login
# Cole seu token quando solicitado
```

### Erro: "Dataset too large"
- Dataset tem ~300MB (arrow format comprimido)
- Não deve ter problemas
- Se houver, considerar usar Git LFS (já habilitado por padrão)

### README não aparece
1. Verificar que arquivo se chama exatamente `README.md` (case-sensitive)
2. Verificar que está na raiz do repositório
3. Limpar cache do navegador
4. Aguardar alguns minutos (cache do HuggingFace)

## 📞 Suporte

Se encontrar problemas:
1. Verificar logs de erro
2. Consultar documentação: https://huggingface.co/docs/datasets
3. Abrir issue no GitHub do projeto

---

**Criado**: 2026-02-09
**Versão**: 1.0
