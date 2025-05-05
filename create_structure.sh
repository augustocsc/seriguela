#!/bin/bash

echo "Criando estrutura de pastas para o projeto de fine-tuning..."

# Diretórios Principais
mkdir -p data/raw
mkdir -p data/processed
mkdir -p scripts
mkdir -p configs
mkdir -p output
mkdir -p notebooks

echo "Diretórios criados."

# Arquivos Placeholder e de Configuração Inicial
touch data/raw/.gitkeep                 # Mantém a pasta no Git mesmo vazia
touch data/processed/.gitkeep           # Mantém a pasta no Git mesmo vazia

echo "# Script para pré-processar dados (raw -> processed)" > scripts/preprocess_data.py
echo "# Script principal de treinamento (usa Trainer)" > scripts/train.py
echo "# Script para avaliação customizada" > scripts/evaluate.py
echo "# Script para geração de texto com modelo treinado" > scripts/generate.py

echo "{}" > configs/training_args.json   # Placeholder para argumentos do Trainer
echo "{}" > configs/peft_config.json     # Placeholder para config PEFT (se usar)
echo "{}" > configs/model_config.json    # Placeholder para config do modelo base

touch notebooks/01_data_exploration.ipynb
touch notebooks/.gitkeep                # Mantém a pasta no Git mesmo vazia

touch requirements.txt

echo "Arquivos placeholder criados."

# Conteúdo Inicial para .gitignore
echo "Gerando .gitignore..."
cat << EOF > .gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
# Usually these files are written by a python script from a template
# before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Environments
.env
.venv
venv/
ENV/
env/
env.bak/
venv.bak/

# IDEs / Editors
.idea/
.vscode/
*.suo
*.ntvs*
*.njsproj
*.sln
*.sw?

# Jupyter Notebook
.ipynb_checkpoints

# Output folder (geralmente grande demais para Git)
output/*
!output/.gitkeep # Não ignore um .gitkeep se precisar manter a pasta

# Dados (podem ser grandes, usar Git LFS ou armazenar fora se necessário)
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
EOF

# Conteúdo Inicial para README.md (será preenchido com o texto gerado abaixo)
echo "Gerando README.md inicial..."
echo "# Nome do Seu Projeto de Fine-Tuning" > README.md
echo "" >> README.md
echo "(Breve descrição do objetivo do projeto)" >> README.md
echo "" >> README.md
echo "## Estrutura de Pastas" >> README.md
echo "" >> README.md
echo "**(COPIE E COLE A EXPLICAÇÃO DA ESTRUTURA GERADA NA PRÓXIMA SEÇÃO AQUI)**" >> README.md
echo "" >> README.md
echo "## Como Usar" >> README.md
echo "" >> README.md
echo "1.  **Setup:** Crie um ambiente virtual e instale as dependências:" >> README.md
echo "    \`\`\`bash" >> README.md
echo "    python -m venv venv" >> README.md
echo "    source venv/bin/activate  # Linux/macOS" >> README.md
echo "    # venv\\Scripts\\activate  # Windows" >> README.md
echo "    pip install -r requirements.txt" >> README.md
echo "    \`\`\`" >> README.md
echo "2.  **Dados:** Coloque seus dados brutos em \`data/raw/\` e execute (ou crie) o script \`scripts/preprocess_data.py\` para gerar os arquivos em \`data/processed/\`." >> README.md
echo "3.  **Configuração:** Ajuste os arquivos em \`configs/\` (argumentos de treino, modelo base, PEFT se aplicável)." >> README.md
echo "4.  **Treinamento:** Execute o script principal:" >> README.md
echo "    \`\`\`bash" >> README.md
echo "    python scripts/train.py --args_config configs/training_args.json --model_config configs/model_config.json" >> README.md
echo "    \`\`\`" >> README.md
echo "    *(Adapte os argumentos conforme necessário)*" >> README.md
echo "" >> README.md
echo "## Dependências" >> README.md
echo "" >> README.md
echo "As dependências Python estão listadas no arquivo \`requirements.txt\`." >> README.md

chmod +x create_structure.sh

echo "--------------------------------------------------"
echo "Estrutura criada com sucesso!"
echo "Para usar:"
echo "1. Torne o script executável: chmod +x create_structure.sh"
echo "2. Execute o script: ./create_structure.sh"
echo "3. Copie a explicação da estrutura (gerada na resposta anterior) para dentro do README.md onde indicado."
echo "--------------------------------------------------"