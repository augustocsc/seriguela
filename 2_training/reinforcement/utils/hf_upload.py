"""
HuggingFace Hub upload utilities.

Handles automatic upload of:
- Trained models (LoRA adapters)
- Results datasets
- Model cards
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class HuggingFaceUploader:
    """
    Upload trained models and results to HuggingFace Hub.

    Automatically:
    - Creates repositories if they don't exist
    - Generates model cards with training info
    - Uploads model files and results
    """

    def __init__(self, username: str = "augustocsc"):
        """
        Initialize HuggingFace uploader.

        Args:
            username: HuggingFace username for repository creation
        """
        self.username = username
        self._api = None

    @property
    def api(self):
        """Lazy initialization of HfApi."""
        if self._api is None:
            try:
                from huggingface_hub import HfApi
                self._api = HfApi()
            except ImportError:
                raise ImportError("huggingface_hub not installed. Run: pip install huggingface_hub")
        return self._api

    def upload_model(
        self,
        model_dir: Path,
        repo_name: str,
        private: bool = False,
        training_config: Optional[Dict] = None,
    ) -> str:
        """
        Upload trained model to HuggingFace Hub.

        Args:
            model_dir: Directory containing model files
            repo_name: Name for the repository
            private: Whether to make repository private
            training_config: Training configuration for model card

        Returns:
            URL of the uploaded model
        """
        from huggingface_hub import create_repo

        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise ValueError(f"Model directory does not exist: {model_dir}")

        repo_id = f"{self.username}/{repo_name}"

        # Create repository
        try:
            create_repo(repo_id, exist_ok=True, private=private)
            logger.info(f"Repository created/exists: {repo_id}")
        except Exception as e:
            logger.warning(f"Could not create repository: {e}")

        # Generate model card if config provided
        if training_config:
            model_card = self._generate_model_card(repo_name, training_config)
            readme_path = model_dir / "README.md"
            with open(readme_path, "w") as f:
                f.write(model_card)

        # Upload files
        try:
            self.api.upload_folder(
                folder_path=str(model_dir),
                repo_id=repo_id,
                repo_type="model",
            )
            logger.info(f"Model uploaded successfully to {repo_id}")
        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            raise

        return f"https://huggingface.co/{repo_id}"

    def upload_results(
        self,
        results_dir: Path,
        dataset_name: str,
        private: bool = False,
    ) -> str:
        """
        Upload results as a HuggingFace dataset.

        Args:
            results_dir: Directory containing result files
            dataset_name: Name for the dataset repository
            private: Whether to make repository private

        Returns:
            URL of the uploaded dataset
        """
        from huggingface_hub import create_repo

        results_dir = Path(results_dir)
        if not results_dir.exists():
            raise ValueError(f"Results directory does not exist: {results_dir}")

        repo_id = f"{self.username}/{dataset_name}"

        # Create repository
        try:
            create_repo(repo_id, exist_ok=True, private=private, repo_type="dataset")
            logger.info(f"Dataset repository created/exists: {repo_id}")
        except Exception as e:
            logger.warning(f"Could not create repository: {e}")

        # Upload files
        try:
            self.api.upload_folder(
                folder_path=str(results_dir),
                repo_id=repo_id,
                repo_type="dataset",
            )
            logger.info(f"Results uploaded successfully to {repo_id}")
        except Exception as e:
            logger.error(f"Failed to upload results: {e}")
            raise

        return f"https://huggingface.co/datasets/{repo_id}"

    def _generate_model_card(self, repo_name: str, config: Dict) -> str:
        """Generate a model card README.md."""
        return f"""---
license: mit
language: en
tags:
- symbolic-regression
- gpt2
- lora
- seriguela
datasets:
- augustocsc/sintetico_natural_prefix_682k
---

# {repo_name}

This model was trained using the Seriguela framework for symbolic regression.

## Training Configuration

- **Algorithm**: {config.get('algorithm', 'Unknown')}
- **Base Model**: {config.get('base_model', 'gpt2')}
- **Reward Function**: {config.get('reward_fn', 'Unknown')}
- **Penalty Strategy**: {config.get('penalty_strategy', 'Unknown')}
- **Temperature Schedule**: {config.get('temp_scheduler', 'Unknown')}
- **Learning Rate**: {config.get('learning_rate', 'Unknown')}
- **Batch Size**: {config.get('batch_size', 'Unknown')}
- **Max Steps**: {config.get('max_steps', 'Unknown')}

## Results

- **Best R²**: {config.get('best_r2', 'Unknown')}
- **Best Expression**: `{config.get('best_expression', 'Unknown')}`

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load
tokenizer = AutoTokenizer.from_pretrained("{self.username}/{repo_name}")
base_model = AutoModelForCausalLM.from_pretrained("{config.get('base_model', 'gpt2')}")
model = PeftModel.from_pretrained(base_model, "{self.username}/{repo_name}")
model.eval()

# Generate
prompt = '{{"vars": ["x_1"], "ops": ["+", "*", "sin"], "cons": "C", "expr": "'
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
print(tokenizer.decode(outputs[0]))
```

## Citation

```bibtex
@misc{{seriguela2026,
  title={{Seriguela: Symbolic Regression with LLMs}},
  author={{Augusto Cesar}},
  year={{2026}},
  note={{RL-optimized GPT-2 for symbolic regression}}
}}
```

---
Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Seriguela
"""
