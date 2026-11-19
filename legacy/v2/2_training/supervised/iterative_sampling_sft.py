#!/usr/bin/env python3
"""
Iterative Sampling + SFT for Symbolic Regression

This approach:
1. Generate N expressions using the current model
2. Evaluate R^2 for each expression
3. Filter expressions with R^2 > threshold
4. Fine-tune the model on the best expressions
5. Repeat

This is a form of "Expert Iteration" or "Self-Play" adapted for symbolic regression.
"""

import os
import sys
import json
import argparse
import logging
import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classes"))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import PeftModel, LoraConfig, get_peft_model

from expression import Expression
from dataset import RegressionDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class IterativeSamplingSFT:
    """Iterative Sampling with Supervised Fine-Tuning."""

    def __init__(
        self,
        model_path: str,
        X: np.ndarray,
        y: np.ndarray,
        output_dir: str = "./output/iterative_sft",
        device: str = None,
    ):
        self.X = X
        self.y = y
        self.n_vars = X.shape[1]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model
        self._load_model(model_path)

        # Build prompt template
        self.prompt = self._build_prompt()

        # Track results
        self.best_r2 = -np.inf
        self.best_expression = None
        self.history = []

    def _load_model(self, model_path: str):
        """Load model and tokenizer."""
        logger.info(f"Loading model from {model_path}")

        if Path(model_path).exists():
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained("gpt2")
            if len(self.tokenizer) != base_model.config.vocab_size:
                base_model.resize_token_embeddings(len(self.tokenizer))

            try:
                model_with_lora = PeftModel.from_pretrained(base_model, model_path)
                self.model = model_with_lora.merge_and_unload()
                logger.info("LoRA adapter loaded and merged")
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(model_path)

        self.model = self.model.to(self.device)
        logger.info("Model loaded")

    def _build_prompt(self) -> str:
        """Build JSON format prompt."""
        vars_list = [f"x_{i+1}" for i in range(self.n_vars)]
        ops_list = ["+", "-", "*", "sin", "cos"]

        prompt = json.dumps({
            "vars": vars_list,
            "ops": ops_list,
            "cons": None,
            "expr": ""
        })[:-3]

        return prompt

    def extract_expression(self, text: str) -> str:
        """Extract expression from generated text."""
        try:
            if '"expr": "' in text:
                start = text.index('"expr": "') + len('"expr": "')
                remaining = text[start:]
                if '"}' in remaining:
                    return remaining[:remaining.index('"}')].strip()
                if '"' in remaining:
                    return remaining[:remaining.index('"')].strip()

            if '"expr": ' in text:
                start = text.index('"expr": ') + len('"expr": ')
                remaining = text[start:]
                if '"}' in remaining:
                    return remaining[:remaining.index('"}')].strip()

        except (ValueError, IndexError):
            pass

        return text.split('"expr"')[-1].strip(' ":}')

    def compute_r2(self, expression_str: str) -> float:
        """Compute R^2 score."""
        if not expression_str or expression_str.isspace():
            return -np.inf

        if 'C' in expression_str:
            expression_str = expression_str.replace('C', '1')

        try:
            expr = Expression(expression_str, is_prefix=False)
            if not expr.is_valid_on_dataset(self.X):
                return -np.inf

            y_pred = expr.evaluate(self.X)
            if not np.all(np.isfinite(y_pred)):
                return -np.inf

            ss_res = np.sum((self.y - y_pred) ** 2)
            ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)

            if ss_tot == 0:
                return 0.0

            return 1 - (ss_res / ss_tot)
        except Exception:
            return -np.inf

    def sample_expressions(self, n_samples: int, temperature: float = 0.7) -> List[Tuple[str, str, float]]:
        """Generate N expressions and evaluate them."""
        self.model.eval()

        inputs = self.tokenizer(self.prompt, return_tensors="pt").to(self.device)
        results = []

        for _ in tqdm(range(n_samples), desc="Sampling"):
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            expr_str = self.extract_expression(text)
            r2 = self.compute_r2(expr_str)

            if np.isfinite(r2):
                results.append((text, expr_str, r2))

                if r2 > self.best_r2:
                    self.best_r2 = r2
                    self.best_expression = expr_str

        return results

    def filter_best(self, results: List[Tuple[str, str, float]], threshold: float = 0.5) -> List[str]:
        """Filter expressions with R^2 above threshold."""
        best = [(text, expr, r2) for text, expr, r2 in results if r2 > threshold]
        best.sort(key=lambda x: x[2], reverse=True)

        # Return full texts for fine-tuning
        return [text for text, expr, r2 in best]

    def fine_tune(self, good_texts: List[str], epochs: int = 1):
        """Fine-tune on good expressions."""
        if not good_texts:
            logger.warning("No good expressions to fine-tune on")
            return

        logger.info(f"Fine-tuning on {len(good_texts)} good expressions")

        # Create dataset
        dataset = Dataset.from_dict({"text": good_texts})

        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=128,
                padding="max_length",
            )

        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

        # Add LoRA for fine-tuning
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["c_attn"],
            lora_dropout=0.05,
            bias="none",
        )

        self.model = get_peft_model(self.model, lora_config)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=min(4, len(good_texts)),
            learning_rate=5e-5,
            logging_steps=10,
            save_strategy="no",
            report_to=[],
            use_cpu=self.device.type == "cpu",
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator,
        )

        trainer.train()

        # Merge LoRA back
        self.model = self.model.merge_and_unload()
        logger.info("Fine-tuning complete")

    def run(
        self,
        n_iterations: int = 5,
        samples_per_iteration: int = 100,
        r2_threshold: float = 0.5,
        target_r2: float = 0.99,
    ):
        """Run iterative sampling + SFT."""
        logger.info("=" * 60)
        logger.info("ITERATIVE SAMPLING + SFT")
        logger.info("=" * 60)
        logger.info(f"Iterations: {n_iterations}")
        logger.info(f"Samples per iteration: {samples_per_iteration}")
        logger.info(f"R^2 threshold: {r2_threshold}")
        logger.info("=" * 60)

        for iteration in range(n_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration + 1}/{n_iterations}")
            logger.info(f"{'='*60}")

            # Step 1: Sample expressions
            results = self.sample_expressions(samples_per_iteration)

            # Step 2: Analyze results
            if results:
                r2_scores = [r2 for _, _, r2 in results]
                logger.info(f"Valid expressions: {len(results)}/{samples_per_iteration}")
                logger.info(f"Mean R^2: {np.mean(r2_scores):.4f}")
                logger.info(f"Max R^2: {np.max(r2_scores):.4f}")
                logger.info(f"Best overall: {self.best_r2:.4f} - {self.best_expression}")

                self.history.append({
                    "iteration": iteration + 1,
                    "valid_count": len(results),
                    "mean_r2": float(np.mean(r2_scores)),
                    "max_r2": float(np.max(r2_scores)),
                    "best_overall_r2": self.best_r2,
                })

                # Early stop if we found perfect match
                if self.best_r2 >= target_r2:
                    logger.info(f"Target R^2 {target_r2} reached!")
                    break

                # Step 3: Filter best and fine-tune
                good_texts = self.filter_best(results, threshold=r2_threshold)
                if good_texts:
                    logger.info(f"Fine-tuning on {len(good_texts)} expressions with R^2 > {r2_threshold}")
                    self.fine_tune(good_texts, epochs=1)

                    # Increase threshold for next iteration
                    r2_threshold = min(r2_threshold + 0.1, 0.9)
            else:
                logger.warning("No valid expressions generated")

        # Final results
        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)
        logger.info(f"Best R^2: {self.best_r2:.4f}")
        logger.info(f"Best expression: {self.best_expression}")

        return {
            "best_r2": self.best_r2,
            "best_expression": self.best_expression,
            "history": self.history,
        }


def main():
    parser = argparse.ArgumentParser(description="Iterative Sampling + SFT")
    parser.add_argument("--model_path", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, default="./data/ppo_test/sin_x1.csv")
    parser.add_argument("--output_dir", type=str, default="./output/iterative_sft")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    reg = RegressionDataset(str(dataset_path.parent), dataset_path.name)
    X, y = reg.get_numpy()

    # Run experiment
    experiment = IterativeSamplingSFT(
        model_path=args.model_path,
        X=X,
        y=y,
        output_dir=args.output_dir,
        device="cpu" if args.cpu else None,
    )

    results = experiment.run(
        n_iterations=args.iterations,
        samples_per_iteration=args.samples,
        r2_threshold=args.threshold,
    )

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(args.output_dir) / f"results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
