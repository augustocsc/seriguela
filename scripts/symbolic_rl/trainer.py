import os
import torch
import numpy as np
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from datasets import Dataset
from peft import PeftModel, AutoPeftModelForCausalLM
import sys

# Add path for Expression class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../classes')))
from expression import Expression
from dataset import RegressionDataset

# === Load Data ===
reg = RegressionDataset('./data/evaluate/srsd-feynman_easy/train', 'feynman-i.12.1.txt')
X, y = reg.get_numpy()

# === Configs ===
BASE_MODEL = "gpt2"
LORA_REPO = "augustocsc/Se124M100KInfPrompt_EOS"
TOKENIZER_REPO = LORA_REPO
PROMPT = """
vars: x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10
oper: *, **, +, -, /
cons: C
expr:"""

ppo_config = PPOConfig(
    #model_name=BASE_MODEL,
    learning_rate=1e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
)

# === Load Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
tokenizer.pad_token = tokenizer.eos_token

# === Load base model and apply LoRA ===
base_model = AutoPeftModelForCausalLM.from_pretrained(BASE_MODEL)
peft_model = PeftModel.from_pretrained(base_model, LORA_REPO)

# === Convert to ValueHead model ===
model = AutoModelForCausalLMWithValueHead.from_pretrained(BASE_MODEL)
model.resize_token_embeddings(len(tokenizer))
model.transformer = peft_model.transformer
model.lm_head = peft_model.lm_head

# === Reference model (no LoRA needed) ===
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(BASE_MODEL)
ref_model.resize_token_embeddings(len(tokenizer))
ref_model.eval()

# === Dummy dataset ===
dummy_dataset = Dataset.from_dict({
    "prompt": [PROMPT] * 100
})

# === Reward function ===
def compute_reward(expression_str: str) -> float:
    try:
        expr = Expression(expression_str)
        score = expr.fit_constants(X, y)
        return float(score) if np.isfinite(score) else -1.0
    except Exception as e:
        print(f"Erro ao avaliar expressão: {expression_str} - {e}")
        return -1.0

# === PPO Trainer ===
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    reward_fn=compute_reward,
    train_dataset=dummy_dataset,
)

# === Helper to extract expression ===
def extract_expression(response: str) -> str:
    return response.split("expr: ")[1].split("<|endoftext|>")[0].strip()

# === PPO Training Loop ===
inputs = tokenizer([PROMPT] * ppo_config.batch_size, return_tensors="pt", padding=True).to(model.device)

for epoch in range(10):  # adjust as needed
    responses = []
    for i in range(ppo_config.batch_size):
        output = model.generate(
            input_ids=inputs["input_ids"][i].unsqueeze(0),
            attention_mask=inputs["attention_mask"][i].unsqueeze(0),
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0
        )
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        response = generated[len(PROMPT):].strip()
        responses.append(response)

    rewards = [compute_reward(r) for r in responses]

    # PPO Step
    ppo_trainer.step([PROMPT] * ppo_config.batch_size, responses, rewards)

    # Log top expressions
    top_k = 3
    sorted_responses = sorted(zip(responses, rewards), key=lambda x: -x[1])
    print(f"\nEpoch {epoch + 1} melhores expressões:")
    for i, (expr, score) in enumerate(sorted_responses[:top_k]):
        print(f"{i+1}. {expr} -> R² = {score:.4f}")
