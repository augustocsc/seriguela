import os
import torch
import numpy as np
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from datasets import Dataset
from peft import PeftModel, AutoPeftModelForCausalLM
import sys
from transformers import AutoModelForCausalLM

# Add path for Expression class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../classes')))
from expression import Expression
from dataset import RegressionDataset

# === Reward function ===
def compute_reward(expression_str: str) -> float:
    try:
        expr = Expression(expression_str)
        
        # Check if the expression is valid and can be evaluated
        if expr.is_valid_on_dataset(X):
            score = expr.fit_constants(X, y)
            return max(0.1 , (float(score) if np.isfinite(score) else -1.0))
        else:
            #print(f"Expressão inválida: {expression_str}")
            return -1.0
    except Exception as e:
        #print(f"Erro ao avaliar expressão: {expression_str} - {e}")
        return -1.0

# === Helper to extract expression ===
def extract_expression(response: str) -> str:
    return response.split("expr: ")[1].split("<|endoftext|>")[0].strip()

# === Load Data ===
#reg = RegressionDataset('../data/evaluate/srsd-feynman_hard/train', 'feynman-bonus.12.txt', delimiter=' ')
reg = RegressionDataset('./data/evaluate/srsd-feynman_easy/train', 'feynman-i.18.16.txt', delimiter=' ')
X, y = reg.get_numpy()

# === Configs ===
BASE_MODEL = "augustocsc/Se124M100KInfPrompt_EOS_Merged"
LORA_REPO = "augustocsc/Se124M100KInfPrompt_EOS_Merged"
TOKENIZER_REPO = LORA_REPO

# ppo_config = PPOConfig(
#     #model_name=BASE_MODEL,
#     learning_rate=1e-5,
#     batch_size=32,
#     mini_batch_size=8,
#     gradient_accumulation_steps=1,
# )


model = AutoModelForCausalLMWithValueHead.from_pretrained(BASE_MODEL)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(BASE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
ref_model = ref_model.to(device)



import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import numpy as np

def get_safe_functions(X, functions=['log', 'sqrt', 'asin', 'tan', 'abs', 'exp', 'sin', 'cos']):
    """
    Returns a list of functions from `functions` that are safe to use on all columns of X.

    Parameters:
        X: np.ndarray of shape (n_samples, n_features)
        functions: list of function names to check

    Returns:
        List of function names that are safe to use given the data
    """
    safe_functions = []

    for fn in functions:
        if fn in {'sin', 'cos', 'exp', 'abs'}:
            # These are defined for all real values
            safe_functions.append(fn)

        elif fn == 'log':
            if np.all(X > 0):
                safe_functions.append(fn)

        elif fn == 'sqrt':
            if np.all(X >= 0):
                safe_functions.append(fn)

        elif fn == 'asin':
            if np.all((X >= -1) & (X <= 1)):
                safe_functions.append(fn)

        elif fn == 'tan':
            # Check if cos(x) ≈ 0 anywhere → tan(x) will explode
            # We use np.cos to simulate tan issues (e.g., near π/2, 3π/2, etc.)
            cos_vals = np.cos(X)
            if np.all(np.abs(cos_vals) > 1e-6):  # adjustable tolerance
                safe_functions.append(fn)

        # else skip unknown functions

    return safe_functions


safe_functions = get_safe_functions(X)

from tqdm import tqdm

ppo_config = PPOConfig(
    model_name=None,  # definimos o modelo manualmente
    learning_rate=1e-5,
    batch_size=1024,            # total prompts/responses por step
    mini_batch_size=64,        # 4 minibatches por batch
    gradient_accumulation_steps=1,
    ppo_epochs=4,              # 4 passes por minibatch
    log_with=None,             # ou "wandb"
    optimize_cuda_cache=True,  # 👍 melhora uso da A100
)

# === PPO Trainer ===
ppo_trainer = PPOTrainer(
    config=ppo_config,
    tokenizer=tokenizer,
    model=model,
    ref_model=ref_model,
    
)

# Define the prompt with the safe functions
# PROMPT = f"""
# vars: x_1, x_2, x_3
# oper: *, +, /, **, {', '.join(safe_functions)}
# cons: C
# expr:"""



PROMPT = f"""
vars: {", ".join([f"x_{i+1}" for i in range(X.shape[1])])}
oper: *, sin
cons: C
expr:"""

# === Dummy dataset ===
dummy_dataset = Dataset.from_dict({
    "prompt": [PROMPT] * 1024
})

# saving current timestamp for logging
import datetime
import json
import subprocess
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M")

# Get the device of the model
device = next(model.parameters()).device

# === PPO Training Loop ===
# Tokenize the prompt and convert it to tensors
inputs = tokenizer([PROMPT] * ppo_config.batch_size, return_tensors="pt", padding=True)

# Move inputs to the same device as the model
inputs = {key: value.to(device) for key, value in inputs.items()}

# Clear the terminal before starting training
subprocess.run("clear", shell=True)
# Convert the batch tensor into a list of individual tensors
queries = [inputs["input_ids"][i] for i in range(inputs["input_ids"].size(0))]
all_rewards = []
all_responses = []
for epoch in tqdm(range(10), desc="Training Epochs"):  # adjust as needed
    responses = []
    constants = []
    rewards = []
    for i in tqdm(range(ppo_config.batch_size), desc="Batch Progress", leave=False):  # Nested progress bar
        try:
            input_ids = inputs["input_ids"][i].unsqueeze(0)
            attention_mask = inputs["attention_mask"][i].unsqueeze(0)

            # === VALIDATION PATCH ===
            assert torch.all((input_ids >= 0) & (input_ids < model.config.vocab_size)), \
                f"Token inválido detectado: max={input_ids.max().item()}, vocab_size={model.config.vocab_size}"

            # (opcional)
            model.config.pad_token_id = tokenizer.pad_token_id
            reward = -1
            while reward < 0:
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=30,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.5,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=False
                )
                response_ids = output.sequences[0][input_ids.shape[1]:]
                response = tokenizer.decode(response_ids, skip_special_tokens=True)

                reward = compute_reward(response)


        except Exception as e:
            print(f"Error at index {i}: {e}")
            print(f"Input IDs: {input_ids}")
            print(f"Token range: min={input_ids.min()}, max={input_ids.max()}, vocab_size={model.config.vocab_size}")
            raise e

        responses.append(response)
        rewards.append(reward)
    all_responses.extend(responses)
    all_rewards.extend(rewards)

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../output"))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"responses_{timestamp}.txt")

    # If file does not exist, write model and PPO config at the top
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("# Model config:\n")
            f.write(json.dumps(model.config.to_dict(), indent=2))
            f.write("\n# PPO config:\n")
            f.write(json.dumps(ppo_config.__dict__, indent=2))
            f.write("\n# Responses and rewards:\n")

    # Append responses and rewards for this epoch
    with open(output_file, "a") as f:
        for expr_str, rew in zip(responses, rewards):
            f.write(json.dumps({"expression": expr_str, "reward": float(rew)}) + "\n")
    
    #if one reward is >= .9 break
    if any(r >= 0.9 for r in rewards):
        print("Reward >= 0.9 found, stopping training.")
        break

    # Compute rewards with a progress bar
    
    import concurrent.futures

    # # Use process-based parallelism
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     rewards = list(tqdm(executor.map(compute_reward, responses), total=len(responses), desc="Computing Rewards", leave=False))
  
    #rewards = [ compute_reward(response) for response in tqdm(responses, desc="Computing Rewards", leave=False)]
  
    # Convert rewards to a list of PyTorch tensors
    rewards = [torch.tensor(reward, dtype=torch.float32, device=device) for reward in rewards]
    
    # Ensure responses are also tokenized and converted to tensors
    responses = [tokenizer(response, return_tensors="pt", padding=True)["input_ids"].squeeze(0).to(device) for response in responses]

    # Pass the tokenized tensors to ppo_trainer.step()
    ppo_trainer.step(queries, responses, rewards)

    # Log top expressions
    top_k = 3
    sorted_responses = sorted(zip(responses, rewards), key=lambda x: -x[1])
    print(f"\nEpoch {epoch + 1} melhores expressões:")
    for i, (expr, score) in enumerate(sorted_responses[:top_k]):
        print(f"{i+1}. {tokenizer.decode(expr, skip_special_tokens=True)} -> R² = {score:.4f}")
    # Print average, median, and std of rewards
    avg_reward = torch.mean(torch.stack(rewards)).item()
    median_reward = torch.median(torch.stack(rewards)).item()
    count_invalid = sum(1 for r in rewards if r == -1.0)
    print(f"Average Reward: {avg_reward:.4f}, Median Reward: {median_reward:.4f}, Invalid Count: {count_invalid}")

