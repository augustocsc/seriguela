"""Política minúscula e hermética: GPT-2 aleatório de 2 camadas + tokenizer de 16 símbolos.
Nenhum teste baixa pesos nem usa GPU."""
import numpy as np
import pytest
import torch

from seriguela.data import Problem
from seriguela.policy import Policy


class ToyTokenizer:
    """Maior casamento sobre um vocabulário pequeno o bastante para o acaso gerar expressões válidas."""
    vocab = ["<eos>", '"', "x_1", "+", "*", "-", "(", ")", "C", "sin(", "cos(", "**", "2", "/", " ", "?"]
    eos_token_id = 0

    def encode(self, text):
        ids, i = [], 0
        ordered = sorted(range(len(self.vocab)), key=lambda k: -len(self.vocab[k]))
        while i < len(text):
            for k in ordered:
                if k and text.startswith(self.vocab[k], i):
                    ids.append(k)
                    i += len(self.vocab[k])
                    break
            else:
                ids.append(len(self.vocab) - 1)
                i += 1
        return ids

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)


def make_policy(seed=0, **kwargs):
    from peft import LoraConfig, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(seed)
    tokenizer = ToyTokenizer()
    config = GPT2Config(vocab_size=len(tokenizer.vocab), n_layer=2, n_head=2, n_embd=32, n_positions=128,
                        bos_token_id=0, eos_token_id=0,
                        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = get_peft_model(GPT2LMHeadModel(config),
                           LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], lora_dropout=0.0, bias="none", fan_in_fan_out=True))
    kwargs.setdefault("max_new_tokens", 12)
    kwargs.setdefault("gen_batch", 64)
    return Policy(model, tokenizer, 'x_1 "', device="cpu", **kwargs)


@pytest.fixture
def policy():
    return make_policy()


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    x_train, x_test = rng.uniform(-1, 1, (20, 1)), rng.uniform(-1, 1, (50, 1))
    f = lambda x: x[:, 0] ** 2 + x[:, 0]
    return Problem("toy", x_train, f(x_train), x_test, f(x_test), truth="x_1**2 + x_1")
