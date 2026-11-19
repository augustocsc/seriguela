"""A política: um modelo de linguagem que escreve expressões após um prompt fixo.

Há UM caminho para amostrar (``sample``) e UM para pontuar (``log_probs``), ambos
em float32, com a mesma temperatura, sem top-k/top-p e sem dropout. A distribuição
de onde as amostras saem é, então, exatamente a que a atualização diferencia. Na v2
isso não valia (top-k=50 do ``generate``, temperatura ausente nas elites, dropout
do LoRA ligado no update, pesos em FP16).

Uma ação é a sequência de tokens até o primeiro token que contém aspas (fim do campo
``"expr"``) ou até o EOS. Tokens depois disso não existem para a política.
"""
from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

OPERATORS = ["+", "-", "*", "/", "sin", "cos", "sqrt", "log", "exp", "**"]


def build_prompt(n_vars: int) -> str:
    """Prompt JSON padrão: variáveis, operadores e ``cons``; termina em ``"expr": "``."""
    payload = {"vars": [f"x_{i + 1}" for i in range(n_vars)], "ops": OPERATORS, "cons": "C", "expr": ""}
    return json.dumps(payload)[:-2]


@dataclass
class Sample:
    tokens: Tuple[int, ...]          # inclui o token terminador
    expression: Optional[str]        # None se atingiu o limite sem terminar
    logp: np.ndarray                 # log-prob por token, no momento da amostragem


class Policy:
    def __init__(self, model, tokenizer, prompt: str, device: Optional[str] = None,
                 max_new_tokens: int = 64, gen_batch: int = 256):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).float().eval()
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=self.device)
        self.max_new_tokens = max_new_tokens
        self.gen_batch = gen_batch
        self.eos_id = int(tokenizer.eos_token_id)
        vocab = self.model.get_input_embeddings().num_embeddings
        stops = [i for i in range(vocab) if '"' in tokenizer.decode([i])]
        self._stop = torch.zeros(vocab, dtype=torch.bool, device=self.device)
        self._stop[stops] = True
        self._stop_set = set(stops)

    # ------------------------------------------------------------------ modelo
    def _base(self):
        return self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model

    def _step(self, input_ids, past=None):
        """Logits da ÚLTIMA posição (sem materializar o vocabulário nas demais)."""
        base = self._base()
        if hasattr(base, "transformer") and hasattr(base, "lm_head"):
            out = base.transformer(input_ids=input_ids, past_key_values=past, use_cache=True)
            return base.lm_head(out.last_hidden_state[:, -1]), out.past_key_values
        out = base(input_ids=input_ids, past_key_values=past, use_cache=True)
        return out.logits[:, -1], out.past_key_values

    def _logits_from(self, input_ids, start: int):
        base = self._base()
        if hasattr(base, "transformer") and hasattr(base, "lm_head"):
            return base.lm_head(base.transformer(input_ids=input_ids).last_hidden_state[:, start:])
        return base(input_ids=input_ids).logits[:, start:]

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    # --------------------------------------------------------------- amostragem
    def _expression(self, tokens: Sequence[int]) -> Optional[str]:
        if tokens[-1] in self._stop_set:
            return self.tokenizer.decode(list(tokens)).split('"')[0].strip()
        if tokens[-1] == self.eos_id:
            return self.tokenizer.decode(list(tokens[:-1])).split('"')[0].strip()
        return None

    @torch.no_grad()
    def sample(self, n: int, temperature: float) -> List[Sample]:
        samples: List[Sample] = []
        while len(samples) < n:
            b = min(self.gen_batch, n - len(samples))
            logits, past = self._step(self.prompt_ids.expand(b, -1))
            done = torch.zeros(b, dtype=torch.bool, device=self.device)
            tokens, logps = [], []
            for _ in range(self.max_new_tokens):
                logp = F.log_softmax(logits.float() / temperature, dim=-1)
                nxt = torch.multinomial(logp.exp(), 1).squeeze(1)
                nxt = torch.where(done, torch.full_like(nxt, self.eos_id), nxt)
                tokens.append(nxt)
                logps.append(logp.gather(1, nxt[:, None]).squeeze(1))
                done = done | self._stop[nxt] | (nxt == self.eos_id)
                if bool(done.all()):
                    break
                logits, past = self._step(nxt[:, None], past)
            tokens = torch.stack(tokens, 1).cpu().numpy()
            logps = torch.stack(logps, 1).cpu().numpy()
            for row in range(b):
                length = tokens.shape[1]
                for t in range(tokens.shape[1]):
                    if tokens[row, t] in self._stop_set or tokens[row, t] == self.eos_id:
                        length = t + 1
                        break
                seq = tuple(int(v) for v in tokens[row, :length])
                samples.append(Sample(seq, self._expression(seq), logps[row, :length].copy()))
        return samples

    # ---------------------------------------------------------------- pontuação
    def log_probs(self, sequences: Sequence[Sequence[int]], temperature: float, grad: bool = False,
                  ref: bool = False, entropy: bool = False):
        """Log-prob por token das sequências dadas. Devolve ``(logp, máscara, entropia)``.

        ``ref=True`` usa a política de referência (adaptador desligado = prior do SFT).
        """
        n, length = len(sequences), max(len(s) for s in sequences)
        tokens = torch.full((n, length), self.eos_id, dtype=torch.long, device=self.device)
        mask = torch.zeros((n, length), dtype=torch.bool, device=self.device)
        for i, seq in enumerate(sequences):
            tokens[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
            mask[i, :len(seq)] = True
        prompt_len = self.prompt_ids.shape[0]
        inputs = torch.cat([self.prompt_ids.expand(n, -1), tokens[:, :-1]], dim=1)
        adapter = self.model.disable_adapter() if ref and hasattr(self.model, "disable_adapter") \
            else contextlib.nullcontext()
        with adapter, torch.set_grad_enabled(grad):
            logits = self._logits_from(inputs, prompt_len - 1).float() / temperature
            logp_all = F.log_softmax(logits, dim=-1)
            logp = logp_all.gather(2, tokens[:, :, None]).squeeze(2)
            ent = -(logp_all.exp() * logp_all).sum(-1) if entropy else None
        return logp, mask, ent

    @torch.no_grad()
    def score(self, sequences: Sequence[Sequence[int]], temperature: float, ref: bool = False,
              batch: int = 128) -> List[np.ndarray]:
        """``log_probs`` sem gradiente, em lotes; uma lista de vetores (um por sequência)."""
        out: List[np.ndarray] = []
        for start in range(0, len(sequences), batch):
            chunk = sequences[start:start + batch]
            logp, _, _ = self.log_probs(chunk, temperature, ref=ref)
            logp = logp.cpu().numpy()
            out.extend(logp[i, :len(seq)].copy() for i, seq in enumerate(chunk))
        return out


def load_policy(model_repo: str, n_vars: int, base_model: str = "gpt2", device: Optional[str] = None,
                lora_r: int = 8, lora_alpha: int = 16, **kwargs) -> Policy:
    """Base + adaptador do SFT (fundido) + um LoRA novo e treinável para o RL.

    O LoRA novo nasce com B = 0, logo no passo 0 a política É o prior do SFT, e
    desligar o adaptador devolve esse prior a qualquer momento (referência da KL).
    """
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    base = AutoModelForCausalLM.from_pretrained(base_model)
    if len(tokenizer) != base.config.vocab_size:
        base.resize_token_embeddings(len(tokenizer))
    merged = PeftModel.from_pretrained(base, model_repo).merge_and_unload()
    config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=["c_attn"],
                        lora_dropout=0.0, bias="none", fan_in_fan_out=True)
    return Policy(get_peft_model(merged, config), tokenizer, build_prompt(n_vars), device, **kwargs)
