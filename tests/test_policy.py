import numpy as np
import torch

from seriguela.policy import build_prompt


def test_prompt_ends_inside_the_expr_field():
    prompt = build_prompt(2)
    assert prompt.endswith('"expr": "') and '"x_2"' in prompt and '"cons": "C"' in prompt


def test_samples_end_at_the_first_quote_or_eos(policy):
    torch.manual_seed(0)
    samples = policy.sample(200, 1.0)
    assert len(samples) == 200
    for s in samples:
        body, last = s.tokens[:-1], s.tokens[-1]
        assert all(t != 1 and t != 0 for t in body)               # nada depois do terminador
        terminated = last in (0, 1)
        assert (s.expression is not None) == terminated
        assert terminated or len(s.tokens) == policy.max_new_tokens
        assert len(s.logp) == len(s.tokens)


def test_sampling_and_scoring_agree(policy):
    """A distribuição amostrada é a que a atualização diferencia (falhava na v2)."""
    for temperature in (1.0, 0.5):
        torch.manual_seed(1)
        samples = policy.sample(64, temperature)
        scored = policy.score([s.tokens for s in samples], temperature)
        for s, logp in zip(samples, scored):
            assert np.allclose(s.logp, logp, atol=1e-4)


def test_scoring_twice_gives_ratio_one(policy):
    torch.manual_seed(2)
    seqs = [s.tokens for s in policy.sample(32, 0.8)]
    a, mask, _ = policy.log_probs(seqs, 0.8)
    b, _, _ = policy.log_probs(seqs, 0.8, grad=True)
    assert torch.allclose(a[mask], b[mask], atol=1e-6)


def test_reference_policy_is_the_prior(policy):
    torch.manual_seed(3)
    seqs = [s.tokens for s in policy.sample(32, 1.0)]
    now, prior = policy.score(seqs, 1.0), policy.score(seqs, 1.0, ref=True)
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(now, prior))     # LoRA nasce com B = 0
    with torch.no_grad():
        for name, p in policy.model.named_parameters():
            if "lora_B" in name:
                p.add_(0.5)
    moved, prior_after = policy.score(seqs, 1.0), policy.score(seqs, 1.0, ref=True)
    assert any(not np.allclose(a, b, atol=1e-4) for a, b in zip(moved, prior))
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(prior, prior_after))


def test_only_the_new_lora_is_trainable_and_weights_are_float32(policy):
    names = [n for n, p in policy.model.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
    assert all(p.dtype == torch.float32 for p in policy.model.parameters())
