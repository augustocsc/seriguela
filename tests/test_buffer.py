import numpy as np

from seriguela.buffer import EliteBuffer, ReservoirBuffer


def test_elite_keeps_the_best_distinct_expressions():
    buf = EliteBuffer(capacity=3)
    for i, reward in enumerate([0.1, 0.9, 0.5, 0.7, 0.3]):
        buf.add(f"e{i}", (i,), reward, reward)
    assert sorted(e.reward for e in buf._heap) == [0.5, 0.7, 0.9]
    assert not buf.add("e1", (1,), 0.99, 0.99)        # repetida
    assert not buf.add("zero", (9,), 0.0, 0.0)        # sem recompensa não é elite
    assert len(buf.sample(10, np.random.default_rng(0))) == 3


def test_elite_stores_the_original_tokens():
    buf = EliteBuffer()
    buf.add("x_1", (7, 8, 9), 0.4, 0.4)
    assert buf.sample(1, np.random.default_rng(0))[0].tokens == (7, 8, 9)


def test_reservoir_is_roughly_uniform():
    rng = np.random.default_rng(0)
    kept = np.zeros(200)
    for _ in range(300):
        buf = ReservoirBuffer(capacity=20)
        for i in range(200):
            buf.add(f"e{i}", (i,), 0.0, 0.0, rng)
        for entry in buf._items:
            kept[entry.tokens[0]] += 1
    assert len(buf) == 20
    assert abs(kept[:100].sum() - kept[100:].sum()) < 0.1 * kept.sum()
