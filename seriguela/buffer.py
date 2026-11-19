"""Memórias de replay. Guardam a sequência de TOKENS original da amostra, de modo
que a probabilidade reavaliada é a da mesma ação que foi amostrada (na v2 o texto
era retokenizado, e a sequência podia ser outra).

* ``EliteBuffer``     -- as K melhores expressões distintas por recompensa.
* ``ReservoirBuffer`` -- amostra uniforme de tudo o que já foi visto (controle:
  separa "reaprender as melhores" de "ter mais dados no update").
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass(order=True)
class Entry:
    reward: float
    order: int                                  # desempate estável
    expression: str = field(compare=False)
    tokens: tuple = field(compare=False)
    r2: float = field(compare=False, default=0.0)


class EliteBuffer:
    kind = "elite"

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._heap: List[Entry] = []
        self._seen: Dict[str, Entry] = {}
        self._count = 0

    def __len__(self) -> int:
        return len(self._heap)

    def add(self, expression: str, tokens, reward: float, r2: float) -> bool:
        if reward <= 0 or expression in self._seen:
            return False
        self._count += 1
        entry = Entry(reward, self._count, expression, tuple(tokens), r2)
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, entry)
        elif entry > self._heap[0]:
            del self._seen[heapq.heapreplace(self._heap, entry).expression]
        else:
            return False
        self._seen[expression] = entry
        return True

    def sample(self, k: int, rng: np.random.Generator) -> List[Entry]:
        k = min(k, len(self._heap))
        if k == 0:
            return []
        picks = rng.choice(len(self._heap), size=k, replace=False)
        return [self._heap[i] for i in picks]


class ReservoirBuffer:
    """Amostragem de reservatório (Algoritmo R) sobre expressões válidas distintas."""
    kind = "random"

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._items: List[Entry] = []
        self._seen = set()
        self._count = 0

    def __len__(self) -> int:
        return len(self._items)

    def add(self, expression: str, tokens, reward: float, r2: float, rng: np.random.Generator) -> bool:
        if expression in self._seen:
            return False
        self._seen.add(expression)
        self._count += 1
        entry = Entry(reward, self._count, expression, tuple(tokens), r2)
        if len(self._items) < self.capacity:
            self._items.append(entry)
            return True
        slot = int(rng.integers(0, self._count))
        if slot < self.capacity:
            self._items[slot] = entry
            return True
        return False

    def sample(self, k: int, rng: np.random.Generator) -> List[Entry]:
        k = min(k, len(self._items))
        if k == 0:
            return []
        picks = rng.choice(len(self._items), size=k, replace=False)
        return [self._items[i] for i in picks]
