# =========================
# Run state (your progress)
# =========================
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RunState:
    kc: int = 0
    drops: Dict[str, int] = field(default_factory=dict)  # total counts
    drop_kcs: Dict[str, list[int]] = field(
        default_factory=dict
    )  # history: KCs where drops happened

    def add_kill(self, n: int = 1) -> None:
        self.kc = max(0, self.kc + n)

    def add_drop(self, target_key: str, amount: int = 1) -> None:
        # total
        self.drops[target_key] = self.drops.get(target_key, 0) + amount
        # history (record at current KC)
        if target_key not in self.drop_kcs:
            self.drop_kcs[target_key] = []
        for _ in range(max(0, amount)):
            self.drop_kcs[target_key].append(self.kc)

    def get_count(self, target_key: str) -> int:
        return self.drops.get(target_key, 0)

    def get_drop_kcs(self, target_key: str) -> list[int]:
        return self.drop_kcs.get(target_key, [])