# =========================
# Encounter model
# =========================
from enum import Enum
from typing import Dict

from run_state import RunState
from math_engine import *


class CompletionRule(Enum):
    ALL = "all"
    ANY = "any"


@dataclass(frozen=True)
class DropTarget:
    key: str
    display_name: str
    distribution: Distribution
    required_count: int


@dataclass(frozen=True)
class Encounter:
    key: str
    display_name: str
    rule: CompletionRule
    targets: Tuple[DropTarget, ...]

    def combined_distribution(self) -> Distribution:
        if self.rule == CompletionRule.ALL:
            return IndependentAllOf(tuple(t.distribution for t in self.targets))
        elif self.rule == CompletionRule.ANY:
            raise NotImplementedError("ANY rule not implemented yet.")
        raise ValueError(f"Unknown rule: {self.rule}")


def build_encounters() -> Dict[str, Encounter]:
    cg_targets = (
        DropTarget(
            key="enhanced_seed",
            display_name="Enhanced Crystal Weapon Seed",
            distribution=GeometricDist(p=1 / 400),
            required_count=1,
        ),
        DropTarget(
            key="armor_seeds",
            display_name="Crystal Armor Seed",
            distribution=NegBinomDist(r=6, p=1 / 50),
            required_count=6,
        ),
    )

    cg = Encounter(
        key="corrupted_gauntlet",
        display_name="Corrupted Gauntlet",
        rule=CompletionRule.ALL,
        targets=cg_targets,
    )

    return {cg.key: cg}

def remaining_combined_distribution(
    encounter: Encounter, state: RunState
) -> Distribution:
    """
    Build a new Distribution describing how many MORE kills are needed
    given current drop counts. Assumes independence (video assumption).
    """
    parts: list[Distribution] = []

    for t in encounter.targets:
        got = state.get_count(t.key)
        remaining = max(0, t.required_count - got)

        if remaining == 0:
            parts.append(CertainDist())
            continue

        d = t.distribution
        if isinstance(d, GeometricDist):
            # Still need 1 success
            parts.append(GeometricDist(p=d.p))
        elif isinstance(d, NegBinomDist):
            # Need "remaining" successes
            parts.append(NegBinomDist(r=remaining, p=d.p))
        else:
            raise TypeError(f"Unknown distribution type: {type(d)}")

    return IndependentAllOf(tuple(parts))


def progress_likelihood_at_kc(encounter: Encounter, state: RunState) -> float:
    """
    "Your %" point at current KC:
    Probability that a fresh player would have AT LEAST your current drops by now.

    - Enhanced (geometric): P(X>=1) or P(X>=0)
    - Armor seeds: X ~ Binomial(n=kc, p=1/50) so use tail P(X>=observed)
    Multiply (independence).
    """
    n = int(state.kc)
    if n < 0:
        return 0.0

    prod = 1.0
    for t in encounter.targets:
        got = state.get_count(t.key)
        d = t.distribution

        if isinstance(d, GeometricDist):
            # observed 0 -> P(X>=0)=1; observed >=1 -> P(X>=1)=1-(1-p)^n
            if got <= 0:
                tail = 1.0
            else:
                tail = 1.0 - (1.0 - d.p) ** n
            prod *= tail

        elif isinstance(d, NegBinomDist):
            # observed armor count by KC uses binomial tail
            tail = binom_tail_ge(n=n, k=max(0, got), p=d.p)
            prod *= tail

        else:
            raise TypeError(f"Unknown distribution type: {type(d)}")
