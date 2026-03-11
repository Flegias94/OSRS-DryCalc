

# =========================
# Math / Distributions
# =========================
from dataclasses import dataclass
from typing import Tuple

import numpy as np


class Distribution:
    """Common interface for plotting + point queries."""

    def cdf(self, n: int) -> float:
        raise NotImplementedError

    def curve(self, n_max: int) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class GeometricDist(Distribution):
    """Trials until first success. CDF: 1 - (1-p)^n"""

    p: float

    def cdf(self, n: int) -> float:
        if n <= 0:
            return 0.0
        if not (0.0 < self.p <= 1.0):
            raise ValueError("p must be in (0, 1].")
        return float(1.0 - (1.0 - self.p) ** n)

    def curve(self, n_max: int) -> np.ndarray:
        if n_max < 0:
            raise ValueError("n_max must be >= 0.")
        if not (0.0 < self.p <= 1.0):
            raise ValueError("p must be in (0, 1].")
        n = np.arange(n_max + 1, dtype=np.float64)
        return np.clip(1.0 - (1.0 - self.p) ** n, 0.0, 1.0)


@dataclass(frozen=True)
class NegBinomDist(Distribution):
    """
    Trials until r-th success (video parameterization).
    PMF: P(N=n)=C(n-1,r-1)p^r(1-p)^(n-r), n=r,r+1,...
    CDF computed via stable recurrence.
    """

    r: int
    p: float

    def cdf(self, n: int) -> float:
        r, p = self.r, self.p
        if r <= 0:
            raise ValueError("r must be >= 1.")
        if not (0.0 < p <= 1.0):
            raise ValueError("p must be in (0, 1].")
        if n < r:
            return 0.0

        q = 1.0 - p
        pmf = p**r  # at n=r
        cdf = pmf

        k = r
        while k < n:
            pmf *= (k / (k - r + 1)) * q
            cdf += pmf
            k += 1

        return float(np.clip(cdf, 0.0, 1.0))

    def curve(self, n_max: int) -> np.ndarray:
        r, p = self.r, self.p
        if n_max < 0:
            raise ValueError("n_max must be >= 0.")
        if r <= 0:
            raise ValueError("r must be >= 1.")
        if not (0.0 < p <= 1.0):
            raise ValueError("p must be in (0, 1].")

        out = np.zeros(n_max + 1, dtype=np.float64)
        if n_max < r:
            return out

        q = 1.0 - p
        pmf = p**r
        cdf = pmf
        out[r] = cdf

        k = r
        while k < n_max:
            pmf *= (k / (k - r + 1)) * q
            cdf += pmf
            out[k + 1] = cdf
            k += 1

        return np.clip(out, 0.0, 1.0)


@dataclass(frozen=True)
class IndependentAllOf(Distribution):
    """Combine independent requirements: CDF = product of component CDFs."""

    parts: Tuple[Distribution, ...]

    def cdf(self, n: int) -> float:
        prod = 1.0
        for d in self.parts:
            prod *= d.cdf(n)
        return float(np.clip(prod, 0.0, 1.0))

    def curve(self, n_max: int) -> np.ndarray:
        if n_max < 0:
            raise ValueError("n_max must be >= 0.")
        out = np.ones(n_max + 1, dtype=np.float64)
        for d in self.parts:
            out *= d.curve(n_max)
        return np.clip(out, 0.0, 1.0)


@dataclass(frozen=True)
class CertainDist(Distribution):
    """Already completed requirement."""

    def cdf(self, n: int) -> float:
        return 1.0

    def curve(self, n_max: int) -> np.ndarray:
        return np.ones(n_max + 1, dtype=np.float64)


# -------------------------
# Binomial tail (no SciPy)
# P(X >= k) for X ~ Binomial(n, p)
# -------------------------
def binom_tail_ge(n: int, k: int, p: float) -> float:
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1].")

    q = 1.0 - p

    # pmf(0) = q^n
    pmf = q**n
    cdf = pmf  # sum_{i=0..0}

    # Recurrence: pmf(i+1) = pmf(i) * (n-i)/(i+1) * p/q
    # Sum up to k-1 to get P(X <= k-1), then tail = 1 - that.
    for i in range(0, k - 1):
        if q == 0.0:
            # p=1 -> X=n always
            return 1.0 if k <= n else 0.0
        pmf *= (n - i) / (i + 1) * (p / q)
        cdf += pmf

    return float(np.clip(1.0 - cdf, 0.0, 1.0))

def project_x_for_y(y_curve: np.ndarray, y_target: float) -> int:
    """
    Given a monotone increasing curve y_curve indexed by x=0..n_max,
    return the smallest x such that y_curve[x] >= y_target.
    If y_target is above the curve max, returns n_max.
    """
    y_target = float(y_target)
    if y_target <= 0.0:
        return 0
    if y_target >= float(y_curve[-1]):
        return len(y_curve) - 1

    # np.searchsorted works on ascending arrays
    return int(np.searchsorted(y_curve, y_target, side="left"))


def binom_pmf(n: int, k: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    if p == 0.0:
        return 1.0 if k == 0 else 0.0
    if p == 1.0:
        return 1.0 if k == n else 0.0

    q = 1.0 - p
    pmf0 = q**n
    if k == 0:
        return float(pmf0)

    pmf = pmf0
    for i in range(0, k):
        pmf *= (n - i) / (i + 1) * (p / q)
    return float(pmf)


def binom_cdf_lt(n: int, k: int, p: float) -> float:
    """P(X < k)"""
    if k <= 0:
        return 0.0
    if k > n + 1:
        return 1.0
    # P(X <= k-1) = 1 - P(X >= k)
    return float(np.clip(1.0 - binom_tail_ge(n, k, p), 0.0, 1.0))