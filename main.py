from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple

from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import numpy as np
import pyqtgraph as pg


# =========================
# Math / Distributions
# =========================
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


# =========================
# Encounter model
# =========================
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


# =========================
# Run state (your progress)
# =========================
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


# =========================
# "Your curve" (remaining)
# =========================
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

    return float(np.clip(prod, 0.0, 1.0))


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


# =========================
# Plotting
# =========================
class PlotController:
    """
    Completion view:

    - Base curve:   F_base(n) = P(done by n) from scratch (Enhanced AND 6 Armor)
    - Your curve:   F_you(n)  = P(done by n | your current drops at KC0) (future only)

    Overlay:
    - Solid vertical line at current KC
    - Dotted horizontal line at your completion progress (y = progress)
      from KC to projected KC (where base CDF reaches your progress)
    - Dotted vertical line at projected KC (green/red based on ahead/behind)
    - Projected KC label placed near the bottom of the plot (ViewBox space)
    - Predicted KC (99.99%) based on YOUR curve: dashed vertical line + label
    - Right-aligned info box (ViewBox space):
        * Expected(done), You(progress), Ahead/Behind
        * Reference odds (static)
        * Exact odds @ KC for Enhanced and Armor using BINOMIAL:
              P(X = k), P(X < k), P(X > k)
          and "Both (exact)" = P_enh_exact * P_arm_exact (independence)
        * Predicted KC (99.99%) section (additive)
    """

    def __init__(self, plot_widget: pg.PlotWidget):
        self.plot = plot_widget

        # Curves
        self.base_curve = self.plot.plot([], [], name="Base CDF")
        self.you_curve = self.plot.plot([], [], name="Your curve")

        # ---- Colors / hierarchy (dark UI friendly)
        # Base = soft blue (slightly transparent), Player = bright cyan
        self.base_curve.setPen(pg.mkPen((77, 163, 255, 160), width=2))
        self.you_curve.setPen(pg.mkPen("#00E5FF", width=3))

        # Current KC marker (solid)
        self.kc_line = pg.InfiniteLine(angle=90, movable=False)
        self.plot.addItem(self.kc_line)

        # Dotted pen
        self.dotted_pen = pg.mkPen(style=QtCore.Qt.DotLine)

        # Projected KC marker (dotted vertical)
        self.proj_kc_line = pg.InfiniteLine(
            angle=90, movable=False, pen=self.dotted_pen
        )
        self.plot.addItem(self.proj_kc_line)

        # Horizontal dotted connector (your progress level)
        self.h_line = self.plot.plot([], [], pen=self.dotted_pen)
        self.h_line.setZValue(50)

        # Points (optional)
        self.expected_point = pg.ScatterPlotItem()
        self.you_point = pg.ScatterPlotItem()
        self.plot.addItem(self.expected_point)
        self.plot.addItem(self.you_point)

        # Axis labels / ranges
        self.plot.setLabel("bottom", "Kill Count (KC)")
        self.plot.setLabel("left", "Probability / Percent")
        self.plot.setYRange(0.0, 1.0)

        # -----------------------------
        # Projected KC label (stable; ViewBox space)
        # -----------------------------
        self.proj_label = pg.TextItem(anchor=(0.3, 1.5))
        self.proj_label.setZValue(1000)
        self.plot.addItem(self.proj_label)

        font_proj = QtGui.QFont()
        font_proj.setPointSize(10)
        font_proj.setBold(True)
        self.proj_label.setFont(font_proj)
        self.proj_label.setColor("w")
        self.proj_label.fill = pg.mkBrush(0, 0, 0, 200)
        self.proj_label.border = pg.mkPen(255, 255, 255, 80)

        # -----------------------------
        # Predicted KC (YOUR curve)  [ADDED]
        # -----------------------------
        self._pred_target = 0.9999  # 99.99%

        self.pred_pen = pg.mkPen("#FFC857", style=QtCore.Qt.DashLine, width=2)
        self.pred_kc_line = pg.InfiniteLine(angle=90, movable=False, pen=self.pred_pen)
        self.plot.addItem(self.pred_kc_line)

        self.pred_label = pg.TextItem(anchor=(0.3, 1.5))
        self.pred_label.setZValue(1000)
        self.plot.addItem(self.pred_label)

        font_pred = QtGui.QFont()
        font_pred.setPointSize(10)
        font_pred.setBold(True)
        self.pred_label.setFont(font_pred)
        self.pred_label.setColor("w")
        self.pred_label.fill = pg.mkBrush(0, 0, 0, 200)
        self.pred_label.border = pg.mkPen(255, 255, 255, 80)

        # -----------------------------
        # Right-aligned info box (stable; ViewBox space)
        # -----------------------------
        self.right_info = pg.TextItem(anchor=(0, 0))
        self.right_info.setZValue(1000)
        self.plot.addItem(self.right_info)

        font_info = QtGui.QFont()
        font_info.setPointSize(10)
        font_info.setBold(True)
        self.right_info.setFont(font_info)
        self.right_info.setColor("w")
        self.right_info.fill = pg.mkBrush(0, 0, 0, 200)
        self.right_info.border = pg.mkPen(255, 255, 255, 80)

        # Cached base curve
        self._n_max = 2000
        self._base_y: np.ndarray | None = None
        self._encounter: Encounter | None = None

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _completion_progress(encounter: Encounter, state: RunState) -> float:
        parts: list[float] = []
        for t in encounter.targets:
            got = state.get_count(t.key)

            # Enhanced: treat as "have it or not" for completion progress
            if isinstance(t.distribution, GeometricDist):
                parts.append(1.0 if got >= 1 else 0.0)

            # Armor: linear fraction toward required_count for completion progress
            elif isinstance(t.distribution, NegBinomDist):
                if t.required_count <= 0:
                    parts.append(1.0)
                else:
                    parts.append(min(1.0, got / t.required_count))
            else:
                raise TypeError(f"Unknown distribution type: {type(t.distribution)}")

        if not parts:
            return 0.0
        return float(np.clip(sum(parts) / len(parts), 0.0, 1.0))

    @staticmethod
    def _project_x_for_y(y_curve: np.ndarray, y_target: float) -> int:
        y_target = float(y_target)
        if y_target <= 0.0:
            return 0
        if y_target >= float(y_curve[-1]):
            return len(y_curve) - 1
        return int(np.searchsorted(y_curve, y_target, side="left"))

    @staticmethod
    def _predict_x_for_prob(y_curve: np.ndarray, prob_target: float) -> int:
        """
        Return the smallest index x such that y_curve[x] >= prob_target.
        For YOUR curve, x = additional kills needed from 'now'.
        """
        prob_target = float(prob_target)
        if prob_target <= 0.0:
            return 0
        if prob_target >= 1.0:
            prob_target = 0.999999999999

        if prob_target >= float(y_curve[-1]):
            return len(y_curve) - 1

        return int(np.searchsorted(y_curve, prob_target, side="left"))

    def _right_stack_anchor(self) -> tuple[float, float]:
        vb = self.plot.getViewBox()
        (xmin, xmax), (ymin, ymax) = vb.viewRange()
        x = xmin + 0.74 * (xmax - xmin)
        y = ymin + 0.50 * (ymax - ymin)
        return x, y

    @staticmethod
    def _fmt_pct(prob: float, digits: int = 6) -> str:
        prob = float(np.clip(prob, 0.0, 1.0))
        return f"{prob*100:.{digits}f}%"

    @staticmethod
    def _fmt_one_in(prob: float) -> str:
        prob = float(np.clip(prob, 0.0, 1.0))
        if prob <= 0.0:
            return "∞"
        return f"{1.0/prob:,.2f}"

    def set_encounter(self, encounter: Encounter, n_max: int = 2000) -> None:
        if n_max <= 0:
            raise ValueError("n_max must be > 0.")
        self._n_max = n_max
        self._encounter = encounter

        base_dist = encounter.combined_distribution()
        y = base_dist.curve(n_max)
        x = np.arange(n_max + 1)

        self._base_y = y
        self.base_curve.setData(x, y)
        self.plot.setXRange(0, n_max)

    def update_state(self, encounter: Encounter, state: RunState) -> None:
        if self._base_y is None or self._encounter is None:
            self.set_encounter(encounter, n_max=self._n_max)

        kc = int(state.kc)
        kc = max(0, min(kc, self._n_max))
        self.kc_line.setValue(kc)

        # Expected completion probability at KC (base curve)
        expected_y = float(self._base_y[kc])

        # Your completion progress (0..1)
        you_y = self._completion_progress(encounter, state)

        # Points
        self.expected_point.setData([kc], [expected_y])
        self.you_point.setData([kc], [you_y])

        # Project your progress onto the base curve  (Projected KC = "on-curve equivalent")
        kc_proj = self._project_x_for_y(self._base_y, you_y)
        self.proj_kc_line.setValue(kc_proj)

        # Horizontal dotted line from current KC to projected KC at y=you_y
        self.h_line.setData([kc, kc_proj], [you_y, you_y])

        # Ahead/behind coloring for projected overlays (unchanged)
        delta = kc_proj - kc
        if delta > 0:
            status = f"Ahead by {delta} KC"
            pen = pg.mkPen("g", style=QtCore.Qt.DotLine)
        elif delta < 0:
            status = f"Behind by {abs(delta)} KC"
            pen = pg.mkPen("r", style=QtCore.Qt.DotLine)
        else:
            status = "On-curve"
            pen = self.dotted_pen

        self.proj_kc_line.setPen(pen)
        self.h_line.setPen(pen)

        # -----------------------------
        # Stable label placement (ViewBox space)
        # -----------------------------
        vb = self.plot.getViewBox()
        (xmin, xmax), (ymin, ymax) = vb.viewRange()
        y_text = ymin + 0.02 * (ymax - ymin)
        self.proj_label.setText(f"Projected KC: {kc_proj}")
        self.proj_label.setPos(kc_proj, y_text)

        # -----------------------------
        # Exact odds @ KC (OSRS dry-calc style)
        # -----------------------------
        enh_k = int(state.get_count("enhanced_seed"))
        arm_k = int(state.get_count("armor_seeds"))

        enh_p = 1.0 / 400.0
        arm_p = 1.0 / 50.0

        enh_exact = binom_pmf(kc, enh_k, enh_p)
        enh_lt = binom_cdf_lt(kc, enh_k, enh_p)  # P(X < k)
        enh_gt = float(np.clip(1.0 - (enh_lt + enh_exact), 0.0, 1.0))

        arm_exact = binom_pmf(kc, arm_k, arm_p)
        arm_lt = binom_cdf_lt(kc, arm_k, arm_p)
        arm_gt = float(np.clip(1.0 - (arm_lt + arm_exact), 0.0, 1.0))

        both_exact = float(np.clip(enh_exact * arm_exact, 0.0, 1.0))

        # -----------------------------
        # Your curve (remaining / conditional)
        # -----------------------------
        rem_dist = remaining_combined_distribution(encounter, state)
        m_max = self._n_max - kc
        y_rem = rem_dist.curve(m_max)
        x_rem = np.arange(kc, self._n_max + 1)
        self.you_curve.setData(x_rem, y_rem)

        # -----------------------------
        # Predicted KC (99.99%) based on YOUR curve  [ADDED]
        # y_rem is indexed by "additional kills"
        # -----------------------------
        addl = self._predict_x_for_prob(y_rem, self._pred_target)
        kc_pred = kc + addl
        self.pred_kc_line.setValue(kc_pred)

        y_text_pred = ymin + 0.06 * (ymax - ymin)  # slightly above projected label
        self.pred_label.setText(f"Predicted KC (99.99%): {kc_pred}  (+{addl})")
        self.pred_label.setPos(kc_pred, y_text_pred)

        # -----------------------------
        # Right-side info box
        # -----------------------------
        info = (
            f"Expected (done): {expected_y*100:.1f}%\n"
            f"You (progress): {you_y*100:.1f}%\n"
            f"{status}\n"
            f"\n"
            f"Reference odds:\n"
            f"  Enhanced: 1/400 (0.25%)\n"
            f"  Armor:    1/50  (2.00%)\n"
            f"\n"
            f"Exact odds @ {kc} KC:\n"
            f"Enhanced (k={enh_k}):\n"
            f"  exactly: {self._fmt_pct(enh_exact)} (1/{self._fmt_one_in(enh_exact)})\n"
            f"  <k:      {self._fmt_pct(enh_lt)} (1/{self._fmt_one_in(enh_lt)})\n"
            f"  >k:      {self._fmt_pct(enh_gt)} (1/{self._fmt_one_in(enh_gt)})\n"
            f"Armor (k={arm_k}):\n"
            f"  exactly: {self._fmt_pct(arm_exact)} (1/{self._fmt_one_in(arm_exact)})\n"
            f"  <k:      {self._fmt_pct(arm_lt)} (1/{self._fmt_one_in(arm_lt)})\n"
            f"  >k:      {self._fmt_pct(arm_gt)} (1/{self._fmt_one_in(arm_gt)})\n"
            f"Both (exact): {self._fmt_pct(both_exact)} (1/{self._fmt_one_in(both_exact)})"
        )

        # Additive predicted section
        info += (
            f"\n\nPredicted (99.99%) on YOUR curve:\n"
            f"  KC: {kc_pred}\n"
            f"  Remaining: {max(0, addl)} KC"
        )

        self.right_info.setText(info)
        x_info, y_info = self._right_stack_anchor()
        self.right_info.setPos(x_info, y_info)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, encounter: Encounter):
        super().__init__()
        self.encounter = encounter
        self.state = RunState(kc=0)

        self.setWindowTitle("Red prison: get fucked rate.")
        self.resize(1200, 900)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Plot
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.plot = PlotController(self.plot_widget)
        self.plot.set_encounter(self.encounter, n_max=2000)

        # Controls area
        controls = QtWidgets.QGridLayout()
        layout.addLayout(controls)

        # --- KC row (input + -/+)
        controls.addWidget(QtWidgets.QLabel("KC"), 0, 0)

        self.kc_input = QtWidgets.QSpinBox()
        self.kc_input.setRange(0, 2000000)
        self.kc_input.setValue(self.state.kc)
        self.kc_input.setKeyboardTracking(False)  # don't spam updates while typing
        controls.addWidget(self.kc_input, 0, 1)

        self.kc_minus = QtWidgets.QPushButton("−")
        self.kc_plus = QtWidgets.QPushButton("+")
        self.kc_minus.setFixedWidth(40)
        self.kc_plus.setFixedWidth(40)
        controls.addWidget(self.kc_minus, 0, 2)
        controls.addWidget(self.kc_plus, 0, 3)

        # --- Armor seeds row (input + -/+)
        controls.addWidget(QtWidgets.QLabel("Armor Seeds"), 1, 0)

        self.armor_input = QtWidgets.QSpinBox()
        self.armor_input.setRange(0, 9999)
        self.armor_input.setValue(self.state.get_count("armor_seeds"))
        self.armor_input.setKeyboardTracking(False)
        controls.addWidget(self.armor_input, 1, 1)

        self.armor_minus = QtWidgets.QPushButton("−")
        self.armor_plus = QtWidgets.QPushButton("+")
        self.armor_minus.setFixedWidth(40)
        self.armor_plus.setFixedWidth(40)
        controls.addWidget(self.armor_minus, 1, 2)
        controls.addWidget(self.armor_plus, 1, 3)

        # --- Enhanced row (input + -/+)
        controls.addWidget(QtWidgets.QLabel("Enhanced"), 2, 0)

        self.enh_input = QtWidgets.QSpinBox()
        self.enh_input.setRange(0, 9999)  # keep 0/1 for now
        self.enh_input.setValue(self.state.get_count("enhanced_seed"))
        self.enh_input.setKeyboardTracking(False)
        controls.addWidget(self.enh_input, 2, 1)

        self.enh_minus = QtWidgets.QPushButton("−")
        self.enh_plus = QtWidgets.QPushButton("+")
        self.enh_minus.setFixedWidth(40)
        self.enh_plus.setFixedWidth(40)
        controls.addWidget(self.enh_minus, 2, 2)
        controls.addWidget(self.enh_plus, 2, 3)

        # Stretch column 1 so spinboxes look clean
        controls.setColumnStretch(1, 1)

        # Signals
        self.kc_input.valueChanged.connect(self.on_kc_set)
        self.armor_input.valueChanged.connect(self.on_armor_set)
        self.enh_input.valueChanged.connect(self.on_enh_set)

        self.kc_minus.clicked.connect(
            lambda: self.kc_input.setValue(self.kc_input.value() - 1)
        )
        self.kc_plus.clicked.connect(
            lambda: self.kc_input.setValue(self.kc_input.value() + 1)
        )

        self.armor_minus.clicked.connect(
            lambda: self.armor_input.setValue(self.armor_input.value() - 1)
        )
        self.armor_plus.clicked.connect(
            lambda: self.armor_input.setValue(self.armor_input.value() + 1)
        )

        self.enh_minus.clicked.connect(
            lambda: self.enh_input.setValue(max(0, self.enh_input.value() - 1))
        )
        self.enh_plus.clicked.connect(
            lambda: self.enh_input.setValue(self.enh_input.value() + 1)
        )

        # Initial draw
        self.refresh()

    def refresh(self) -> None:
        self.plot.update_state(self.encounter, self.state)

    # --- Setters from UI fields (single source of truth: state)
    def on_kc_set(self, value: int) -> None:
        self.state.kc = int(value)
        self.refresh()

    def on_armor_set(self, value: int) -> None:
        self.state.drops["armor_seeds"] = int(value)
        self.refresh()

    def on_enh_set(self, value: int) -> None:
        self.state.drops["enhanced_seed"] = int(value)
        self.refresh()


# =========================
# Demo / test run
# =========================
def main() -> None:
    app = QtWidgets.QApplication(sys.argv)

    encounters = build_encounters()
    cg = encounters["corrupted_gauntlet"]

    win = MainWindow(cg)
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
