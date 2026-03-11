# =========================
# Plotting
# =========================
import numpy as np

from encounters import Encounter, remaining_combined_distribution

from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
from run_state import RunState
from math_engine import *


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
        # Exact odds @ KC (generic encounter)
        # -----------------------------
        targets = encounter.targets

        exact_results = []

        for t in targets:
            k = int(state.get_count(t.key))
            p = t.distribution.p

            exact = binom_pmf(kc, k, p)
            lt = binom_cdf_lt(kc, k, p)
            gt = float(np.clip(1.0 - (lt + exact), 0.0, 1.0))

            exact_results.append((t.display_name, k, exact, lt, gt))

        # Combined exact (independent)
        both_exact = 1.0
        for _, _, exact, _, _ in exact_results:
            both_exact *= exact

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
            f"{status}\n\n"
        )

        info += "Exact odds @ KC:\n"

        for name, k, exact, lt, gt in exact_results:
            info += (
                f"{name} (k={k}):\n"
                f"  exactly: {self._fmt_pct(exact)} (1/{self._fmt_one_in(exact)})\n"
                f"  <k:      {self._fmt_pct(lt)} (1/{self._fmt_one_in(lt)})\n"
                f"  >k:      {self._fmt_pct(gt)} (1/{self._fmt_one_in(gt)})\n"
            )

        info += f"\nAll targets (exact): {self._fmt_pct(both_exact)} (1/{self._fmt_one_in(both_exact)})"

        # Additive predicted section
        info += (
            f"\n\nPredicted (99.99%) on YOUR curve:\n"
            f"  KC: {kc_pred}\n"
            f"  Remaining: {max(0, addl)} KC"
        )

        self.right_info.setText(info)
        x_info, y_info = self._right_stack_anchor()
        self.right_info.setPos(x_info, y_info)