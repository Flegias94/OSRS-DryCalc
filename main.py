from __future__ import annotations


from PyQt5 import QtWidgets
import sys
import pyqtgraph as pg

from encounters import Encounter
from plot_controller import PlotController
from run_state import RunState
from boss_loader import load_encounters


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, encounters):
        super().__init__()

        self.state = RunState(kc=0)
        self.encounters = encounters
        self.encounter = next(iter(encounters.values()))

        self.target_inputs = {}
        self.target_rows = []   # add this

        self.setWindowTitle(f"{self.target_inputs} - fucked :)")
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
        self.controls = controls 

        controls.addWidget(QtWidgets.QLabel("Select Boss"), 0, 0)

        self.encounter_select = QtWidgets.QComboBox()
        self.encounter_select.addItems([e.display_name for e in self.encounters.values()])
        controls.addWidget(self.encounter_select, 0, 1)

        self.rebuild_target_inputs()


        # --- KC row (input + -/+)
        controls.addWidget(QtWidgets.QLabel("KC"), 1, 0)

        self.kc_input = QtWidgets.QSpinBox()
        self.kc_input.setRange(0, 2000000)
        self.kc_input.setValue(self.state.kc)
        self.kc_input.setKeyboardTracking(False)  # don't spam updates while typing
        controls.addWidget(self.kc_input, 1, 1)

        self.kc_minus = QtWidgets.QPushButton("−")
        self.kc_plus = QtWidgets.QPushButton("+")
        self.kc_minus.setFixedWidth(40)
        self.kc_plus.setFixedWidth(40)
        controls.addWidget(self.kc_minus, 1, 2)
        controls.addWidget(self.kc_plus, 1, 3)


        # Stretch column 1 so spinboxes look clean
        controls.setColumnStretch(1, 1)

        # Signals
        self.encounter_select.currentIndexChanged.connect(self.on_encounter_set)
        self.kc_input.valueChanged.connect(self.on_kc_set)
        
        # Initial draw
        self.refresh()

    def rebuild_target_inputs(self):
        # remove old widgets
        for w in self.target_rows:
            self.controls.removeWidget(w)
            w.deleteLater()

        self.target_rows.clear()
        self.target_inputs.clear()

        row = 2

        for target in self.encounter.targets:

            label = QtWidgets.QLabel(target.display_name)

            spin = QtWidgets.QSpinBox()
            spin.setRange(0, 9999)
            spin.setValue(self.state.get_count(target.key))
            spin.setKeyboardTracking(False)

            minus = QtWidgets.QPushButton("−")
            plus = QtWidgets.QPushButton("+")

            minus.setFixedWidth(40)
            plus.setFixedWidth(40)

            self.controls.addWidget(label, row, 0)
            self.controls.addWidget(spin, row, 1)
            self.controls.addWidget(minus, row, 2)
            self.controls.addWidget(plus, row, 3)

            self.target_rows.extend([label, spin, minus, plus])
            self.target_inputs[target.key] = spin

            spin.valueChanged.connect(
                lambda value, k=target.key: self.on_target_set(k, value)
            )

            minus.clicked.connect(lambda _, s=spin: s.setValue(s.value() - 1))
            plus.clicked.connect(lambda _, s=spin: s.setValue(s.value() + 1))

            row += 1



    def refresh(self) -> None:
        self.plot.update_state(self.encounter, self.state)

    # --- Setters from UI fields (single source of truth: state)
    def on_encounter_set(self, index: int):
        self.encounter = list(self.encounters.values())[index]

        self.plot.set_encounter(self.encounter, n_max=2000)

        self.state.drops.clear()

        self.rebuild_target_inputs()

        self.refresh()

        # reset inputs
        for key in self.target_inputs:
            self.state.drops[key] = 0

        self.refresh()

    def on_kc_set(self, value: int) -> None:
        self.state.kc = int(value)
        self.refresh()

    def on_target_set(self, key: str, value: int):
        self.state.drops[key] = int(value)
        self.refresh()


# =========================
# Demo / test run
# =========================
def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    encounters = load_encounters()

    win = MainWindow(encounters)
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
