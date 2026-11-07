"""
Interactive Epoch Spike Viewer (matplotlib plot embedded in a PySide6 GUI)
===========================================================================

Standalone, GUI-based example program built on top of IPFX. This file lives
outside the ipfx package itself (it isn't installed or imported by ipfx) --
its dependencies are ``import ipfx``, matplotlib (which ipfx already
requires), and PySide6. Copy it anywhere and run it directly.

Design: the trace/spike plot is drawn with matplotlib, embedded via
``FigureCanvasQTAgg``. Every interactive control -- buttons, the sweep
slider, the epoch list, the spike-detection parameter fields -- is a native
PySide6 widget, laid out with Qt layout managers. matplotlib's own widget
toolkit (``Button``/``Slider``/``RadioButtons``/``TextBox``) isn't used at
all here.

Why PySide6 instead of pure matplotlib widgets
-----------------------------------------------
An earlier version of this script used only matplotlib.widgets, deliberately
avoiding any GUI toolkit dependency. That ran into hard limits once files
with a realistic number of epochs were tried:

    * ``RadioButtons`` has no scrolling: labels are spaced evenly across a
      fixed axes rect regardless of how many there are, so past roughly a
      dozen epochs (the exact number depends on font size and window size)
      labels start overlapping and become unreadable -- there is no way to
      fit more without shrinking text past legibility.
    * Every matplotlib widget's screen position is a fixed *fraction* of the
      whole figure. Resizing the window grows every element -- and every
      margin -- by the same proportion; there's no way to keep, say, a
      fixed-height button row while letting only the epoch list grow.
    * matplotlib's ``TextBox``/``RadioButtons`` don't route keyboard focus
      the way a real GUI toolkit does, which needed manual workarounds (a
      guard so arrow-key handling didn't fight with a ``TextBox`` that was
      actively capturing keystrokes, and a monkeypatch for a matplotlib 3.11
      ``TextBox`` regression).

None of that is a problem for the widgets a real GUI toolkit provides:
``QListWidget`` scrolls natively at any size, Qt layouts handle "this grows,
that stays fixed" for free, and ``QLineEdit``/``QSlider`` get correct
keyboard-focus handling without any extra code. The plot itself is left as
matplotlib -- there's no reason to reimplement trace/spike plotting -- just
embedded in a Qt widget instead of its own top-level window.

Workflow:
    1. Click "Open NWB File..." to pick an NWB2 file via the native, cross-
       platform Qt file dialog, or paste a path into the text box next to it
       and press Enter instead.
    2. Drag the "Sweep" slider to one of the sweep numbers available in that
       file, or use the Left/Right arrow keys to step to the previous/next
       sweep from anywhere in the window (they move the text cursor instead
       while a path/parameter field has focus).
    3. Pick an epoch in the left "Epoch" list -- both the legacy,
       algorithmically detected epochs ("test"/"sweep"/"recording"/"stim"/
       "experiment") and, if the file was written by MIES, the
       "nwb:"-prefixed nwbEpochs (see Sweep.NWB_EPOCH_PREFIX and
       MIESNWBData.get_nwb_epochs) show up in the same list, however many
       there are. The second "Epoch" list, immediately to its right, offers
       the same choices for the same sweep -- pick a different epoch there
       to compare it against the first.
    4. Each chosen epoch's t/v/i slice is handed to SpikeFeatureExtractor
       automatically (re-run on every sweep/epoch change, or via the
       "Detect Spikes" button), and both traces plus their detected spike
       peaks/thresholds are (re-)plotted together on the same axes -- the
       first epoch in black/red/black (Vm/peak/threshold), the second epoch
       entirely in dark red (using triangle markers, up for peaks and down
       for thresholds, to stay distinguishable from the line itself).
    5. SpikeFeatureExtractor's own tunable arguments (filter, dv_cutoff,
       max_interval, min_height, min_peak, thresh_frac,
       reject_at_stim_start_interval -- everything except start/end, which
       come from the epoch selection instead) are editable text fields,
       pre-filled with SpikeFeatureExtractor's own defaults. Edit one and
       press Enter (or click "Detect Spikes") to re-run with the new value.
    6. Drag the splitter handle between the plot and the controls to resize
       either area; the built-in matplotlib toolbar (below the plot) offers
       pan/zoom/save.

Usage:
    python interactive_epoch_spike_viewer.py [path/to/file.nwb]

The path argument is optional -- if given, that file is loaded on startup;
otherwise use "Open NWB File..." or the text box once the window is open.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QFontDatabase, QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT
)
from matplotlib.figure import Figure

from ipfx.dataset.create import create_ephys_data_set
from ipfx.dataset.ephys_data_set import EphysDataSet
from ipfx.feature_extractor import SpikeFeatureExtractor
from ipfx.sweep import Sweep


class InteractiveEpochSpikeViewer(QMainWindow):
    """Pick a sweep + epoch from an NWB file and plot detected spikes.

    matplotlib is used only for the plot itself (embedded via
    ``FigureCanvasQTAgg``); every control is a native PySide6 widget.
    """

    # SpikeFeatureExtractor constructor arguments exposed as editable fields,
    # with their defaults taken directly from SpikeFeatureExtractor itself
    # (everything except start/end, which this viewer derives from the
    # selected epoch instead of asking for them separately).
    _PARAM_DEFAULTS = [
        ("filter", 10.0),
        ("dv_cutoff", 20.0),
        ("max_interval", 0.005),
        ("min_height", 2.0),
        ("min_peak", -30.0),
        ("thresh_frac", 0.05),
        ("reject_at_stim_start_interval", 0.0),
    ]
    _PARAM_GRID_COLS = 4

    def __init__(self, initial_path: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("Epoch Spike Viewer Example")
        self.resize(1100, 850)

        self.dataset: Optional[EphysDataSet] = None
        self.current_sweep: Optional[Sweep] = None
        self.sweep_numbers: List[int] = []
        self.param_edits: Dict[str, QLineEdit] = {}

        self._build_ui()

        # Global Left/Right arrow-key stepping (see eventFilter) needs to see
        # key presses regardless of which widget currently has focus.
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        if initial_path:
            self._load_dataset(initial_path)

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer_layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Vertical)
        outer_layout.addWidget(splitter)

        splitter.addWidget(self._build_plot_panel())
        splitter.addWidget(self._build_controls_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.statusBar().showMessage("Open an NWB file to begin.")

    def _build_plot_panel(self) -> QWidget:
        # "constrained" layout keeps the axes box sized to fill the figure
        # minus only what's actually needed for tick labels/axis labels/
        # legend/title -- unlike the default fixed-fraction margins (left
        # 12.5%, right 10%, etc.), which leave a wide blank strip on both
        # sides regardless of how wide the window actually is. This is
        # recomputed automatically on every resize, so it stays tight at
        # any window size rather than needing a fixed margin tuned once.
        self.fig = Figure(figsize=(9, 5), layout="constrained")
        self.ax_plot = self.fig.add_subplot(111)
        self.ax_plot.set_xlabel("Time (s)")
        self.ax_plot.set_ylabel("Membrane potential (mV)")

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        return panel

    def _build_controls_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        layout.addLayout(self._build_open_row())
        layout.addLayout(self._build_sweep_row())

        middle_row = QHBoxLayout()
        epoch_box_1, self.epoch_list_1 = self._build_epoch_box()
        epoch_box_2, self.epoch_list_2 = self._build_epoch_box()
        middle_row.addWidget(epoch_box_1, 1)
        middle_row.addWidget(epoch_box_2, 1)
        middle_row.addWidget(self._build_param_box(), 2)
        layout.addLayout(middle_row)

        # Three lines of text: start/end/length (in ms) of the epoch
        # selected in each of the two boxes above, plus their differences.
        # A fixed-width font is required for the numeric columns (and the
        # decimal points within them) to line up vertically across the
        # three lines -- a proportional font would not keep e.g. "1" and
        # "." the same width as "0" and "-", breaking the alignment.
        self.epoch_info_label = QLabel("No epoch selected yet.")
        self.epoch_info_label.setFont(
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        )
        layout.addWidget(self.epoch_info_label)

        layout.addLayout(self._build_detect_row())
        return panel

    def _build_open_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.open_btn = QPushButton("Open NWB File...")
        self.open_btn.clicked.connect(self._on_open_clicked)
        row.addWidget(self.open_btn)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(
            "Paste a path to an NWB file and press Enter..."
        )
        self.path_edit.returnPressed.connect(self._on_path_submitted)
        row.addWidget(self.path_edit, 1)
        return row

    def _build_sweep_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("Sweep:"))

        self.sweep_slider = QSlider(Qt.Orientation.Horizontal)
        self.sweep_slider.setEnabled(False)
        self.sweep_slider.valueChanged.connect(self._on_sweep_slider_changed)
        row.addWidget(self.sweep_slider, 1)

        self.sweep_label = QLabel("-")
        self.sweep_label.setMinimumWidth(40)
        row.addWidget(self.sweep_label)
        return row

    # Number of epoch rows that should be visible in each list box without
    # scrolling, on startup (before any file is loaded and the splitter/
    # group box could otherwise settle to a smaller natural size hint).
    _EPOCH_LIST_VISIBLE_ROWS = 6

    def _build_epoch_box(self) -> Tuple[QGroupBox, QListWidget]:
        """Build one "Epoch" list box. Two independent instances of this are
        created (side by side), each letting the user pick a different
        epoch of the same sweep -- the caller assigns the returned list
        widget to whichever of self.epoch_list_1 / self.epoch_list_2 it is.
        """
        box = QGroupBox("Epoch")
        layout = QVBoxLayout(box)

        # QListWidget scrolls natively -- unlike matplotlib's RadioButtons,
        # this has no fixed capacity before entries start overlapping.
        list_widget = QListWidget()
        list_widget.currentItemChanged.connect(self._on_epoch_changed)
        list_widget.setMinimumHeight(self._epoch_list_height_for_rows(
            list_widget, self._EPOCH_LIST_VISIBLE_ROWS
        ))
        layout.addWidget(list_widget)
        return box, list_widget

    @staticmethod
    def _epoch_list_height_for_rows(
        list_widget: QListWidget, n_rows: int
    ) -> int:
        """Pixel height for ``list_widget`` to show ``n_rows`` without
        scrolling, measured via a temporary dummy item so it accounts for
        the current style's actual per-item padding/margins (which can
        differ from a plain font-metrics estimate) rather than guessing.
        Called before the list is ever populated with real epoch names, so
        the dummy item is added and removed again immediately.
        """
        list_widget.addItem("dummy")
        row_height = list_widget.sizeHintForRow(0)
        list_widget.clear()

        frame = 2 * list_widget.frameWidth()
        return row_height * n_rows + frame

    def _build_param_box(self) -> QGroupBox:
        box = QGroupBox("Spike Detection Parameters (SpikeFeatureExtractor)")
        grid = QGridLayout(box)

        n_cols = self._PARAM_GRID_COLS
        # ceil(n_params / n_cols), x2 since each param takes a label row
        # plus a field row.
        n_content_rows = -(-len(self._PARAM_DEFAULTS) // n_cols) * 2

        for idx, (name, default) in enumerate(self._PARAM_DEFAULTS):
            row, col = divmod(idx, n_cols)
            grid.addWidget(QLabel(name), row * 2, col)

            edit = QLineEdit(str(default))
            edit.returnPressed.connect(self.analyze)
            grid.addWidget(edit, row * 2 + 1, col)
            self.param_edits[name] = edit

        # Without this, QGridLayout spreads any extra vertical space (from
        # the group box growing taller than its contents need) evenly
        # between all rows -- the label/field pairs drift further apart as
        # the window is resized. Pinning every content row's stretch to 0
        # and dumping all the stretch into one empty row below them keeps
        # the controls fixed at the top; the box just grows extra blank
        # space underneath instead.
        for row in range(n_content_rows):
            grid.setRowStretch(row, 0)
        grid.setRowStretch(n_content_rows, 1)

        return box

    def _build_detect_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.detect_btn = QPushButton("Detect Spikes")
        self.detect_btn.clicked.connect(self.analyze)
        row.addWidget(self.detect_btn)
        row.addStretch(1)
        return row

    def _read_extractor_kwargs(self) -> Optional[Dict[str, float]]:
        """Parse the parameter grid's current text into float kwargs for
        SpikeFeatureExtractor. Returns None (after setting an explanatory
        status message) if any field doesn't parse as a number.
        """
        kwargs: Dict[str, float] = {}
        for name, edit in self.param_edits.items():
            text = edit.text().strip()
            try:
                kwargs[name] = float(text)
            except ValueError:
                self._set_status(
                    f"Invalid value for {name!r}: {text!r} (must be a number)"
                )
                return None
        return kwargs

    # -- data loading --------------------------------------------------------

    def _load_dataset(self, path: str) -> None:
        try:
            self.dataset = create_ephys_data_set(nwb_file=path)
        except Exception as exc:
            self._set_status(f"Failed to load {path!r}: {exc}")
            return

        # Reflect the path that was actually loaded in the path box -- keeps
        # it populated when a path was supplied on the command line (which
        # doesn't otherwise touch this field), and is a harmless no-op when
        # it's already there (Open button / Enter in this same box).
        self.path_edit.setText(path)

        self.sweep_numbers = sorted(
            self.dataset.sweep_table[EphysDataSet.SWEEP_NUMBER].tolist()
        )
        if not self.sweep_numbers:
            self._set_status(f"Loaded {path!r} but it has no sweeps.")
            return

        self.sweep_slider.blockSignals(True)
        self.sweep_slider.setMinimum(0)
        self.sweep_slider.setMaximum(len(self.sweep_numbers) - 1)
        self.sweep_slider.setValue(0)
        self.sweep_slider.setEnabled(True)
        self.sweep_slider.blockSignals(False)

        self._set_status(
            f"Loaded {len(self.sweep_numbers)} sweep(s) from "
            f"{os.path.basename(path)}"
        )
        self._select_sweep(self.sweep_numbers[0])

    def _select_sweep(self, sweep_number: int) -> None:
        # Remember each box's currently selected epoch (if any) so it can be
        # carried over to the new sweep instead of always resetting to
        # "recording" -- captured before the lists are repopulated. The two
        # boxes keep independent memories of their own last selection.
        previous_epoch_1 = self._current_list_text(self.epoch_list_1)
        previous_epoch_2 = self._current_list_text(self.epoch_list_2)

        # _select_sweep is only reachable once _load_dataset has already
        # assigned self.dataset -- asserted (rather than silently trusted)
        # so the type checker can verify the .sweep() call below.
        assert self.dataset is not None, \
            "_select_sweep called before a dataset was loaded"
        try:
            self.current_sweep = self.dataset.sweep(sweep_number)
        except Exception as exc:
            self._set_status(f"Failed to load sweep {sweep_number}: {exc}")
            self.current_sweep = None
            return

        self.sweep_label.setText(str(sweep_number))

        # Legacy, algorithmically detected epochs plus any MIES nwbEpochs
        # (already "nwb:"-prefixed by MIESNWBData.get_nwb_epochs) available
        # for this specific sweep -- both boxes offer the identical choices.
        epoch_names = list(self.current_sweep.epochs.keys())
        epoch_names += [
            record["name"]
            for record in self.current_sweep.nwb_epochs
            if record["name"]
        ]
        self._populate_epoch_list(
            self.epoch_list_1, epoch_names, preferred=previous_epoch_1
        )
        self._populate_epoch_list(
            self.epoch_list_2, epoch_names, preferred=previous_epoch_2
        )

        if epoch_names:
            self.analyze()
        else:
            self._set_status(f"Sweep {sweep_number} has no epochs available.")

    @staticmethod
    def _current_list_text(list_widget: QListWidget) -> Optional[str]:
        item = list_widget.currentItem()
        return item.text() if item is not None else None

    def _populate_epoch_list(
        self,
        list_widget: QListWidget,
        epoch_names: List[str],
        preferred: Optional[str] = None,
    ) -> None:
        list_widget.blockSignals(True)
        list_widget.clear()
        list_widget.addItems(epoch_names)

        if epoch_names:
            if preferred and preferred in epoch_names:
                default = preferred
            elif "recording" in epoch_names:
                default = "recording"
            else:
                default = epoch_names[0]
            list_widget.setCurrentRow(epoch_names.index(default))

        list_widget.blockSignals(False)

    # -- callbacks ----------------------------------------------------------

    def _on_open_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open NWB file", "", "NWB files (*.nwb);;All files (*.*)"
        )
        if path:
            self.path_edit.setText(path)
            self._load_dataset(path)

    def _on_path_submitted(self) -> None:
        text = self.path_edit.text().strip()
        if text:
            self._load_dataset(text)

    def _on_sweep_slider_changed(self, index: int) -> None:
        if not self.sweep_numbers:
            return
        self._select_sweep(self.sweep_numbers[index])

    def _on_epoch_changed(
        self,
        current: Optional[QListWidgetItem],
        previous: Optional[QListWidgetItem],
    ) -> None:
        if current is not None:
            self.analyze()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Left/Right arrow keys step the sweep slider to the previous/next
        available sweep, from anywhere in the window.

        Installed on the QApplication instance (rather than relying on
        whichever widget happens to have focus) so it works the same way
        regardless of which control was last clicked -- except while a
        QLineEdit (path box or a parameter field) has focus, where those
        keys need to keep moving the text cursor instead.
        """
        if (
            isinstance(event, QKeyEvent)
            and event.type() == QEvent.Type.KeyPress
            and event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right)
        ):
            if isinstance(QApplication.focusWidget(), QLineEdit):
                return False
            if not self.sweep_numbers:
                return False

            current = self.sweep_slider.value()
            if event.key() == Qt.Key.Key_Right:
                new_value = min(current + 1, len(self.sweep_numbers) - 1)
            else:
                new_value = max(current - 1, 0)

            if new_value != current:
                self.sweep_slider.setValue(new_value)
            return True

        return super().eventFilter(obj, event)

    # -- analysis -------------------------------------------------------------

    def _detect_spikes_for_epoch(
        self, epoch_name: str, extractor_kwargs: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """Select ``epoch_name`` on ``self.current_sweep`` and run
        SpikeFeatureExtractor on its t/v/i slice.

        Both epoch boxes read from the same underlying Sweep object, so
        this is called once per box, sequentially -- each call re-slices
        the sweep to a (possibly different) epoch and immediately returns
        its own arrays, before the next call re-slices it again.

        Parameters
        ----------
        epoch_name
            Name of a legacy epoch or nwbEpoch of ``self.current_sweep``.
        extractor_kwargs
            Keyword arguments forwarded to ``SpikeFeatureExtractor``.

        Returns
        -------
        t, v, i
            The epoch's sliced time/voltage/current arrays.
        spikes
            SpikeFeatureExtractor's per-spike feature table.

        Raises
        ------
        KeyError
            If ``epoch_name`` isn't a valid epoch for this sweep (raised by
            ``Sweep.select_epoch``).
        Exception
            Propagated as-is from SpikeFeatureExtractor.
        """
        # Only called from analyze(), which already returned early if
        # self.current_sweep was None -- asserted so the type checker can
        # verify the attribute accesses below.
        assert self.current_sweep is not None, \
            "_detect_spikes_for_epoch called with no current sweep"
        self.current_sweep.select_epoch(epoch_name)
        t = self.current_sweep.t
        v = self.current_sweep.v
        i = self.current_sweep.i
        ext = SpikeFeatureExtractor(**extractor_kwargs)
        spikes = ext.process(t=t, v=v, i=i)
        return t, v, i, spikes

    # Total character width (including the sign/space column) of each of the
    # three number columns below -- fixed so the decimal point always falls
    # at the same column regardless of how many integer digits a value has
    # (the fractional part + '.' is always the last 4 characters of the
    # field), which is what keeps it aligned across all three lines.
    _EPOCH_INFO_NUM_WIDTH = 10
    _EPOCH_INFO_LABEL_WIDTH = 14

    @staticmethod
    def _epoch_time_bounds_ms(
        t_array: Optional[np.ndarray]
    ) -> Optional[Tuple[float, float, float]]:
        """(start, end, length) in ms from an epoch-sliced time array --
        t_array[0]/t_array[-1] are exactly that epoch's start/end time,
        consistent with what's plotted. None if there's no such epoch.
        """
        if t_array is None or len(t_array) == 0:
            return None
        start_ms = t_array[0] * 1e3
        end_ms = t_array[-1] * 1e3
        return start_ms, end_ms, end_ms - start_ms

    def _format_epoch_info_row(
        self,
        label: str,
        bounds_ms: Optional[Tuple[float, float, float]],
        signed: bool,
    ) -> str:
        """One line: a left-justified label followed by start/end/length,
        each right-justified to the same fixed width. ``signed`` selects
        between reserving a blank sign column (plain epoch bounds, always
        >= 0) and always showing +/- (the difference row, which can go
        either way) -- both occupy the same number of characters, so the
        decimal points still line up between a plain row and the signed row.
        """
        prefix = f"{label:<{self._EPOCH_INFO_LABEL_WIDTH}}"
        if bounds_ms is None:
            return prefix + "n/a"

        start_ms, end_ms, length_ms = bounds_ms
        num_spec = f"{self._EPOCH_INFO_NUM_WIDTH}.3f"
        sign_flag = "+" if signed else " "
        return (
            f"{prefix}"
            f"start={start_ms:{sign_flag}{num_spec}} ms   "
            f"end={end_ms:{sign_flag}{num_spec}} ms   "
            f"length={length_ms:{sign_flag}{num_spec}} ms"
        )

    def _format_epoch_info_lines(
        self,
        epoch_name_1: str,
        t1: np.ndarray,
        epoch_name_2: Optional[str],
        t2: Optional[np.ndarray],
    ) -> str:
        bounds1 = self._epoch_time_bounds_ms(t1)
        bounds2 = self._epoch_time_bounds_ms(t2) if t2 is not None else None

        line1 = self._format_epoch_info_row(
            f"{epoch_name_1}:", bounds1, signed=False
        )
        line2 = self._format_epoch_info_row(
            f"{epoch_name_2}:" if epoch_name_2 is not None else "Epoch 2:",
            bounds2, signed=False
        )

        diff_bounds: Optional[Tuple[float, float, float]] = None
        if bounds1 is not None and bounds2 is not None:
            diff_bounds = (
                bounds2[0] - bounds1[0],
                bounds2[1] - bounds1[1],
                bounds2[2] - bounds1[2],
            )
        line3 = self._format_epoch_info_row(
            "Diff (2-1):", diff_bounds, signed=True
        )

        return "\n".join([line1, line2, line3])

    def analyze(self) -> None:
        if self.current_sweep is None:
            return

        item1 = self.epoch_list_1.currentItem()
        if item1 is None:
            return
        epoch_name_1 = item1.text()

        item2 = self.epoch_list_2.currentItem()
        epoch_name_2 = item2.text() if item2 is not None else None

        extractor_kwargs = self._read_extractor_kwargs()
        if extractor_kwargs is None:
            return

        try:
            t1, v1, _i1, spikes1 = self._detect_spikes_for_epoch(
                epoch_name_1, extractor_kwargs
            )
        except KeyError as exc:
            self._set_status(str(exc))
            return
        except Exception as exc:
            self._set_status(f"Spike detection failed: {exc}")
            return

        # The second epoch selection is independent of the first -- if it
        # fails for any reason, still show the first epoch's trace rather
        # than blanking the whole plot. Bundled into one Optional tuple
        # (rather than three separately-Optional variables) so checking it
        # once narrows t2/v2/spikes2 together for the type checker, since
        # they're only ever meaningful as a matched set.
        epoch2_result: Optional[Tuple[np.ndarray, np.ndarray, pd.DataFrame]]
        epoch2_result = None
        error2: Optional[str] = None
        if epoch_name_2 is not None:
            try:
                t2, v2, _i2, spikes2 = self._detect_spikes_for_epoch(
                    epoch_name_2, extractor_kwargs
                )
                epoch2_result = (t2, v2, spikes2)
            except KeyError as exc:
                error2 = str(exc)
            except Exception as exc:
                error2 = f"spike detection failed: {exc}"
            finally:
                # Leave current_sweep's selection matching the first epoch,
                # since that's what the rest of the UI (and re-entrant
                # analyze() calls) treats as "the" current selection.
                try:
                    self.current_sweep.select_epoch(epoch_name_1)
                except Exception:
                    pass

        self.epoch_info_label.setText(
            self._format_epoch_info_lines(
                epoch_name_1, t1, epoch_name_2,
                epoch2_result[0] if epoch2_result is not None else None
            )
        )

        sweep_number = self.sweep_numbers[self.sweep_slider.value()]

        # SpikeFeatureExtractor works in seconds; only convert to
        # milliseconds right at the point of plotting.
        s_to_ms = 1e3

        self.ax_plot.clear()
        self.ax_plot.plot(
            t1 * s_to_ms, v1, color="black", linewidth=0.8,
            label=f"Vm ({epoch_name_1})"
        )
        if len(spikes1):
            self.ax_plot.plot(
                spikes1["peak_t"] * s_to_ms, spikes1["peak_v"], "r.",
                label=f"peak ({epoch_name_1})"
            )
            self.ax_plot.plot(
                spikes1["threshold_t"] * s_to_ms, spikes1["threshold_v"], "k.",
                label=f"threshold ({epoch_name_1})"
            )

        if epoch2_result is not None:
            t2, v2, spikes2 = epoch2_result
            self.ax_plot.plot(
                t2 * s_to_ms, v2, color="darkred", linewidth=1.2,
                label=f"Vm ({epoch_name_2})"
            )
            if len(spikes2):
                self.ax_plot.plot(
                    spikes2["peak_t"] * s_to_ms, spikes2["peak_v"],
                    color="darkred", marker="^", linestyle="None",
                    label=f"peak ({epoch_name_2})"
                )
                self.ax_plot.plot(
                    spikes2["threshold_t"] * s_to_ms, spikes2["threshold_v"],
                    color="darkred", marker="v", linestyle="None",
                    label=f"threshold ({epoch_name_2})"
                )

        self.ax_plot.set_xlabel("Time (ms)")
        self.ax_plot.set_ylabel("Membrane potential (mV)")
        # Default data margins pad 5% of the x-range on both sides beyond
        # the actual trace -- on top of the figure-level margin, that
        # doubled up on blank space at the left/right edges of the plot.
        self.ax_plot.margins(x=0.01)

        title = (
            f"Sweep {sweep_number} — epoch {epoch_name_1!r} "
            f"({len(spikes1)} spike(s))"
        )
        if epoch2_result is not None:
            n_spikes2 = len(epoch2_result[2])
            title += f"  |  epoch {epoch_name_2!r} ({n_spikes2} spike(s))"
        self.ax_plot.set_title(title)
        self.ax_plot.legend(loc="upper right", fontsize=8)
        self.canvas.draw_idle()

        status = (
            f"Sweep {sweep_number}, epoch {epoch_name_1!r}: "
            f"{len(spikes1)} spike(s) detected"
        )
        if epoch2_result is not None:
            n_spikes2 = len(epoch2_result[2])
            status += (
                f"; epoch {epoch_name_2!r}: {n_spikes2} spike(s) detected"
            )
        if error2:
            status += f" (2nd epoch: {error2})"
        self._set_status(status)

    # -- misc -------------------------------------------------------------

    def _set_status(self, message: str) -> None:
        self.statusBar().showMessage(message)


def main() -> None:
    app = QApplication(sys.argv)
    initial_path = sys.argv[1] if len(sys.argv) > 1 else None
    window = InteractiveEpochSpikeViewer(initial_path=initial_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
