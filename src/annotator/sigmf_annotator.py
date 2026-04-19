"""
SigMF Waterfall Annotator
Requires: pip install PyQt6 numpy scipy

Architecture
============
BaseDetector          – subclass this to add a new energy-detection algorithm
DetectorParam         – declares one tunable hyperparameter (drives the GUI)
DetectionContext      – passed to BaseDetector.run(); contains IQ + metadata
DetectionResult       – one annotation returned by a detector
CommandStack          – undo/redo for all annotation mutations
SigMFDataset          – loads any valid SigMF datatype, converts to cf32 on read
TileWorker/Thread     – background FFT rendering
WaterfallCanvas       – scrollable waterfall with annotation editing
DetectorPanel         – dynamic GUI for whichever detector is selected
MainWindow            – assembles everything
"""

from __future__ import annotations
import sys, re, json, math, copy, time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QScrollArea, QMessageBox,
    QInputDialog, QStatusBar, QSplitter, QSizePolicy, QToolBar,
    QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QListWidgetItem,
    QAbstractItemView, QMenu, QProgressBar, QTabWidget, QFormLayout,
    QLineEdit, QGroupBox, QCheckBox, QFrame, QDockWidget, QSizePolicy
)
from PyQt6.QtCore import (
    Qt, QRect, QPoint, QPointF, QRectF, QThread, pyqtSignal, QObject, QSize
)
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont, QPixmap, QImage,
    QCursor, QPalette, QKeySequence, QAction, QShortcut
)

# ═══════════════════════════════════════════════════════════════════════════════
#  DETECTOR PLUGIN INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectorParam:
    """
    Declares one hyperparameter for a detector.

    Attributes
    ----------
    name        Internal key used when passing kwargs to run().
    label       Human-readable label shown in the GUI.
    kind        One of: 'float', 'int', 'str', 'bool', 'choice'
    default     Default value (must match kind).
    minimum     For float/int spinboxes.
    maximum     For float/int spinboxes.
    step        For float/int spinboxes.
    decimals    For float spinboxes.
    choices     List of strings when kind='choice'.
    tooltip     Optional tooltip shown in the GUI.
    """
    name:     str
    label:    str
    kind:     str           # 'float' | 'int' | 'str' | 'bool' | 'choice'
    default:  Any
    minimum:  float = 0.0
    maximum:  float = 1e9
    step:     float = 1.0
    decimals: int   = 3
    choices:  list[str] = field(default_factory=list)
    tooltip:  str   = ""


@dataclass
class DetectionContext:
    """
    Everything a detector could need.

    Attributes
    ----------
    samples         complex64 ndarray for the region being analysed.
    sample_rate     Hz.
    center_freq     Hz.
    sample_start    Absolute sample index where `samples` begins in the file.
    fft_size        FFT size currently used by the viewer.
    sigmf_meta      Full parsed metadata dict (read-only reference).
    """
    samples:      np.ndarray
    sample_rate:  float
    center_freq:  float
    sample_start: int
    fft_size:     int
    sigmf_meta:   dict


@dataclass
class DetectionResult:
    """
    One annotation bounding box returned by a detector.

    All coordinates are in SigMF convention:
      sample_start  – absolute sample index
      sample_count  – number of samples
      freq_lower    – Hz (absolute)
      freq_upper    – Hz (absolute)
      label         – annotation label string
    """
    sample_start:  int
    sample_count:  int
    freq_lower:    float
    freq_upper:    float
    label:         str = ""


class BaseDetector(ABC):
    """
    Subclass this to add a new detector.

    Class attributes
    ----------------
    NAME    Display name shown in the GUI dropdown.
    PARAMS  List of DetectorParam objects that drive the parameter panel.

    The GUI reads PARAMS once at startup to build widgets.  When the user
    clicks Run, the current widget values are collected into a dict and
    passed as **kwargs to run().
    """
    NAME:   str = "Unnamed Detector"
    PARAMS: list[DetectorParam] = []

    @abstractmethod
    def run(self, ctx: DetectionContext, **kwargs) -> list[DetectionResult]:
        """
        Analyse ctx.samples and return a list of DetectionResult objects.
        kwargs contains one entry per DetectorParam, keyed by param.name,
        already cast to the correct Python type.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════════
#  BUILT-IN DETECTORS
#  Add new ones below — they are auto-discovered via BaseDetector.__subclasses__
# ═══════════════════════════════════════════════════════════════════════════════

class EnergyThresholdDetector(BaseDetector):
    """
    Simple per-FFT-row energy threshold.
    Scans row-by-row; groups consecutive rows that exceed the threshold into
    one annotation, then finds the frequency extent from the active bins.
    """
    NAME = "Energy Threshold"
    PARAMS = [
        DetectorParam("label",          "Annotation Label",   "str",   "signal"),
        DetectorParam("threshold_db",   "Threshold (dB)",     "float", -45.0,
                      minimum=-120.0, maximum=0.0, step=1.0, decimals=1,
                      tooltip="Rows with mean power above this are marked as signal."),
        DetectorParam("min_duration",   "Min Duration (rows)","int",   3,
                      minimum=1, maximum=10000, step=1,
                      tooltip="Discard detections shorter than this many FFT rows."),
        DetectorParam("freq_percentile","Freq Percentile",    "float", 5.0,
                      minimum=0.0, maximum=49.9, step=0.5, decimals=1,
                      tooltip="Percentile used to trim noise from freq extent estimate."),
        DetectorParam("merge_gap",      "Merge Gap (rows)",   "int",   2,
                      minimum=0, maximum=1000, step=1,
                      tooltip="Merge detections separated by fewer rows than this."),
    ]

    def run(self, ctx: DetectionContext, **kw) -> list[DetectionResult]:
        samples   = ctx.samples
        fs        = ctx.fft_size
        sr        = ctx.sample_rate
        cf        = ctx.center_freq
        ss_base   = ctx.sample_start
        thr_db    = float(kw["threshold_db"])
        min_dur   = int(kw["min_duration"])
        freq_pct  = float(kw["freq_percentile"])
        merge_gap = int(kw["merge_gap"])
        label     = str(kw["label"])

        if samples.size < fs:
            return []

        pad = (-len(samples)) % fs
        if pad:
            samples = np.concatenate([samples, np.zeros(pad, dtype=np.complex64)])
        rows = len(samples) // fs
        mat  = samples.reshape(rows, fs)

        win  = np.hanning(fs).astype(np.float32)
        wg   = np.sum(win**2) / fs
        spec = np.fft.fftshift(np.fft.fft(mat * win, axis=1), axes=1)
        pwr  = 10 * np.log10(np.abs(spec / fs)**2 / wg + 1e-12)   # (rows, fs)
        row_energy = pwr.mean(axis=1)

        # Find contiguous above-threshold runs
        hot = row_energy >= thr_db
        runs: list[tuple[int,int]] = []   # (start_row, end_row) inclusive
        i = 0
        while i < rows:
            if hot[i]:
                j = i
                while j < rows and hot[j]:
                    j += 1
                runs.append((i, j - 1))
                i = j
            else:
                i += 1

        # Merge close runs
        if merge_gap > 0 and len(runs) > 1:
            merged = [runs[0]]
            for (rs, re) in runs[1:]:
                if rs - merged[-1][1] <= merge_gap:
                    merged[-1] = (merged[-1][0], re)
                else:
                    merged.append((rs, re))
            runs = merged

        results = []
        for (rs, re) in runs:
            if (re - rs + 1) < min_dur:
                continue
            # Frequency extent from the brightest bins in this run
            run_pwr = pwr[rs:re+1, :]          # (run_rows, fs)
            col_max = run_pwr.max(axis=0)      # per-bin peak
            lo_pct  = np.percentile(col_max, freq_pct)
            active  = np.where(col_max >= lo_pct)[0]
            if active.size == 0:
                continue
            bin_lo  = int(active[0]);  bin_hi = int(active[-1])
            # bin index → frequency  (fftshift: bin 0 = CF - SR/2)
            freq_lo = cf - sr/2 + bin_lo * (sr / fs)
            freq_hi = cf - sr/2 + (bin_hi + 1) * (sr / fs)
            abs_ss  = ss_base + rs * fs
            abs_sc  = (re - rs + 1) * fs
            results.append(DetectionResult(abs_ss, abs_sc, freq_lo, freq_hi, label))

        return results


class SpectralVarianceDetector(BaseDetector):
    """
    Detects signals by looking for columns (frequency bins) whose power
    variance over time exceeds a threshold — works well for pulsed / bursty
    signals that stand out against a stable noise floor.
    """
    NAME = "Spectral Variance"
    PARAMS = [
        DetectorParam("label",        "Annotation Label",     "str",   "burst"),
        DetectorParam("var_threshold","Variance Threshold dB²","float", 2.0,
                      minimum=0.01, maximum=1000.0, step=0.5, decimals=2,
                      tooltip="Bins with power variance above this are considered active."),
        DetectorParam("min_bw_bins",  "Min BW (bins)",        "int",   4,
                      minimum=1, maximum=4096, step=1,
                      tooltip="Ignore detections narrower than this many bins."),
        DetectorParam("chunk_rows",   "Analysis Chunk (rows)","int",   256,
                      minimum=16, maximum=4096, step=16,
                      tooltip="Number of FFT rows analysed per detection window."),
        DetectorParam("overlap_rows", "Chunk Overlap (rows)", "int",   32,
                      minimum=0, maximum=512, step=8),
    ]

    def run(self, ctx: DetectionContext, **kw) -> list[DetectionResult]:
        samples     = ctx.samples
        fs          = ctx.fft_size
        sr          = ctx.sample_rate
        cf          = ctx.center_freq
        ss_base     = ctx.sample_start
        var_thr     = float(kw["var_threshold"])
        min_bw      = int(kw["min_bw_bins"])
        chunk_rows  = int(kw["chunk_rows"])
        overlap     = int(kw["overlap_rows"])
        label       = str(kw["label"])

        if samples.size < fs:
            return []

        pad = (-len(samples)) % fs
        if pad:
            samples = np.concatenate([samples, np.zeros(pad, dtype=np.complex64)])
        total_rows = len(samples) // fs
        mat  = samples.reshape(total_rows, fs)
        win  = np.hanning(fs).astype(np.float32)
        wg   = np.sum(win**2) / fs
        spec = np.fft.fftshift(np.fft.fft(mat * win, axis=1), axes=1)
        pwr  = 10 * np.log10(np.abs(spec / fs)**2 / wg + 1e-12)

        results = []
        step    = max(1, chunk_rows - overlap)
        rs = 0
        while rs < total_rows:
            re      = min(rs + chunk_rows, total_rows)
            chunk   = pwr[rs:re, :]
            var     = chunk.var(axis=0)
            active  = np.where(var >= var_thr)[0]
            if active.size >= min_bw:
                # Find contiguous active-bin groups
                groups = []
                gi = 0
                while gi < len(active):
                    gj = gi
                    while gj + 1 < len(active) and active[gj+1] == active[gj] + 1:
                        gj += 1
                    if (gj - gi + 1) >= min_bw:
                        groups.append((int(active[gi]), int(active[gj])))
                    gi = gj + 1
                for b_lo, b_hi in groups:
                    freq_lo = cf - sr/2 + b_lo * (sr / fs)
                    freq_hi = cf - sr/2 + (b_hi + 1) * (sr / fs)
                    abs_ss  = ss_base + rs * fs
                    abs_sc  = (re - rs) * fs
                    results.append(DetectionResult(abs_ss, abs_sc,
                                                   freq_lo, freq_hi, label))
            rs += step

        return results


# ─── Registry: all BaseDetector subclasses are auto-collected ────────────────
def get_all_detectors() -> list[type[BaseDetector]]:
    return BaseDetector.__subclasses__()


# ═══════════════════════════════════════════════════════════════════════════════
#  UNDO / REDO  (Command pattern)
# ═══════════════════════════════════════════════════════════════════════════════

class Command(ABC):
    @abstractmethod
    def execute(self):  ...
    @abstractmethod
    def undo(self):     ...
    description: str = ""


class CommandStack:
    def __init__(self, max_depth: int = 200):
        self._undo: list[Command] = []
        self._redo: list[Command] = []
        self._max  = max_depth
        self.changed = None    # callable injected by owner

    def push(self, cmd: Command):
        cmd.execute()
        self._undo.append(cmd)
        if len(self._undo) > self._max:
            self._undo.pop(0)
        self._redo.clear()
        self._notify()

    def undo(self):
        if not self._undo:
            return
        cmd = self._undo.pop()
        cmd.undo()
        self._redo.append(cmd)
        self._notify()

    def redo(self):
        if not self._redo:
            return
        cmd = self._redo.pop()
        cmd.execute()
        self._undo.append(cmd)
        self._notify()

    def clear(self):
        self._undo.clear(); self._redo.clear(); self._notify()

    @property
    def can_undo(self): return bool(self._undo)
    @property
    def can_redo(self): return bool(self._redo)
    @property
    def undo_label(self): return self._undo[-1].description if self._undo else ""
    @property
    def redo_label(self): return self._redo[-1].description if self._redo else ""

    def _notify(self):
        if callable(self.changed):
            self.changed()


# ── Concrete commands ─────────────────────────────────────────────────────────

class AddAnnotationCmd(Command):
    def __init__(self, dataset, ann: dict):
        self.ds  = dataset
        self.ann = ann
        self.description = f"Add annotation '{ann.get('core:label','')!r}'"

    def execute(self):
        self.ds.annotations.append(self.ann)

    def undo(self):
        if self.ann in self.ds.annotations:
            self.ds.annotations.remove(self.ann)


class DeleteAnnotationCmd(Command):
    def __init__(self, dataset, index: int):
        self.ds    = dataset
        self.index = index
        self.ann   = copy.deepcopy(dataset.annotations[index])
        self.description = f"Delete annotation '{self.ann.get('core:label','')!r}'"

    def execute(self):
        # find by identity in case indices shifted
        try:
            self.ds.annotations.pop(self.index)
        except IndexError:
            pass

    def undo(self):
        self.ds.annotations.insert(self.index, copy.deepcopy(self.ann))


class EditAnnotationCmd(Command):
    def __init__(self, dataset, index: int, new_values: dict):
        self.ds    = dataset
        self.index = index
        self.new   = new_values
        self.old   = {k: dataset.annotations[index][k]
                      for k in new_values if k in dataset.annotations[index]}
        self.description = "Edit annotation"

    def execute(self):
        self.ds.annotations[self.index].update(self.new)

    def undo(self):
        self.ds.annotations[self.index].update(self.old)
        # remove keys that didn't exist before
        for k in self.new:
            if k not in self.old:
                self.ds.annotations[self.index].pop(k, None)


class BulkAddAnnotationsCmd(Command):
    def __init__(self, dataset, anns: list[dict]):
        self.ds   = dataset
        self.anns = anns
        self.description = f"Detect: add {len(anns)} annotation(s)"

    def execute(self):
        self.ds.annotations.extend(self.anns)

    def undo(self):
        for a in self.anns:
            if a in self.ds.annotations:
                self.ds.annotations.remove(a)


class BulkDeleteAnnotationsCmd(Command):
    def __init__(self, dataset, indices: list[int]):
        self.ds      = dataset
        self.saved   = [(i, copy.deepcopy(dataset.annotations[i])) for i in sorted(indices)]
        self.description = f"Delete {len(indices)} annotation(s)"

    def execute(self):
        for i, _ in sorted(self.saved, reverse=True):
            try: self.ds.annotations.pop(i)
            except IndexError: pass

    def undo(self):
        for i, ann in self.saved:
            self.ds.annotations.insert(i, copy.deepcopy(ann))


# ═══════════════════════════════════════════════════════════════════════════════
#  SIGMF DATASET  (all datatypes → cf32 on read)
# ═══════════════════════════════════════════════════════════════════════════════

class SigMFDataset:
    _SCALAR: dict[tuple[bool, bool, int], str] = {
        (True,  False,  8): "u1", (False, False,  8): "i1",
        (True,  False, 16): "u2", (False, False, 16): "i2",
        (True,  False, 32): "u4", (False, False, 32): "i4",
        (True,  False, 64): "u8", (False, False, 64): "i8",
        (False, True,  32): "f4", (False, True,  64): "f8",
    }

    def __init__(self, meta_path: str):
        self.meta_path = Path(meta_path)
        self.data_path = self.meta_path.with_suffix(".sigmf-data")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Meta not found: {self.meta_path}")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found: {self.data_path}")
        with open(self.meta_path) as f:
            self.meta = json.load(f)
        g    = self.meta.get("global", {})
        dt   = g.get("core:datatype", "").lower().strip()
        caps = self.meta.get("captures", [{}])
        self.sample_rate  = float(g.get("core:sample_rate", 1.0))
        self.center_freq  = float(caps[0].get("core:frequency",
                                  g.get("core:frequency", 0.0)))
        self.annotations  = self.meta.get("annotations", [])
        self._is_complex, self._numpy_dtype, self._bps = self._parse_datatype(dt)
        self.total_samples = self.data_path.stat().st_size // self._bps

    @classmethod
    def _parse_datatype(cls, dt: str):
        dt = dt.replace("complex64", "cf32_le").replace("complex128", "cf64_le")
        m  = re.fullmatch(r'(c|r)(f|i|u)(\d+)(?:_(le|be))?', dt)
        if not m:
            raise ValueError(f"Unrecognised SigMF datatype: '{dt}'")
        kind_cr, kind_fiu, bits_str, endian = m.groups()
        bits = int(bits_str)
        is_complex  = kind_cr == "c"
        is_float    = kind_fiu == "f"
        is_unsigned = kind_fiu == "u"
        key = (is_unsigned, is_float, bits)
        if key not in cls._SCALAR:
            raise ValueError(f"Unsupported component type: {kind_fiu}{bits}")
        base = np.dtype(cls._SCALAR[key])
        if bits > 8:
            base = base.newbyteorder('>' if endian == "be" else '<')
        bps = (bits // 8) * (2 if is_complex else 1)
        return is_complex, base, bps

    def read_samples(self, start: int, count: int) -> np.ndarray:
        start = max(0, min(start, self.total_samples - 1))
        count = min(count, self.total_samples - start)
        if count <= 0:
            return np.array([], dtype=np.complex64)
        with open(self.data_path, "rb") as f:
            f.seek(start * self._bps)
            raw = f.read(count * self._bps)
        dt  = self._numpy_dtype
        arr = np.frombuffer(raw, dtype=dt)
        if dt.byteorder == '>':
            arr = arr.byteswap().newbyteorder()
        return self._to_cf32(arr)

    def _to_cf32(self, arr: np.ndarray) -> np.ndarray:
        dt = self._numpy_dtype
        is_float    = dt.kind == 'f'
        is_unsigned = dt.kind == 'u'
        bits = dt.itemsize * 8
        if self._is_complex:
            if arr.size % 2:
                arr = arr[:-(arr.size % 2)]
            iq = arr.reshape(-1, 2).astype(np.float32)
        else:
            iq = np.stack([arr.astype(np.float32),
                           np.zeros(len(arr), np.float32)], axis=1)
        if not is_float:
            scale = float(1 << (bits - 1))
            iq = (iq - (scale if is_unsigned else 0)) / scale
        return (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)

    def save(self):
        self.meta["annotations"] = self.annotations
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  COLORMAPS
# ═══════════════════════════════════════════════════════════════════════════════

COLORMAPS: dict[str, np.ndarray] = {}

def _build_lut(name: str) -> np.ndarray:
    t = np.linspace(0, 1, 256)
    if name == "Viridis":
        r = np.interp(t,[0,.25,.5,.75,1],[68,59,33,94,253])
        g = np.interp(t,[0,.25,.5,.75,1],[1,82,145,201,231])
        b = np.interp(t,[0,.25,.5,.75,1],[84,139,140,98,37])
    elif name == "Inferno":
        r = np.interp(t,[0,.25,.5,.75,1],[0,87,188,249,252])
        g = np.interp(t,[0,.25,.5,.75,1],[0,16,55,142,255])
        b = np.interp(t,[0,.25,.5,.75,1],[4,110,85,8,164])
    elif name == "Plasma":
        r = np.interp(t,[0,.25,.5,.75,1],[13,84,163,229,240])
        g = np.interp(t,[0,.25,.5,.75,1],[8,2,44,126,249])
        b = np.interp(t,[0,.25,.5,.75,1],[135,163,164,56,33])
    elif name == "Jet":
        r = np.interp(t,[0,.35,.66,.89,1],[0,0,255,255,128])
        g = np.interp(t,[0,.12,.38,.64,.91,1],[0,0,255,255,0,0])
        b = np.interp(t,[0,.11,.34,.65,1],[128,255,255,0,0])
    else:
        r = g = b = t * 255
    return np.stack([r,g,b],axis=1).astype(np.uint8)

for _n in ["Viridis","Inferno","Plasma","Jet","Greys"]:
    COLORMAPS[_n] = _build_lut(_n)

TILE_ROWS   = 512
FFT_SIZE    = 1024
HANDLE_SIZE = 8
MIN_BOX_PX  = 6
ANN_COLOR   = QColor(255, 220, 0)
SEL_COLOR   = QColor(0, 220, 255)
DET_COLOR   = QColor(255, 100, 100)

# ═══════════════════════════════════════════════════════════════════════════════
#  TILE RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

class TileWorker(QObject):
    tile_ready = pyqtSignal(int, QImage)

    def __init__(self, ds, fft_size, cmap, vmin, vmax, row_start, row_count):
        super().__init__()
        self.ds=ds; self.fft_size=fft_size; self.cmap=cmap
        self.vmin=vmin; self.vmax=vmax
        self.row_start=row_start; self.row_count=row_count

    def run(self):
        fs = self.fft_size
        samples = self.ds.read_samples(self.row_start*fs, self.row_count*fs)
        if samples.size == 0: return
        pad = (-len(samples))%fs
        if pad: samples = np.concatenate([samples, np.zeros(pad,dtype=np.complex64)])
        rows = len(samples)//fs
        mat  = samples.reshape(rows, fs)
        win  = np.hanning(fs).astype(np.float32)
        wg   = np.sum(win**2)/fs
        spec = np.fft.fftshift(np.fft.fft(mat*win,axis=1),axes=1)
        pwr  = 10*np.log10(np.abs(spec/fs)**2/wg+1e-12)
        idx  = np.clip((pwr-self.vmin)/(self.vmax-self.vmin),0,1)
        idx  = (idx*255).astype(np.uint8)
        rgb  = COLORMAPS[self.cmap][idx]
        h,w  = rgb.shape[:2]
        img  = QImage(rgb.tobytes(), w, h, w*3, QImage.Format.Format_RGB888)
        self.tile_ready.emit(self.row_start, img.copy())

class TileThread(QThread):
    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.worker.moveToThread(self)
        self.started.connect(self.worker.run)

# ═══════════════════════════════════════════════════════════════════════════════
#  ANNOTATION ITEM  (pixel ↔ SigMF coordinate conversion)
# ═══════════════════════════════════════════════════════════════════════════════

class AnnotationItem:
    HANDLE_CURSORS = {
        "tl":Qt.CursorShape.SizeFDiagCursor,"br":Qt.CursorShape.SizeFDiagCursor,
        "tr":Qt.CursorShape.SizeBDiagCursor,"bl":Qt.CursorShape.SizeBDiagCursor,
        "tm":Qt.CursorShape.SizeVerCursor,  "bm":Qt.CursorShape.SizeVerCursor,
        "lm":Qt.CursorShape.SizeHorCursor,  "rm":Qt.CursorShape.SizeHorCursor,
    }

    def __init__(self, ann: dict, index: int, from_detector: bool = False):
        self.ann = ann; self.index = index; self.from_detector = from_detector

    def label(self) -> str: return self.ann.get("core:label","")

    @staticmethod
    def freq_to_x(freq,cf,sr,w): return ((freq-cf)/sr+0.5)*w
    @staticmethod
    def x_to_freq(x,cf,sr,w):   return cf+(x/w-0.5)*sr
    @staticmethod
    def row_to_y(row,rh):        return row*rh
    @staticmethod
    def y_to_row(y,rh):          return y/rh

    def to_rect(self, fft_size, row_h, cf, sr, cw) -> QRectF:
        a  = self.ann
        ss = a.get("core:sample_start",0)
        sc = a.get("core:sample_count",fft_size)
        fl = a.get("core:freq_lower_edge", cf-sr/2)
        fu = a.get("core:freq_upper_edge", cf+sr/2)
        yt = self.row_to_y(ss/fft_size, row_h)
        yb = self.row_to_y((ss+sc)/fft_size, row_h)
        xl = self.freq_to_x(fl,cf,sr,cw)
        xr = self.freq_to_x(fu,cf,sr,cw)
        return QRectF(xl,yt,xr-xl,yb-yt)

    def handles(self, r: QRectF) -> dict:
        cx,cy = r.center().x(),r.center().y()
        return {"tl":QPointF(r.left(),r.top()),   "tr":QPointF(r.right(),r.top()),
                "bl":QPointF(r.left(),r.bottom()),"br":QPointF(r.right(),r.bottom()),
                "tm":QPointF(cx,r.top()),          "bm":QPointF(cx,r.bottom()),
                "lm":QPointF(r.left(),cy),         "rm":QPointF(r.right(),cy)}

    def handle_rect(self, pt: QPointF) -> QRectF:
        s=HANDLE_SIZE; return QRectF(pt.x()-s/2,pt.y()-s/2,s,s)

    def hit_handle(self, pos: QPointF, r: QRectF) -> str | None:
        for n,pt in self.handles(r).items():
            if self.handle_rect(pt).contains(pos): return n
        return None

# ═══════════════════════════════════════════════════════════════════════════════
#  AXIS WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

class FreqAxisWidget(QWidget):
    HEIGHT = 36
    def __init__(self,p=None):
        super().__init__(p); self.setFixedHeight(self.HEIGHT)
        self.cf=0.0; self.sr=1.0

    def set_params(self,cf,sr): self.cf=cf; self.sr=sr; self.update()

    def paintEvent(self,_):
        p=QPainter(self); p.fillRect(self.rect(),QColor(25,25,35))
        p.setPen(QColor(200,200,200)); p.setFont(QFont("Courier",7))
        w=self.width(); n=9
        for i in range(n):
            frac=i/(n-1); freq=(self.cf-self.sr/2)+frac*self.sr; x=int(frac*w)
            lbl=(f"{freq/1e9:.3f}G" if abs(freq)>=1e9 else
                 f"{freq/1e6:.3f}M" if abs(freq)>=1e6 else
                 f"{freq/1e3:.1f}k" if abs(freq)>=1e3 else f"{freq:.0f}")
            p.drawLine(x,0,x,6)
            fm=p.fontMetrics(); tw=fm.horizontalAdvance(lbl)
            p.drawText(max(0,min(x-tw//2,w-tw)), self.HEIGHT-4, lbl)


class TimeAxisWidget(QWidget):
    WIDTH = 72
    def __init__(self,p=None):
        super().__init__(p); self.setFixedWidth(self.WIDTH)
        self.sr=1.0; self.fft_size=FFT_SIZE; self.row_h=1
        self.scroll_y=0; self.view_h=400

    def update_params(self,sr,fft_size,row_h,scroll_y,view_h):
        self.sr=sr; self.fft_size=fft_size; self.row_h=row_h
        self.scroll_y=scroll_y; self.view_h=view_h; self.update()

    def paintEvent(self,_):
        p=QPainter(self); p.fillRect(self.rect(),QColor(25,25,35))
        p.setPen(QColor(200,200,200)); p.setFont(QFont("Courier",7))
        n=8
        for i in range(n+1):
            frac=i/n; py=int(frac*self.view_h)
            t=(self.scroll_y+py)/self.row_h*self.fft_size/self.sr
            lbl=(f"{t:.3f}s" if t>=1 else f"{t*1e3:.2f}ms" if t>=1e-3
                 else f"{t*1e6:.1f}µs")
            p.drawLine(self.WIDTH-5,py,self.WIDTH,py)
            p.drawText(QRect(0,py-8,self.WIDTH-7,16),
                       Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter,lbl)

# ═══════════════════════════════════════════════════════════════════════════════
#  WATERFALL CANVAS
# ═══════════════════════════════════════════════════════════════════════════════

class WaterfallCanvas(QWidget):
    status_msg         = pyqtSignal(str)
    annotation_changed = pyqtSignal()

    MODE_SELECT = "select"
    MODE_CREATE = "create"

    def __init__(self, cmd_stack: CommandStack, parent=None):
        super().__init__(parent)
        self.cmd_stack  = cmd_stack
        self.dataset: SigMFDataset | None = None
        self.fft_size   = FFT_SIZE
        self.cmap_name  = "Viridis"
        self.vmin=-60.; self.vmax=-20.
        self.row_h=1;   self.total_rows=0

        self._tile_cache: dict[int,QPixmap]={}
        self._pending:    set[int]=set()
        self._threads:    list[TileThread]=[]

        self.annotations:  list[AnnotationItem]=[]
        self.selected_idx: int|None=None

        self._drag_handle:  str|None=None
        self._drag_ann_idx: int|None=None
        self._drag_origin:  QPointF|None=None
        self._drag_rect0:   QRectF|None=None
        self._drag_old_ann: dict|None=None   # snapshot before drag

        self._mode="select"
        self._nb_start: QPointF|None=None
        self._nb_rect:  QRectF|None=None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Preferred)

    # ── public ────────────────────────────────────────────────────────────────
    def load_dataset(self, ds):
        self._abort_threads(); self._tile_cache.clear(); self._pending.clear()
        self.dataset=ds; self.total_rows=math.ceil(ds.total_samples/self.fft_size)
        self._rebuild_annotations()
        self.setFixedSize(self.fft_size, max(self.total_rows*self.row_h,1))
        self.update()

    def set_fft_size(self,n):
        self.fft_size=n
        if self.dataset: self.load_dataset(self.dataset)

    def set_colormap(self,name):
        self.cmap_name=name; self._tile_cache.clear(); self._pending.clear(); self.update()

    def set_vrange(self,vmin,vmax):
        self.vmin=vmin; self.vmax=vmax
        self._tile_cache.clear(); self._pending.clear(); self.update()

    def set_mode(self,mode):
        self._mode=mode; self._nb_start=None; self._nb_rect=None
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor if mode==self.MODE_CREATE
                               else Qt.CursorShape.ArrowCursor))

    def visible_sample_range(self, scroll_y: int, viewport_h: int) -> tuple[int,int]:
        """Return (sample_start, sample_count) for the currently visible rows."""
        row_top    = max(0, scroll_y // self.row_h)
        row_bottom = min(self.total_rows,
                         math.ceil((scroll_y+viewport_h)/self.row_h))
        ss = row_top    * self.fft_size
        sc = (row_bottom - row_top) * self.fft_size
        return ss, sc

    def add_detection_results(self, results: list[DetectionResult]):
        """Called after a detector run; wraps in a BulkAdd command."""
        anns = []
        for r in results:
            a: dict = {
                "core:sample_start":    r.sample_start,
                "core:sample_count":    r.sample_count,
                "core:freq_lower_edge": r.freq_lower,
                "core:freq_upper_edge": r.freq_upper,
            }
            if r.label:
                a["core:label"] = r.label
            anns.append(a)
        if anns:
            self.cmd_stack.push(BulkAddAnnotationsCmd(self.dataset, anns))
            self._rebuild_annotations()
            self.annotation_changed.emit()
            self.update()

    # ── tiles ─────────────────────────────────────────────────────────────────
    def _tile_for_row(self,row): return (row//TILE_ROWS)*TILE_ROWS

    def _request_tile(self,tile_row):
        if tile_row in self._tile_cache or tile_row in self._pending or not self.dataset:
            return
        rows=min(TILE_ROWS,self.total_rows-tile_row)
        if rows<=0: return
        self._pending.add(tile_row)
        w=TileWorker(self.dataset,self.fft_size,self.cmap_name,
                     self.vmin,self.vmax,tile_row,rows)
        w.tile_ready.connect(self._on_tile_ready)
        t=TileThread(w)
        t.finished.connect(lambda th=t: self._threads.remove(th) if th in self._threads else None)
        self._threads.append(t); t.start()

    def _on_tile_ready(self,row_start,img):
        self._pending.discard(row_start)
        self._tile_cache[row_start]=QPixmap.fromImage(img)
        self.update()

    def _abort_threads(self):
        for t in self._threads: t.quit(); t.wait(300)
        self._threads.clear(); self._pending.clear()

    # ── annotations ───────────────────────────────────────────────────────────
    def _rebuild_annotations(self):
        self.annotations=[AnnotationItem(a,i)
                          for i,a in enumerate(self.dataset.annotations)] \
                         if self.dataset else []

    def _ann_rect(self, item: AnnotationItem) -> QRectF:
        if not self.dataset: return QRectF()
        return item.to_rect(self.fft_size,self.row_h,
                            self.dataset.center_freq,self.dataset.sample_rate,
                            self.width())

    def _rect_to_sigmf(self, r: QRectF):
        ds=self.dataset; r=r.normalized()
        ss=int(round((r.top()/self.row_h)*self.fft_size))
        se=int(round((r.bottom()/self.row_h)*self.fft_size))
        sc=max(self.fft_size, se-ss)
        fl=AnnotationItem.x_to_freq(r.left(),ds.center_freq,ds.sample_rate,self.width())
        fu=AnnotationItem.x_to_freq(r.right(),ds.center_freq,ds.sample_rate,self.width())
        return ss,sc,min(fl,fu),max(fl,fu)

    def _hit_annotation(self, pos: QPointF) -> int|None:
        for i,item in enumerate(self.annotations):
            if self._ann_rect(item).contains(pos): return i
        return None

    # ── paint ─────────────────────────────────────────────────────────────────
    def paintEvent(self,event):
        painter=QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.dataset:
            painter.fillRect(self.rect(),QColor(20,20,30))
            painter.setPen(QColor(180,180,180))
            painter.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,"No dataset loaded")
            return
        clip=event.rect(); w=self.width()
        row_top=max(0,clip.top()//self.row_h)
        row_bottom=min(self.total_rows,math.ceil((clip.bottom()+1)/self.row_h))
        tile_row=self._tile_for_row(row_top)
        while tile_row<row_bottom:
            thr=min(TILE_ROWS,self.total_rows-tile_row)
            ty=tile_row*self.row_h; thpx=thr*self.row_h
            if tile_row in self._tile_cache:
                pm=self._tile_cache[tile_row]
                painter.drawPixmap(QRect(0,ty,w,thpx),pm,pm.rect())
            else:
                painter.fillRect(QRect(0,ty,w,thpx),QColor(15,15,25))
                painter.setPen(QColor(60,60,80))
                painter.drawText(QRect(0,ty,w,min(20,thpx)),
                                 Qt.AlignmentFlag.AlignCenter,"Loading…")
                self._request_tile(tile_row)
            tile_row+=TILE_ROWS

        for i,item in enumerate(self.annotations):
            r=self._ann_rect(item)
            if not QRectF(clip).intersects(r): continue
            sel=i==self.selected_idx
            col=SEL_COLOR if sel else ANN_COLOR
            painter.setPen(QPen(col,1.5))
            painter.setBrush(QBrush(QColor(col.red(),col.green(),col.blue(),30)))
            painter.drawRect(r)
            lbl=item.label()
            if lbl:
                painter.setPen(col)
                fm=painter.fontMetrics(); tr=fm.boundingRect(lbl)
                tx=r.left()+3; ty=r.top()-3
                painter.fillRect(QRectF(tx-1,ty-tr.height(),tr.width()+4,tr.height()+2),
                                 QColor(0,0,0,160))
                painter.drawText(QPointF(tx,ty),lbl)
            if sel:
                painter.setPen(QPen(SEL_COLOR,1)); painter.setBrush(QBrush(SEL_COLOR))
                for pt in item.handles(r).values():
                    painter.drawRect(item.handle_rect(pt))

        if self._nb_rect and not self._nb_rect.isNull():
            painter.setPen(QPen(QColor(0,255,128),1.5,Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(QColor(0,255,128,25)))
            painter.drawRect(self._nb_rect.normalized())

    # ── mouse ─────────────────────────────────────────────────────────────────
    def mousePressEvent(self,event):
        if not self.dataset: return
        pos=QPointF(event.position())
        if event.button()==Qt.MouseButton.RightButton:
            hit=self._hit_annotation(pos)
            if hit is not None: self._context_menu(hit,event.globalPosition().toPoint())
            return
        if event.button()!=Qt.MouseButton.LeftButton: return
        if self._mode==self.MODE_CREATE:
            self._nb_start=pos; self._nb_rect=QRectF(pos,pos); return
        if self.selected_idx is not None:
            item=self.annotations[self.selected_idx]
            r=self._ann_rect(item); h=item.hit_handle(pos,r)
            if h:
                self._drag_handle=h; self._drag_ann_idx=self.selected_idx
                self._drag_origin=pos; self._drag_rect0=QRectF(r)
                self._drag_old_ann=copy.deepcopy(self.dataset.annotations[self.selected_idx])
                return
        hit=self._hit_annotation(pos)
        if hit is not None:
            self.selected_idx=hit; item=self.annotations[hit]
            self._drag_handle="move"; self._drag_ann_idx=hit
            self._drag_origin=pos; self._drag_rect0=QRectF(self._ann_rect(item))
            self._drag_old_ann=copy.deepcopy(self.dataset.annotations[hit])
        else:
            self.selected_idx=None
        self.update()

    def mouseMoveEvent(self,event):
        if not self.dataset: return
        pos=QPointF(event.position())
        if self._mode==self.MODE_CREATE and self._nb_start:
            x0,y0=self._nb_start.x(),self._nb_start.y()
            self._nb_rect=QRectF(min(x0,pos.x()),min(y0,pos.y()),
                                 abs(pos.x()-x0),abs(pos.y()-y0))
            self.update(); return
        if self._drag_handle and self._drag_origin and self._drag_rect0 is not None:
            dx=pos.x()-self._drag_origin.x(); dy=pos.y()-self._drag_origin.y()
            r=QRectF(self._drag_rect0); h=self._drag_handle
            if   h=="move": r.translate(dx,dy)
            elif h=="tl":   r.setTopLeft(r.topLeft()+QPointF(dx,dy))
            elif h=="tr":   r.setTopRight(r.topRight()+QPointF(dx,dy))
            elif h=="bl":   r.setBottomLeft(r.bottomLeft()+QPointF(dx,dy))
            elif h=="br":   r.setBottomRight(r.bottomRight()+QPointF(dx,dy))
            elif h=="tm":   r.setTop(r.top()+dy)
            elif h=="bm":   r.setBottom(r.bottom()+dy)
            elif h=="lm":   r.setLeft(r.left()+dx)
            elif h=="rm":   r.setRight(r.right()+dx)
            r=r.normalized()
            if r.width()>=MIN_BOX_PX and r.height()>=MIN_BOX_PX:
                ss,sc,fl,fu=self._rect_to_sigmf(r)
                idx=self._drag_ann_idx
                if idx is not None:
                    # Direct update during drag (not via cmd_stack – committed on release)
                    self.dataset.annotations[idx].update({
                        "core:sample_start":ss,"core:sample_count":sc,
                        "core:freq_lower_edge":fl,"core:freq_upper_edge":fu})
                    self.annotations[idx].ann=self.dataset.annotations[idx]
                self.update()
                self.status_msg.emit(f"ss={ss:,}  sc={sc:,}  fl={fl/1e6:.3f}MHz  fu={fu/1e6:.3f}MHz")
            return
        if self._mode==self.MODE_SELECT:
            cur=Qt.CursorShape.ArrowCursor
            if self.selected_idx is not None:
                item=self.annotations[self.selected_idx]; r=self._ann_rect(item)
                h=item.hit_handle(pos,r)
                if h: cur=item.HANDLE_CURSORS.get(h,Qt.CursorShape.ArrowCursor)
                elif r.contains(pos): cur=Qt.CursorShape.SizeAllCursor
            self.setCursor(QCursor(cur))

    def mouseReleaseEvent(self,event):
        if not self.dataset or event.button()!=Qt.MouseButton.LeftButton: return
        pos=QPointF(event.position())
        if self._mode==self.MODE_CREATE and self._nb_rect:
            r=self._nb_rect.normalized()
            if r.width()>=MIN_BOX_PX and r.height()>=MIN_BOX_PX:
                ss,sc,fl,fu=self._rect_to_sigmf(r)
                label,ok=QInputDialog.getText(self,"New Annotation","Label (optional):")
                if ok:
                    ann={"core:sample_start":ss,"core:sample_count":sc,
                         "core:freq_lower_edge":fl,"core:freq_upper_edge":fu}
                    if label.strip(): ann["core:label"]=label.strip()
                    self.cmd_stack.push(AddAnnotationCmd(self.dataset, ann))
                    self._rebuild_annotations()
                    self.selected_idx=len(self.annotations)-1
                    self.annotation_changed.emit()
            self._nb_start=None; self._nb_rect=None; self.update(); return
        if self._drag_handle:
            # Commit drag as an undoable command (diff old vs new)
            idx=self._drag_ann_idx
            if idx is not None and self._drag_old_ann is not None:
                new_ann=self.dataset.annotations[idx]
                new_vals={k:new_ann[k] for k in
                          ("core:sample_start","core:sample_count",
                           "core:freq_lower_edge","core:freq_upper_edge")}
                old_vals={k:self._drag_old_ann.get(k) for k in new_vals}
                if new_vals!=old_vals:
                    # restore old so cmd.execute() applies new
                    self.dataset.annotations[idx].update(old_vals)
                    self.cmd_stack.push(EditAnnotationCmd(self.dataset,idx,new_vals))
            self._rebuild_annotations()
            self.annotation_changed.emit()
            self._drag_handle=None; self._drag_ann_idx=None
            self._drag_origin=None; self._drag_rect0=None; self._drag_old_ann=None
            self.update()

    def mouseDoubleClickEvent(self,event):
        if not self.dataset: return
        hit=self._hit_annotation(QPointF(event.position()))
        if hit is not None: self._edit_label(hit)

    def keyPressEvent(self,event):
        if event.key() in (Qt.Key.Key_Delete,Qt.Key.Key_Backspace):
            if self.selected_idx is not None: self._delete_annotation(self.selected_idx)

    def _context_menu(self,idx,gpos):
        self.selected_idx=idx; self.update()
        menu=QMenu(self)
        a_edt=menu.addAction("Edit Label…"); a_del=menu.addAction("Delete")
        act=menu.exec(gpos)
        if act==a_edt: self._edit_label(idx)
        elif act==a_del: self._delete_annotation(idx)

    def _edit_label(self,idx):
        label,ok=QInputDialog.getText(self,"Edit Label","Label:",
                                      text=self.annotations[idx].label())
        if ok:
            self.cmd_stack.push(EditAnnotationCmd(
                self.dataset, self.annotations[idx].index,
                {"core:label": label.strip()}))
            self._rebuild_annotations()
            self.selected_idx=idx
            self.annotation_changed.emit(); self.update()

    def _delete_annotation(self,idx):
        self.cmd_stack.push(DeleteAnnotationCmd(
            self.dataset, self.annotations[idx].index))
        self._rebuild_annotations()
        self.selected_idx=None
        self.annotation_changed.emit(); self.update()

    def closeEvent(self,event):
        self._abort_threads(); super().closeEvent(event)

# ═══════════════════════════════════════════════════════════════════════════════
#  DETECTOR BACKGROUND WORKER
# ═══════════════════════════════════════════════════════════════════════════════

class DetectorWorker(QObject):
    progress   = pyqtSignal(int)          # 0-100
    finished   = pyqtSignal(list)         # list[DetectionResult]
    error      = pyqtSignal(str)

    def __init__(self, detector_cls: type[BaseDetector], ctx: DetectionContext,
                 kwargs: dict):
        super().__init__()
        self.det_cls = detector_cls
        self.ctx     = ctx
        self.kwargs  = kwargs

    def run(self):
        try:
            det     = self.det_cls()
            results = det.run(self.ctx, **self.kwargs)
            self.progress.emit(100)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

# ═══════════════════════════════════════════════════════════════════════════════
#  DETECTOR PANEL  (dynamic parameter widgets)
# ═══════════════════════════════════════════════════════════════════════════════

class DetectorPanel(QWidget):
    run_visible  = pyqtSignal(type, dict)   # (detector_cls, kwargs)
    run_all      = pyqtSignal(type, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(220)
        self._detectors  = get_all_detectors()
        self._param_widgets: dict[str, QWidget] = {}
        self._current_cls: type[BaseDetector]|None = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        root.addWidget(QLabel("Energy Detection", font=QFont("Arial",10,QFont.Weight.Bold)))

        # Detector selector
        grp = QGroupBox("Algorithm")
        gl  = QVBoxLayout(grp)
        self.det_combo = QComboBox()
        for cls in self._detectors:
            self.det_combo.addItem(cls.NAME, cls)
        self.det_combo.currentIndexChanged.connect(self._on_detector_changed)
        gl.addWidget(self.det_combo)
        root.addWidget(grp)

        # Dynamic parameter area
        self.param_group = QGroupBox("Parameters")
        self.param_layout = QFormLayout(self.param_group)
        self.param_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        root.addWidget(self.param_group)

        # Run buttons
        btn_vis = QPushButton("▶  Run on Visible")
        btn_vis.setToolTip("Run detector on currently visible tiles only")
        btn_vis.clicked.connect(self._emit_visible)
        root.addWidget(btn_vis)

        btn_all = QPushButton("▶▶  Run on Entire Dataset")
        btn_all.setToolTip("Run detector on all samples in the file")
        btn_all.clicked.connect(self._emit_all)
        root.addWidget(btn_all)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0,100); self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        root.addWidget(self.status_lbl)

        root.addStretch()

        # Populate first detector
        if self._detectors:
            self._on_detector_changed(0)

    def _on_detector_changed(self, idx: int):
        cls = self.det_combo.itemData(idx)
        self._current_cls = cls
        self._param_widgets.clear()
        # clear existing rows
        while self.param_layout.rowCount():
            self.param_layout.removeRow(0)
        for p in cls.PARAMS:
            w = self._make_widget(p)
            if w:
                if p.tooltip:
                    w.setToolTip(p.tooltip)
                self._param_widgets[p.name] = w
                self.param_layout.addRow(p.label + ":", w)

    def _make_widget(self, p: DetectorParam) -> QWidget|None:
        if p.kind == "float":
            w = QDoubleSpinBox()
            w.setRange(p.minimum, p.maximum)
            w.setSingleStep(p.step); w.setDecimals(p.decimals)
            w.setValue(float(p.default)); return w
        if p.kind == "int":
            w = QSpinBox()
            w.setRange(int(p.minimum), int(p.maximum))
            w.setSingleStep(int(p.step)); w.setValue(int(p.default)); return w
        if p.kind == "str":
            w = QLineEdit(); w.setText(str(p.default)); return w
        if p.kind == "bool":
            w = QCheckBox(); w.setChecked(bool(p.default)); return w
        if p.kind == "choice":
            w = QComboBox()
            w.addItems(p.choices)
            if p.default in p.choices:
                w.setCurrentText(str(p.default))
            return w
        return None

    def _collect_kwargs(self) -> dict:
        cls = self._current_cls
        if not cls: return {}
        out = {}
        for p in cls.PARAMS:
            w = self._param_widgets.get(p.name)
            if w is None:
                out[p.name] = p.default; continue
            if p.kind == "float":   out[p.name] = w.value()
            elif p.kind == "int":   out[p.name] = w.value()
            elif p.kind == "str":   out[p.name] = w.text()
            elif p.kind == "bool":  out[p.name] = w.isChecked()
            elif p.kind == "choice":out[p.name] = w.currentText()
        return out

    def _emit_visible(self):
        if self._current_cls:
            self.run_visible.emit(self._current_cls, self._collect_kwargs())

    def _emit_all(self):
        if self._current_cls:
            self.run_all.emit(self._current_cls, self._collect_kwargs())

    def set_running(self, running: bool):
        self.progress_bar.setVisible(running)
        if running: self.progress_bar.setValue(0)

    def set_progress(self, v: int):
        self.progress_bar.setValue(v)

    def set_status(self, msg: str):
        self.status_lbl.setText(msg)

# ═══════════════════════════════════════════════════════════════════════════════
#  ANNOTATION LIST PANEL
# ═══════════════════════════════════════════════════════════════════════════════

class AnnotationPanel(QWidget):
    jump_to = pyqtSignal(int)
    def __init__(self,p=None):
        super().__init__(p)
        lay=QVBoxLayout(self); lay.setContentsMargins(4,4,4,4)
        lbl=QLabel("Annotations",font=QFont("Arial",10,QFont.Weight.Bold))
        lay.addWidget(lbl)
        self.lw=QListWidget()
        self.lw.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.lw.itemDoubleClicked.connect(lambda li:
            self.jump_to.emit(li.data(Qt.ItemDataRole.UserRole) or 0))
        lay.addWidget(self.lw)
        self.setMinimumWidth(190); self.setMaximumWidth(270)

    def refresh(self, annotations, fft_size):
        self.lw.clear()
        for item in annotations:
            ss=item.ann.get("core:sample_start",0)
            sc=item.ann.get("core:sample_count",0)
            lbl=item.label() or "(no label)"
            li=QListWidgetItem(f"[{ss:,}–{ss+sc:,}]  {lbl}")
            li.setData(Qt.ItemDataRole.UserRole, ss//fft_size)
            self.lw.addItem(li)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SigMF Waterfall Annotator")
        self.resize(1380, 800)
        self.dataset: SigMFDataset|None = None
        self._det_thread: QThread|None  = None
        self._det_worker: DetectorWorker|None = None

        self.cmd_stack = CommandStack()
        self.cmd_stack.changed = self._on_stack_changed

        self._build_ui()
        self._bind_shortcuts()

    def _build_ui(self):
        # ── toolbar ──────────────────────────────────────────────────────────
        tb = QToolBar(); tb.setMovable(False); self.addToolBar(tb)

        def tbtn(label, slot, checkable=False, enabled=True):
            b=QPushButton(label); b.setFixedHeight(28)
            b.setCheckable(checkable); b.setEnabled(enabled)
            b.clicked.connect(slot); tb.addWidget(b); return b

        tbtn("📂 Open SigMF…", self.open_file)
        tb.addSeparator()
        self.btn_save=tbtn("💾 Save", self.save_file, enabled=False)
        tb.addSeparator()
        tb.addWidget(QLabel("  Mode: "))
        self.btn_sel=tbtn("🔲 Select",lambda:self._set_mode("select"),True)
        self.btn_sel.setChecked(True)
        self.btn_drw=tbtn("✏️ Draw",  lambda:self._set_mode("create"),True)
        tb.addSeparator()
        self.btn_undo=tbtn("↩ Undo", self._undo, enabled=False)
        self.btn_redo=tbtn("↪ Redo", self._redo, enabled=False)
        tb.addSeparator()
        tb.addWidget(QLabel("  FFT: "))
        self.fft_spin=QSpinBox(); self.fft_spin.setRange(64,8192)
        self.fft_spin.setValue(FFT_SIZE); self.fft_spin.setSingleStep(64)
        self.fft_spin.setFixedWidth(80)
        self.fft_spin.editingFinished.connect(self._on_fft)
        tb.addWidget(self.fft_spin); tb.addSeparator()
        tb.addWidget(QLabel("  Colormap: "))
        self.cmap_cb=QComboBox(); self.cmap_cb.addItems(list(COLORMAPS))
        self.cmap_cb.currentTextChanged.connect(lambda n: self.canvas.set_colormap(n))
        tb.addWidget(self.cmap_cb); tb.addSeparator()
        tb.addWidget(QLabel("  dB min: "))
        self.vmin_sp=QDoubleSpinBox(); self.vmin_sp.setRange(-200,0)
        self.vmin_sp.setValue(-60); self.vmin_sp.setFixedWidth(70)
        self.vmin_sp.editingFinished.connect(self._on_vrange); tb.addWidget(self.vmin_sp)
        tb.addWidget(QLabel("  max: "))
        self.vmax_sp=QDoubleSpinBox(); self.vmax_sp.setRange(-200,0)
        self.vmax_sp.setValue(-20); self.vmax_sp.setFixedWidth(70)
        self.vmax_sp.editingFinished.connect(self._on_vrange); tb.addWidget(self.vmax_sp)

        # ── central layout ────────────────────────────────────────────────────
        central=QWidget(); self.setCentralWidget(central)
        root=QHBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        splitter=QSplitter(Qt.Orientation.Horizontal)

        # Left: time axis + [freq axis / canvas]
        left=QWidget(); lh=QHBoxLayout(left)
        lh.setContentsMargins(0,0,0,0); lh.setSpacing(0)
        self.time_axis=TimeAxisWidget(); lh.addWidget(self.time_axis)
        rcol=QWidget(); rv=QVBoxLayout(rcol)
        rv.setContentsMargins(0,0,0,0); rv.setSpacing(0)
        self.freq_axis=FreqAxisWidget(); rv.addWidget(self.freq_axis)
        self.scroll=QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.verticalScrollBar().valueChanged.connect(self._on_scroll)
        self.canvas=WaterfallCanvas(self.cmd_stack)
        self.canvas.status_msg.connect(lambda m: self.statusBar().showMessage(m))
        self.canvas.annotation_changed.connect(self._on_ann_changed)
        self.scroll.setWidget(self.canvas)
        rv.addWidget(self.scroll); lh.addWidget(rcol)
        splitter.addWidget(left)

        # Middle: annotation list
        self.ann_panel=AnnotationPanel()
        self.ann_panel.jump_to.connect(self._jump_to_row)
        splitter.addWidget(self.ann_panel)

        # Right: detector panel
        self.det_panel=DetectorPanel()
        self.det_panel.run_visible.connect(self._run_detector_visible)
        self.det_panel.run_all.connect(self._run_detector_all)
        splitter.addWidget(self.det_panel)

        splitter.setSizes([860, 230, 250])
        root.addWidget(splitter)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready — open a .sigmf-meta file.")

    def _bind_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self._redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self).activated.connect(self._redo)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_file)

    # ── undo/redo ─────────────────────────────────────────────────────────────
    def _undo(self):
        if not self.cmd_stack.can_undo: return
        self.cmd_stack.undo()
        self.canvas._rebuild_annotations()
        self.canvas.selected_idx=None
        self.canvas.update()
        self._on_ann_changed()

    def _redo(self):
        if not self.cmd_stack.can_redo: return
        self.cmd_stack.redo()
        self.canvas._rebuild_annotations()
        self.canvas.selected_idx=None
        self.canvas.update()
        self._on_ann_changed()

    def _on_stack_changed(self):
        self.btn_undo.setEnabled(self.cmd_stack.can_undo)
        self.btn_redo.setEnabled(self.cmd_stack.can_redo)
        ul=self.cmd_stack.undo_label; rl=self.cmd_stack.redo_label
        self.btn_undo.setToolTip(f"Undo: {ul}" if ul else "Nothing to undo")
        self.btn_redo.setToolTip(f"Redo: {rl}" if rl else "Nothing to redo")

    # ── detector ──────────────────────────────────────────────────────────────
    def _run_detector_visible(self, cls, kwargs):
        if not self.dataset: return
        sv = self.scroll.verticalScrollBar().value()
        vh = self.scroll.viewport().height()
        ss, sc = self.canvas.visible_sample_range(sv, vh)
        self._launch_detector(cls, kwargs, ss, sc)

    def _run_detector_all(self, cls, kwargs):
        if not self.dataset: return
        self._launch_detector(cls, kwargs, 0, self.dataset.total_samples)

    def _launch_detector(self, cls: type[BaseDetector], kwargs: dict,
                         ss: int, sc: int):
        if self._det_thread and self._det_thread.isRunning():
            QMessageBox.information(self,"Busy","A detector is already running.")
            return
        samples = self.dataset.read_samples(ss, sc)
        ctx = DetectionContext(
            samples      = samples,
            sample_rate  = self.dataset.sample_rate,
            center_freq  = self.dataset.center_freq,
            sample_start = ss,
            fft_size     = self.canvas.fft_size,
            sigmf_meta   = self.dataset.meta,
        )
        self._det_worker = DetectorWorker(cls, ctx, kwargs)
        self._det_thread = QThread()
        self._det_worker.moveToThread(self._det_thread)
        self._det_thread.started.connect(self._det_worker.run)
        self._det_worker.progress.connect(self.det_panel.set_progress)
        self._det_worker.finished.connect(self._on_detection_done)
        self._det_worker.error.connect(self._on_detection_error)
        self._det_worker.finished.connect(self._det_thread.quit)
        self._det_worker.error.connect(self._det_thread.quit)
        self.det_panel.set_running(True)
        self.det_panel.set_status("Running…")
        self._det_thread.start()

    def _on_detection_done(self, results: list):
        self.det_panel.set_running(False)
        n=len(results)
        self.det_panel.set_status(f"Done — {n} annotation(s) added.")
        self.statusBar().showMessage(f"Detection complete: {n} result(s).")
        if results:
            self.canvas.add_detection_results(results)
            self._on_ann_changed()

    def _on_detection_error(self, msg: str):
        self.det_panel.set_running(False)
        self.det_panel.set_status(f"Error: {msg}")
        QMessageBox.critical(self,"Detector Error", msg)

    # ── file ──────────────────────────────────────────────────────────────────
    def open_file(self):
        path,_=QFileDialog.getOpenFileName(
            self,"Select SigMF Metadata File","","SigMF Metadata (*.sigmf-meta)")
        if not path: return
        if not path.lower().endswith(".sigmf-meta"):
            QMessageBox.warning(self,"Invalid File","Please select a .sigmf-meta file.")
            return
        try: ds=SigMFDataset(path)
        except Exception as e:
            QMessageBox.critical(self,"Load Error",str(e)); return
        self.dataset=ds
        self.cmd_stack.clear()
        self.canvas.set_fft_size(self.fft_spin.value())
        self.canvas.set_colormap(self.cmap_cb.currentText())
        vmin,vmax=self._estimate_vrange(ds)
        self.vmin_sp.setValue(vmin); self.vmax_sp.setValue(vmax)
        self.canvas.set_vrange(vmin,vmax)
        self.canvas.load_dataset(ds)
        self.freq_axis.set_params(ds.center_freq,ds.sample_rate)
        self._sync_canvas_width(); self._on_ann_changed()
        self.btn_save.setEnabled(True)
        self.statusBar().showMessage(
            f"Loaded: {Path(path).name}  |  {ds.total_samples:,} samples  |  "
            f"SR={ds.sample_rate/1e6:.3f} MHz  |  CF={ds.center_freq/1e6:.3f} MHz  |  "
            f"{len(ds.annotations)} annotation(s)")

    def save_file(self):
        if not self.dataset: return
        try:
            self.dataset.save()
            self.statusBar().showMessage(f"Saved → {self.dataset.meta_path}")
        except Exception as e:
            QMessageBox.critical(self,"Save Error",str(e))

    # ── helpers ───────────────────────────────────────────────────────────────
    def _set_mode(self,mode):
        self.canvas.set_mode(mode)
        self.btn_sel.setChecked(mode=="select")
        self.btn_drw.setChecked(mode=="create")

    def _on_fft(self):
        if self.dataset: self.canvas.set_fft_size(self.fft_spin.value())

    def _on_vrange(self):
        self.canvas.set_vrange(self.vmin_sp.value(),self.vmax_sp.value())

    def _on_scroll(self,val):
        if self.dataset:
            self.time_axis.update_params(self.dataset.sample_rate,
                                         self.canvas.fft_size, self.canvas.row_h,
                                         val, self.scroll.viewport().height())

    def _on_ann_changed(self):
        if self.dataset:
            self.ann_panel.refresh(self.canvas.annotations, self.canvas.fft_size)

    def _jump_to_row(self,row):
        self.scroll.verticalScrollBar().setValue(row*self.canvas.row_h)

    def _sync_canvas_width(self):
        vw=self.scroll.viewport().width()
        if self.dataset and vw>0:
            self.canvas.setFixedWidth(vw)
            self.canvas.setFixedHeight(max(self.canvas.total_rows,1))

    def resizeEvent(self,event):
        super().resizeEvent(event); self._sync_canvas_width()

    def _estimate_vrange(self, ds, probe_rows=64):
        fs=self.fft_spin.value()
        samples=ds.read_samples(0,probe_rows*fs)
        if samples.size<fs: return -60.,-20.
        pad=(-len(samples))%fs
        if pad: samples=np.concatenate([samples,np.zeros(pad,dtype=np.complex64)])
        rows=len(samples)//fs; mat=samples.reshape(rows,fs)
        win=np.hanning(fs).astype(np.float32); wg=np.sum(win**2)/fs
        spec=np.fft.fftshift(np.fft.fft(mat*win,axis=1),axes=1)
        pwr=10*np.log10(np.abs(spec/fs)**2/wg+1e-12)
        p2,p98=float(np.percentile(pwr,2)),float(np.percentile(pwr,98))
        m=max((p98-p2)*0.15,2.)
        return round(p2-m,1),round(p98+m,1)

    def closeEvent(self,event):
        self.canvas._abort_threads(); super().closeEvent(event)


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    app=QApplication(sys.argv); app.setStyle("Fusion")
    pal=QPalette()
    pal.setColor(QPalette.ColorRole.Window,        QColor(35,35,45))
    pal.setColor(QPalette.ColorRole.WindowText,    QColor(220,220,220))
    pal.setColor(QPalette.ColorRole.Base,          QColor(25,25,35))
    pal.setColor(QPalette.ColorRole.Text,          QColor(220,220,220))
    pal.setColor(QPalette.ColorRole.Button,        QColor(50,50,65))
    pal.setColor(QPalette.ColorRole.ButtonText,    QColor(220,220,220))
    pal.setColor(QPalette.ColorRole.Highlight,     QColor(42,130,218))
    pal.setColor(QPalette.ColorRole.HighlightedText,QColor(0,0,0))
    app.setPalette(pal)
    win=MainWindow(); win.show(); sys.exit(app.exec())

if __name__=="__main__":
    main()
