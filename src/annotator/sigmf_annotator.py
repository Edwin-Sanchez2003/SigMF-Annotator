"""
SigMF Waterfall Annotator  –  v2
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

v3 changes
==========
* FreqAxisWidget now always matches the viewport width of the scroll area —
  axis and waterfall tiles never misalign on resize or splitter drag.
* Side panels cannot crush the waterfall: the left (waterfall) section gets all
  stretch; side panels have a fixed maximum width and a "collapse" behaviour via
  QSplitter minimum sizes.
* Overlapping annotations: repeated clicks on the same spot cycle through the
  stack underneath.  A translucent badge shows how many annotations are stacked.
* Pan & zoom: scroll-wheel zooms (centred on cursor), Ctrl+drag or middle-drag
  pans.  All coord conversions go through a ViewTransform so annotations, axes,
  and new-box drawing remain pixel-perfect at any zoom level.
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
    QLineEdit, QGroupBox, QCheckBox, QFrame, QDockWidget
)
from PyQt6.QtCore import (
    Qt, QRect, QPoint, QPointF, QRectF, QThread, pyqtSignal, QObject, QSize,
    QTimer
)
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont, QPixmap, QImage,
    QCursor, QPalette, QKeySequence, QAction, QShortcut, QTransform,
    QWheelEvent
)

# ═══════════════════════════════════════════════════════════════════════════════
#  DETECTOR PLUGIN INTERFACE  (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectorParam:
    name:     str
    label:    str
    kind:     str
    default:  Any
    minimum:  float = 0.0
    maximum:  float = 1e9
    step:     float = 1.0
    decimals: int   = 3
    choices:  list[str] = field(default_factory=list)
    tooltip:  str   = ""


@dataclass
class DetectionContext:
    samples:      np.ndarray
    sample_rate:  float
    center_freq:  float
    sample_start: int
    fft_size:     int
    sigmf_meta:   dict


@dataclass
class DetectionResult:
    sample_start:  int
    sample_count:  int
    freq_lower:    float
    freq_upper:    float
    label:         str = ""


class BaseDetector(ABC):
    NAME:   str = "Unnamed Detector"
    PARAMS: list[DetectorParam] = []

    @abstractmethod
    def run(self, ctx: DetectionContext, **kwargs) -> list[DetectionResult]: ...


# ═══════════════════════════════════════════════════════════════════════════════
#  BUILT-IN DETECTORS  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class EnergyThresholdDetector(BaseDetector):
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

        if samples.size < fs: return []
        pad = (-len(samples)) % fs
        if pad: samples = np.concatenate([samples, np.zeros(pad, dtype=np.complex64)])
        rows = len(samples) // fs
        mat  = samples.reshape(rows, fs)
        win  = np.hanning(fs).astype(np.float32)
        wg   = np.sum(win**2) / fs
        spec = np.fft.fftshift(np.fft.fft(mat * win, axis=1), axes=1)
        pwr  = 10 * np.log10(np.abs(spec / fs)**2 / wg + 1e-12)
        row_energy = pwr.mean(axis=1)

        hot = row_energy >= thr_db
        runs: list[tuple[int,int]] = []
        i = 0
        while i < rows:
            if hot[i]:
                j = i
                while j < rows and hot[j]: j += 1
                runs.append((i, j - 1)); i = j
            else: i += 1

        if merge_gap > 0 and len(runs) > 1:
            merged = [runs[0]]
            for (rs, re) in runs[1:]:
                if rs - merged[-1][1] <= merge_gap: merged[-1] = (merged[-1][0], re)
                else: merged.append((rs, re))
            runs = merged

        results = []
        for (rs, re) in runs:
            if (re - rs + 1) < min_dur: continue
            run_pwr = pwr[rs:re+1, :]
            col_max = run_pwr.max(axis=0)
            lo_pct  = np.percentile(col_max, freq_pct)
            active  = np.where(col_max >= lo_pct)[0]
            if active.size == 0: continue
            bin_lo = int(active[0]); bin_hi = int(active[-1])
            freq_lo = cf - sr/2 + bin_lo * (sr / fs)
            freq_hi = cf - sr/2 + (bin_hi + 1) * (sr / fs)
            abs_ss  = ss_base + rs * fs
            abs_sc  = (re - rs + 1) * fs
            results.append(DetectionResult(abs_ss, abs_sc, freq_lo, freq_hi, label))
        return results


class SpectralVarianceDetector(BaseDetector):
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

        if samples.size < fs: return []
        pad = (-len(samples)) % fs
        if pad: samples = np.concatenate([samples, np.zeros(pad, dtype=np.complex64)])
        total_rows = len(samples) // fs
        mat  = samples.reshape(total_rows, fs)
        win  = np.hanning(fs).astype(np.float32)
        wg   = np.sum(win**2) / fs
        spec = np.fft.fftshift(np.fft.fft(mat * win, axis=1), axes=1)
        pwr  = 10 * np.log10(np.abs(spec / fs)**2 / wg + 1e-12)

        results = []
        step = max(1, chunk_rows - overlap)
        rs = 0
        while rs < total_rows:
            re    = min(rs + chunk_rows, total_rows)
            chunk = pwr[rs:re, :]
            var   = chunk.var(axis=0)
            active = np.where(var >= var_thr)[0]
            if active.size >= min_bw:
                groups = []
                gi = 0
                while gi < len(active):
                    gj = gi
                    while gj + 1 < len(active) and active[gj+1] == active[gj] + 1: gj += 1
                    if (gj - gi + 1) >= min_bw: groups.append((int(active[gi]), int(active[gj])))
                    gi = gj + 1
                for b_lo, b_hi in groups:
                    freq_lo = cf - sr/2 + b_lo * (sr / fs)
                    freq_hi = cf - sr/2 + (b_hi + 1) * (sr / fs)
                    abs_ss  = ss_base + rs * fs
                    abs_sc  = (re - rs) * fs
                    results.append(DetectionResult(abs_ss, abs_sc, freq_lo, freq_hi, label))
            rs += step
        return results


def get_all_detectors() -> list[type[BaseDetector]]:
    def _recurse(cls):
        result = []
        for sub in cls.__subclasses__():
            result.append(sub); result.extend(_recurse(sub))
        return result
    return _recurse(BaseDetector)


def load_detectors_from_dir(directory: str | Path) -> list[str]:
    import importlib.util
    directory = Path(directory).resolve()
    if not directory.is_dir(): return []
    _ensure_importable()
    loaded = []
    for py_file in sorted(directory.glob("*.py")):
        if py_file.stem.startswith("_"): continue
        mod_name = f"_sigmf_detector_{py_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, py_file)
            mod  = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
            loaded.append(py_file.name)
        except Exception as e:
            print(f"[detector loader] Could not load {py_file.name}: {e}")
    return loaded


def _ensure_importable():
    here = Path(__file__).resolve().parent
    candidates = [here.parent, here.parent.parent, here]
    for p in candidates:
        s = str(p)
        if s not in sys.path: sys.path.insert(0, s)
    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        pkg_name = "annotator.sigmf_annotator"
        if pkg_name not in sys.modules: sys.modules[pkg_name] = main_mod


# ═══════════════════════════════════════════════════════════════════════════════
#  UNDO / REDO  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class Command(ABC):
    @abstractmethod
    def execute(self): ...
    @abstractmethod
    def undo(self): ...
    description: str = ""


class CommandStack:
    def __init__(self, max_depth: int = 200):
        self._undo: list[Command] = []
        self._redo: list[Command] = []
        self._max  = max_depth
        self.changed = None

    def push(self, cmd: Command):
        cmd.execute()
        self._undo.append(cmd)
        if len(self._undo) > self._max: self._undo.pop(0)
        self._redo.clear(); self._notify()

    def undo(self):
        if not self._undo: return
        cmd = self._undo.pop(); cmd.undo(); self._redo.append(cmd); self._notify()

    def redo(self):
        if not self._redo: return
        cmd = self._redo.pop(); cmd.execute(); self._undo.append(cmd); self._notify()

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
        if callable(self.changed): self.changed()


class AddAnnotationCmd(Command):
    def __init__(self, dataset, ann: dict):
        self.ds = dataset; self.ann = ann
        self.description = f"Add annotation '{ann.get('core:label','')!r}'"
    def execute(self): self.ds.annotations.append(self.ann)
    def undo(self):
        if self.ann in self.ds.annotations: self.ds.annotations.remove(self.ann)


class DeleteAnnotationCmd(Command):
    def __init__(self, dataset, index: int):
        self.ds = dataset; self.index = index
        self.ann = copy.deepcopy(dataset.annotations[index])
        self.description = f"Delete annotation '{self.ann.get('core:label','')!r}'"
    def execute(self):
        try: self.ds.annotations.pop(self.index)
        except IndexError: pass
    def undo(self): self.ds.annotations.insert(self.index, copy.deepcopy(self.ann))


class EditAnnotationCmd(Command):
    def __init__(self, dataset, index: int, new_values: dict):
        self.ds = dataset; self.index = index; self.new = new_values
        self.old = {k: dataset.annotations[index][k]
                    for k in new_values if k in dataset.annotations[index]}
        self.description = "Edit annotation"
    def execute(self): self.ds.annotations[self.index].update(self.new)
    def undo(self):
        self.ds.annotations[self.index].update(self.old)
        for k in self.new:
            if k not in self.old: self.ds.annotations[self.index].pop(k, None)


class BulkAddAnnotationsCmd(Command):
    def __init__(self, dataset, anns: list[dict]):
        self.ds = dataset; self.anns = anns
        self.description = f"Detect: add {len(anns)} annotation(s)"
    def execute(self): self.ds.annotations.extend(self.anns)
    def undo(self):
        for a in self.anns:
            if a in self.ds.annotations: self.ds.annotations.remove(a)


class BulkDeleteAnnotationsCmd(Command):
    def __init__(self, dataset, indices: list[int]):
        self.ds    = dataset
        self.saved = [(i, copy.deepcopy(dataset.annotations[i])) for i in sorted(indices)]
        self.description = f"Delete {len(indices)} annotation(s)"
    def execute(self):
        for i, _ in sorted(self.saved, reverse=True):
            try: self.ds.annotations.pop(i)
            except IndexError: pass
    def undo(self):
        for i, ann in self.saved: self.ds.annotations.insert(i, copy.deepcopy(ann))


# ═══════════════════════════════════════════════════════════════════════════════
#  SIGMF DATASET  (unchanged)
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
        if not self.meta_path.exists(): raise FileNotFoundError(f"Meta not found: {self.meta_path}")
        if not self.data_path.exists(): raise FileNotFoundError(f"Data not found: {self.data_path}")
        with open(self.meta_path) as f: self.meta = json.load(f)
        g    = self.meta.get("global", {})
        dt   = g.get("core:datatype", "").lower().strip()
        caps = self.meta.get("captures", [{}])
        self.sample_rate  = float(g.get("core:sample_rate", 1.0))
        self.center_freq  = float(caps[0].get("core:frequency", g.get("core:frequency", 0.0)))
        self.annotations  = self.meta.get("annotations", [])
        self._is_complex, self._numpy_dtype, self._bps = self._parse_datatype(dt)
        self.total_samples = self.data_path.stat().st_size // self._bps

    @classmethod
    def _parse_datatype(cls, dt: str):
        dt = dt.replace("complex64", "cf32_le").replace("complex128", "cf64_le")
        m  = re.fullmatch(r'(c|r)(f|i|u)(\d+)(?:_(le|be))?', dt)
        if not m: raise ValueError(f"Unrecognised SigMF datatype: '{dt}'")
        kind_cr, kind_fiu, bits_str, endian = m.groups()
        bits = int(bits_str)
        is_complex  = kind_cr == "c"
        is_float    = kind_fiu == "f"
        is_unsigned = kind_fiu == "u"
        key = (is_unsigned, is_float, bits)
        if key not in cls._SCALAR: raise ValueError(f"Unsupported component type: {kind_fiu}{bits}")
        base = np.dtype(cls._SCALAR[key])
        if bits > 8: base = base.newbyteorder('>' if endian == "be" else '<')
        bps = (bits // 8) * (2 if is_complex else 1)
        return is_complex, base, bps

    def read_samples(self, start: int, count: int) -> np.ndarray:
        start = max(0, min(start, self.total_samples - 1))
        count = min(count, self.total_samples - start)
        if count <= 0: return np.array([], dtype=np.complex64)
        with open(self.data_path, "rb") as f:
            f.seek(start * self._bps); raw = f.read(count * self._bps)
        dt  = self._numpy_dtype
        arr = np.frombuffer(raw, dtype=dt)
        if dt.byteorder == '>': arr = arr.byteswap().newbyteorder()
        return self._to_cf32(arr)

    def _to_cf32(self, arr: np.ndarray) -> np.ndarray:
        dt = self._numpy_dtype
        is_float    = dt.kind == 'f'
        is_unsigned = dt.kind == 'u'
        bits = dt.itemsize * 8
        if self._is_complex:
            if arr.size % 2: arr = arr[:-(arr.size % 2)]
            iq = arr.reshape(-1, 2).astype(np.float32)
        else:
            iq = np.stack([arr.astype(np.float32), np.zeros(len(arr), np.float32)], axis=1)
        if not is_float:
            scale = float(1 << (bits - 1))
            iq = (iq - (scale if is_unsigned else 0)) / scale
        return (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)

    def save(self):
        self.meta["annotations"] = self.annotations
        with open(self.meta_path, "w") as f: json.dump(self.meta, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  COLORMAPS  (unchanged)
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
#  VIEW TRANSFORM
#  Separates "logical canvas space" (pixels at zoom=1, scroll=0) from
#  "viewport space" (what's visible on screen).
#
#  logical → viewport:   vx = (lx - pan_x) * zoom_x
#                        vy = (ly - pan_y) * zoom_y
#  viewport → logical:   lx = vx / zoom_x + pan_x
#                        ly = vy / zoom_y + pan_y
#
#  zoom_x and zoom_y are always equal (isotropic) in this implementation.
# ═══════════════════════════════════════════════════════════════════════════════

class ViewTransform:
    """
    Maps between logical canvas coordinates and viewport (widget) coordinates.

    Logical space:  pixel grid at zoom=1, origin = top-left of full waterfall.
    Viewport space: what the WaterfallViewport widget actually renders.
    """
    def __init__(self):
        self.zoom  = 1.0          # isotropic zoom factor
        self.pan_x = 0.0          # logical-px offset (left edge of viewport)
        self.pan_y = 0.0          # logical-px offset (top edge of viewport)
        # logical dimensions of the full canvas (set on dataset load)
        self.logical_w = 1.0
        self.logical_h = 1.0

    # ── conversion helpers ────────────────────────────────────────────────────
    def l2v(self, lx: float, ly: float) -> tuple[float, float]:
        """Logical → viewport."""
        return (lx - self.pan_x) * self.zoom, (ly - self.pan_y) * self.zoom

    def v2l(self, vx: float, vy: float) -> tuple[float, float]:
        """Viewport → logical."""
        return vx / self.zoom + self.pan_x, vy / self.zoom + self.pan_y

    def l2v_rect(self, r: QRectF) -> QRectF:
        x1, y1 = self.l2v(r.left(),  r.top())
        x2, y2 = self.l2v(r.right(), r.bottom())
        return QRectF(x1, y1, x2 - x1, y2 - y1)

    def v2l_rect(self, r: QRectF) -> QRectF:
        x1, y1 = self.v2l(r.left(),  r.top())
        x2, y2 = self.v2l(r.right(), r.bottom())
        return QRectF(x1, y1, x2 - x1, y2 - y1)

    # ── zoom centred on a viewport point ──────────────────────────────────────
    def zoom_at(self, vx: float, vy: float, factor: float,
                min_zoom: float = 0.05, max_zoom: float = 64.0):
        lx, ly = self.v2l(vx, vy)
        new_zoom = max(min_zoom, min(max_zoom, self.zoom * factor))
        self.pan_x = lx - vx / new_zoom
        self.pan_y = ly - vy / new_zoom
        self.zoom  = new_zoom
        self._clamp()

    def pan_by(self, dvx: float, dvy: float):
        """Move viewport by (dvx, dvy) viewport pixels."""
        self.pan_x -= dvx / self.zoom
        self.pan_y -= dvy / self.zoom
        self._clamp()

    def reset(self, viewport_w: float = 0.0, viewport_h: float = 0.0):
        """Reset zoom to 1× keeping the current view centre in place."""
        cx, cy = self.v2l(viewport_w / 2, viewport_h / 2)
        self.zoom = 1.0
        self.pan_x = cx - viewport_w / 2
        self.pan_y = cy - viewport_h / 2
        self._clamp()

    def _clamp(self):
        """Keep pan within logical canvas bounds."""
        max_px = self.logical_w - 1.0 / self.zoom
        max_py = self.logical_h - 1.0 / self.zoom
        self.pan_x = max(0.0, min(self.pan_x, max(0.0, max_px)))
        self.pan_y = max(0.0, min(self.pan_y, max(0.0, max_py)))

    # ── what logical rows / freq range is visible? ────────────────────────────
    def visible_logical_rect(self, vw: float, vh: float) -> QRectF:
        lx1, ly1 = self.v2l(0,  0)
        lx2, ly2 = self.v2l(vw, vh)
        return QRectF(lx1, ly1, lx2 - lx1, ly2 - ly1)


# ═══════════════════════════════════════════════════════════════════════════════
#  TILE RENDERER  (unchanged logic; worker unchanged)
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
#  ANNOTATION ITEM  (logical-space coords; transform-aware helpers added)
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

    # ── SigMF ↔ logical pixel conversions ────────────────────────────────────
    @staticmethod
    def freq_to_lx(freq, cf, sr, logical_w):
        return ((freq - cf) / sr + 0.5) * logical_w
    @staticmethod
    def lx_to_freq(lx, cf, sr, logical_w):
        return cf + (lx / logical_w - 0.5) * sr
    @staticmethod
    def row_to_ly(row, row_h):
        return row * row_h
    @staticmethod
    def ly_to_row(ly, row_h):
        return ly / row_h

    def to_logical_rect(self, fft_size, row_h, cf, sr, logical_w) -> QRectF:
        """Return annotation bounding box in logical (zoom=1) pixel space."""
        a  = self.ann
        ss = a.get("core:sample_start", 0)
        sc = a.get("core:sample_count", fft_size)
        fl = a.get("core:freq_lower_edge", cf - sr/2)
        fu = a.get("core:freq_upper_edge", cf + sr/2)
        yt = self.row_to_ly(ss / fft_size, row_h)
        yb = self.row_to_ly((ss + sc) / fft_size, row_h)
        xl = self.freq_to_lx(fl, cf, sr, logical_w)
        xr = self.freq_to_lx(fu, cf, sr, logical_w)
        return QRectF(xl, yt, xr - xl, yb - yt)

    def handles(self, r: QRectF) -> dict:
        cx, cy = r.center().x(), r.center().y()
        return {"tl":QPointF(r.left(),r.top()),   "tr":QPointF(r.right(),r.top()),
                "bl":QPointF(r.left(),r.bottom()),"br":QPointF(r.right(),r.bottom()),
                "tm":QPointF(cx,r.top()),          "bm":QPointF(cx,r.bottom()),
                "lm":QPointF(r.left(),cy),         "rm":QPointF(r.right(),cy)}

    def handle_rect(self, pt: QPointF) -> QRectF:
        s = HANDLE_SIZE
        return QRectF(pt.x()-s/2, pt.y()-s/2, s, s)

    def hit_handle(self, pos: QPointF, r: QRectF) -> str | None:
        for n, pt in self.handles(r).items():
            if self.handle_rect(pt).contains(pos): return n
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  AXIS WIDGETS  (now transform-aware)
# ═══════════════════════════════════════════════════════════════════════════════

class FreqAxisWidget(QWidget):
    """
    Draws a frequency axis that reflects the current ViewTransform.
    Width is always forced to match the viewport width of the scroll area.
    """
    HEIGHT = 36
    def __init__(self, p=None):
        super().__init__(p)
        self.setFixedHeight(self.HEIGHT)
        self.cf = 0.0; self.sr = 1.0
        self._vt: ViewTransform | None = None
        self._logical_w = 1.0

    def set_params(self, cf, sr, logical_w, vt: ViewTransform):
        self.cf = cf; self.sr = sr
        self._logical_w = logical_w
        self._vt = vt
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(25, 25, 35))
        p.setPen(QColor(200, 200, 200))
        p.setFont(QFont("Courier", 7))
        w = self.width()
        vt = self._vt
        n = 9
        for i in range(n):
            # fraction across the *viewport* width
            frac = i / (n - 1)
            vx   = frac * w
            # back-project viewport x → logical x → frequency
            if vt is not None:
                lx, _ = vt.v2l(vx, 0)
            else:
                lx = frac * self._logical_w
            freq = AnnotationItem.lx_to_freq(lx, self.cf, self.sr, self._logical_w)
            xi   = int(vx)
            lbl  = (f"{freq/1e9:.3f}G" if abs(freq) >= 1e9 else
                    f"{freq/1e6:.3f}M" if abs(freq) >= 1e6 else
                    f"{freq/1e3:.1f}k" if abs(freq) >= 1e3 else f"{freq:.0f}")
            p.drawLine(xi, 0, xi, 6)
            fm = p.fontMetrics(); tw = fm.horizontalAdvance(lbl)
            p.drawText(max(0, min(xi - tw//2, w - tw)), self.HEIGHT - 4, lbl)


class TimeAxisWidget(QWidget):
    """Time axis; reflects ViewTransform pan_y / zoom."""
    WIDTH = 72
    def __init__(self, p=None):
        super().__init__(p)
        self.setFixedWidth(self.WIDTH)
        self.sr = 1.0; self.fft_size = FFT_SIZE; self.row_h = 1
        self._vt: ViewTransform | None = None
        self._view_h = 400

    def update_params(self, sr, fft_size, row_h, vt: ViewTransform, view_h):
        self.sr = sr; self.fft_size = fft_size; self.row_h = row_h
        self._vt = vt; self._view_h = view_h
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(25, 25, 35))
        p.setPen(QColor(200, 200, 200))
        p.setFont(QFont("Courier", 7))
        n = 8
        for i in range(n + 1):
            frac = i / n
            vy   = frac * self._view_h
            # viewport y → logical y → time
            if self._vt is not None:
                _, ly = self._vt.v2l(0, vy)
            else:
                ly = vy
            t = ly / self.row_h * self.fft_size / self.sr
            lbl = (f"{t:.3f}s" if t >= 1 else
                   f"{t*1e3:.2f}ms" if t >= 1e-3 else
                   f"{t*1e6:.1f}µs")
            py = int(vy)
            p.drawLine(self.WIDTH - 5, py, self.WIDTH, py)
            p.drawText(QRect(0, py - 8, self.WIDTH - 7, 16),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, lbl)


# ═══════════════════════════════════════════════════════════════════════════════
#  WATERFALL VIEWPORT  (replaces WaterfallCanvas + QScrollArea)
#
#  A single QWidget that owns the ViewTransform, renders tiles scaled/panned,
#  draws annotations in viewport space, and handles all mouse interaction.
#  The outer QScrollArea is removed; scrolling is replaced by pan/zoom.
# ═══════════════════════════════════════════════════════════════════════════════

class WaterfallViewport(QWidget):
    """
    Self-contained waterfall viewer with pan & zoom.

    Coordinate spaces
    -----------------
    Logical space  : pixel grid at zoom=1, full dataset.
                     logical_w = fft_size (or viewport width on load)
                     logical_h = total_rows * row_h
    Viewport space : what is painted on this widget (== widget size).

    All SigMF annotations live in logical space.  The ViewTransform maps
    between the two spaces.

    Pan   : middle-button drag  OR  Ctrl+left-drag
    Zoom  : scroll wheel (isotropic, centred on cursor)
    Select: left click / drag handles  (MODE_SELECT)
    Draw  : left drag                  (MODE_CREATE)
    """
    status_msg         = pyqtSignal(str)
    annotation_changed = pyqtSignal()
    transform_changed  = pyqtSignal()   # axes listen to this

    MODE_SELECT = "select"
    MODE_CREATE = "create"

    def __init__(self, cmd_stack: CommandStack, parent=None):
        super().__init__(parent)
        self.cmd_stack  = cmd_stack
        self.dataset: SigMFDataset | None = None
        self.fft_size   = FFT_SIZE
        self.cmap_name  = "Viridis"
        self.vmin=-60.; self.vmax=-20.
        self.row_h = 1; self.total_rows = 0
        self._logical_w = FFT_SIZE   # updated on load / resize

        self.vt = ViewTransform()

        self._tile_cache: dict[int, QPixmap] = {}
        self._pending:    set[int] = set()
        self._threads:    list[TileThread] = []

        self.annotations:  list[AnnotationItem] = []
        self.selected_idx: int | None = None

        # Overlap cycling: list of annotation indices under last click point
        self._overlap_stack: list[int] = []
        self._overlap_ptr:   int = 0

        # Drag state
        self._drag_handle:  str | None = None
        self._drag_ann_idx: int | None = None
        self._drag_origin_v: QPointF | None = None   # viewport coords
        self._drag_rect0_l:  QRectF  | None = None   # logical rect at drag start
        self._drag_old_ann:  dict | None = None

        # Pan state
        self._panning        = False
        self._pan_origin_v:  QPointF | None = None

        # New-box draw state  (viewport coords during drag, logical on commit)
        self._mode = "select"
        self._nb_start_v: QPointF | None = None
        self._nb_rect_v:  QRectF  | None = None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # ── public API ────────────────────────────────────────────────────────────
    def load_dataset(self, ds):
        self._abort_threads()
        self._tile_cache.clear(); self._pending.clear()
        self.dataset = ds
        self.total_rows = math.ceil(ds.total_samples / self.fft_size)
        self._logical_w = max(self.width(), self.fft_size)
        self.row_h = 1
        self.vt.logical_w = self._logical_w
        self.vt.logical_h = max(self.total_rows * self.row_h, 1.0)
        self.vt.reset()  # full reset to top on fresh load
        self._rebuild_annotations()
        self.update()

    def set_fft_size(self, n):
        self.fft_size = n
        if self.dataset: self.load_dataset(self.dataset)

    def set_colormap(self, name):
        self.cmap_name = name
        self._tile_cache.clear(); self._pending.clear(); self.update()

    def set_vrange(self, vmin, vmax):
        self.vmin = vmin; self.vmax = vmax
        self._tile_cache.clear(); self._pending.clear(); self.update()

    def set_mode(self, mode):
        self._mode = mode; self._nb_start_v = None; self._nb_rect_v = None
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor if mode == self.MODE_CREATE
                               else Qt.CursorShape.ArrowCursor))

    def logical_w(self): return self._logical_w

    def visible_sample_range(self) -> tuple[int, int]:
        """Return (sample_start, sample_count) for the currently visible region."""
        vr = self.vt.visible_logical_rect(self.width(), self.height())
        row_top    = max(0, int(vr.top()  / self.row_h))
        row_bottom = min(self.total_rows, math.ceil(vr.bottom() / self.row_h))
        ss = row_top * self.fft_size
        sc = max(self.fft_size, (row_bottom - row_top) * self.fft_size)
        return ss, sc

    def add_detection_results(self, results: list[DetectionResult]):
        anns = []
        for r in results:
            a: dict = {
                "core:sample_start":    r.sample_start,
                "core:sample_count":    r.sample_count,
                "core:freq_lower_edge": r.freq_lower,
                "core:freq_upper_edge": r.freq_upper,
            }
            if r.label: a["core:label"] = r.label
            anns.append(a)
        if anns:
            self.cmd_stack.push(BulkAddAnnotationsCmd(self.dataset, anns))
            self._rebuild_annotations()
            self.annotation_changed.emit()
            self.update()

    def reset_zoom(self):
        self.vt.reset(self.width(), self.height())
        self.transform_changed.emit(); self.update()

    # ── tiles ─────────────────────────────────────────────────────────────────
    def _tile_for_row(self, row): return (row // TILE_ROWS) * TILE_ROWS

    def _request_tile(self, tile_row):
        if tile_row in self._tile_cache or tile_row in self._pending or not self.dataset:
            return
        rows = min(TILE_ROWS, self.total_rows - tile_row)
        if rows <= 0: return
        self._pending.add(tile_row)
        w = TileWorker(self.dataset, self.fft_size, self.cmap_name,
                       self.vmin, self.vmax, tile_row, rows)
        w.tile_ready.connect(self._on_tile_ready)
        t = TileThread(w)
        t.finished.connect(lambda th=t: self._threads.remove(th) if th in self._threads else None)
        self._threads.append(t); t.start()

    def _on_tile_ready(self, row_start, img):
        self._pending.discard(row_start)
        self._tile_cache[row_start] = QPixmap.fromImage(img)
        self.update()

    def _abort_threads(self):
        for t in self._threads: t.quit(); t.wait(300)
        self._threads.clear(); self._pending.clear()

    # ── annotations ───────────────────────────────────────────────────────────
    def _rebuild_annotations(self):
        self.annotations = ([AnnotationItem(a, i)
                             for i, a in enumerate(self.dataset.annotations)]
                            if self.dataset else [])

    def _ann_logical_rect(self, item: AnnotationItem) -> QRectF:
        if not self.dataset: return QRectF()
        return item.to_logical_rect(self.fft_size, self.row_h,
                                    self.dataset.center_freq,
                                    self.dataset.sample_rate,
                                    self._logical_w)

    def _ann_viewport_rect(self, item: AnnotationItem) -> QRectF:
        return self.vt.l2v_rect(self._ann_logical_rect(item))

    def _logical_rect_to_sigmf(self, lr: QRectF):
        """Convert a logical-space QRectF to SigMF annotation fields."""
        ds = self.dataset; lr = lr.normalized()
        ss = int(round((lr.top()  / self.row_h) * self.fft_size))
        se = int(round((lr.bottom() / self.row_h) * self.fft_size))
        sc = max(self.fft_size, se - ss)
        fl = AnnotationItem.lx_to_freq(lr.left(),  ds.center_freq, ds.sample_rate, self._logical_w)
        fu = AnnotationItem.lx_to_freq(lr.right(), ds.center_freq, ds.sample_rate, self._logical_w)
        return ss, sc, min(fl, fu), max(fl, fu)

    def _hit_annotations_at(self, vpos: QPointF) -> list[int]:
        """Return indices of all annotations whose viewport rect contains vpos."""
        hits = []
        lpos_x, lpos_y = self.vt.v2l(vpos.x(), vpos.y())
        lpos = QPointF(lpos_x, lpos_y)
        for i, item in enumerate(self.annotations):
            if self._ann_logical_rect(item).contains(lpos):
                hits.append(i)
        return hits

    # ── paint ─────────────────────────────────────────────────────────────────
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        vw, vh = self.width(), self.height()

        if not self.dataset:
            painter.fillRect(self.rect(), QColor(20, 20, 30))
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No dataset loaded")
            return

        # Visible logical rect → which tile rows do we need?
        vr = self.vt.visible_logical_rect(vw, vh)
        row_top    = max(0, int(vr.top() / self.row_h))
        row_bottom = min(self.total_rows, math.ceil(vr.bottom() / self.row_h))

        # Background
        painter.fillRect(self.rect(), QColor(15, 15, 25))

        tile_row = self._tile_for_row(row_top)
        while tile_row < row_bottom:
            t_rows  = min(TILE_ROWS, self.total_rows - tile_row)
            # logical rect of this tile
            l_top   = tile_row * self.row_h
            l_bot   = (tile_row + t_rows) * self.row_h
            l_left  = 0.0
            l_right = self._logical_w
            # viewport rect
            vx1, vy1 = self.vt.l2v(l_left,  l_top)
            vx2, vy2 = self.vt.l2v(l_right, l_bot)
            vr_tile  = QRectF(vx1, vy1, vx2 - vx1, vy2 - vy1)

            if tile_row in self._tile_cache:
                pm = self._tile_cache[tile_row]
                painter.drawPixmap(vr_tile.toRect(), pm, pm.rect())
            else:
                painter.fillRect(vr_tile.toRect(), QColor(15, 15, 25))
                painter.setPen(QColor(60, 60, 80))
                painter.drawText(vr_tile.toRect(),
                                 Qt.AlignmentFlag.AlignCenter, "Loading…")
                self._request_tile(tile_row)
            tile_row += TILE_ROWS

        # Draw annotations (in viewport space)
        for i, item in enumerate(self.annotations):
            vr_ann = self._ann_viewport_rect(item)
            if not QRectF(self.rect()).intersects(vr_ann): continue
            sel = (i == self.selected_idx)
            col = SEL_COLOR if sel else ANN_COLOR
            painter.setPen(QPen(col, 1.5))
            painter.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), 30)))
            painter.drawRect(vr_ann)
            lbl = item.label()
            if lbl:
                painter.setPen(col)
                fm = painter.fontMetrics(); tr = fm.boundingRect(lbl)
                tx = vr_ann.left() + 3; ty = vr_ann.top() - 3
                painter.fillRect(QRectF(tx-1, ty-tr.height(), tr.width()+4, tr.height()+2),
                                 QColor(0, 0, 0, 160))
                painter.drawText(QPointF(tx, ty), lbl)
            if sel:
                painter.setPen(QPen(SEL_COLOR, 1))
                painter.setBrush(QBrush(SEL_COLOR))
                for pt in item.handles(vr_ann).values():
                    painter.drawRect(item.handle_rect(pt))

        # Stack-badge: if multiple annotations at hover point, show count
        if len(self._overlap_stack) > 1:
            # show badge near selected annotation
            if self.selected_idx is not None and self.selected_idx < len(self.annotations):
                vr_sel = self._ann_viewport_rect(self.annotations[self.selected_idx])
                badge_txt = f"⊕ {len(self._overlap_stack)} stacked — click again or Z to cycle"
                painter.setPen(QColor(255, 255, 255))
                painter.setBrush(QColor(40, 40, 80, 200))
                fm = painter.fontMetrics()
                bw = fm.horizontalAdvance(badge_txt) + 8; bh = fm.height() + 4
                bx = vr_sel.right() + 4; by = vr_sel.top()
                bx = min(bx, vw - bw - 2)
                painter.drawRoundedRect(QRectF(bx, by, bw, bh), 3, 3)
                painter.drawText(QPointF(bx + 4, by + bh - 4), badge_txt)

        # New-box rubber-band (viewport coords)
        if self._nb_rect_v and not self._nb_rect_v.isNull():
            painter.setPen(QPen(QColor(0, 255, 128), 1.5, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(QColor(0, 255, 128, 25)))
            painter.drawRect(self._nb_rect_v.normalized())

        # Zoom level indicator (top-right)
        if self.vt.zoom != 1.0:
            zt = f"  {self.vt.zoom:.2f}×  "
            painter.setPen(QColor(220, 220, 220))
            painter.setBrush(QColor(0, 0, 0, 140))
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(zt); th = fm.height()
            painter.drawRect(QRect(vw - tw - 4, 4, tw + 4, th + 4))
            painter.drawText(QPointF(vw - tw - 2, 4 + th), zt)

    # ── mouse ─────────────────────────────────────────────────────────────────
    def _is_pan_trigger(self, event) -> bool:
        """Middle button or Ctrl+left."""
        return (event.button() == Qt.MouseButton.MiddleButton or
                (event.button() == Qt.MouseButton.LeftButton and
                 event.modifiers() & Qt.KeyboardModifier.ControlModifier))

    def _is_pan_move(self, event) -> bool:
        return (event.buttons() & Qt.MouseButton.MiddleButton or
                (event.buttons() & Qt.MouseButton.LeftButton and
                 event.modifiers() & Qt.KeyboardModifier.ControlModifier))

    def mousePressEvent(self, event):
        if not self.dataset: return
        vpos = QPointF(event.position())

        # ── pan ──────────────────────────────────────────────────────────────
        if self._is_pan_trigger(event):
            self._panning = True; self._pan_origin_v = vpos
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            return

        if event.button() == Qt.MouseButton.RightButton:
            hit_list = self._hit_annotations_at(vpos)
            if hit_list:
                # Prefer the already-selected annotation if it's under the cursor,
                # so right-click → Edit never accidentally cycles past the selection.
                if self.selected_idx in hit_list:
                    target = self.selected_idx
                else:
                    target = hit_list[0]
                    self.selected_idx = target
                self.update()
                self._context_menu(target, event.globalPosition().toPoint())
            return

        if event.button() != Qt.MouseButton.LeftButton: return

        # ── draw mode ────────────────────────────────────────────────────────
        if self._mode == self.MODE_CREATE:
            self._nb_start_v = vpos; self._nb_rect_v = QRectF(vpos, vpos)
            return

        # ── select / drag ────────────────────────────────────────────────────
        # Check handles on currently selected annotation first
        if self.selected_idx is not None:
            item = self.annotations[self.selected_idx]
            vr   = self._ann_viewport_rect(item)
            h    = item.hit_handle(vpos, vr)
            if h:
                self._start_drag(h, self.selected_idx, vpos)
                return

        # Hit-test all annotations at this point
        hits = self._hit_annotations_at(vpos)
        if hits:
            # Maintain overlap stack for cycling
            same_spot = (set(hits) == set(self._overlap_stack))
            if same_spot and len(hits) > 1:
                self._overlap_ptr = (self._overlap_ptr + 1) % len(self._overlap_stack)
                chosen = self._overlap_stack[self._overlap_ptr]
            else:
                self._overlap_stack = hits
                self._overlap_ptr   = 0
                chosen              = hits[0]
            self.selected_idx = chosen
            self._start_drag("move", chosen, vpos)
        else:
            self.selected_idx   = None
            self._overlap_stack = []
            self._overlap_ptr   = 0
        self.update()

    def _start_drag(self, handle: str, ann_idx: int, vpos: QPointF):
        item = self.annotations[ann_idx]
        self._drag_handle    = handle
        self._drag_ann_idx   = ann_idx
        self._drag_origin_v  = vpos
        self._drag_rect0_l   = QRectF(self._ann_logical_rect(item))
        self._drag_old_ann   = copy.deepcopy(self.dataset.annotations[ann_idx])

    def mouseMoveEvent(self, event):
        if not self.dataset: return
        vpos = QPointF(event.position())

        # ── panning ──────────────────────────────────────────────────────────
        if self._panning and self._pan_origin_v:
            dx = vpos.x() - self._pan_origin_v.x()
            dy = vpos.y() - self._pan_origin_v.y()
            self.vt.pan_by(dx, dy)
            self._pan_origin_v = vpos
            self.transform_changed.emit(); self.update(); return

        # ── new-box draw ──────────────────────────────────────────────────────
        if self._mode == self.MODE_CREATE and self._nb_start_v:
            x0, y0 = self._nb_start_v.x(), self._nb_start_v.y()
            self._nb_rect_v = QRectF(min(x0, vpos.x()), min(y0, vpos.y()),
                                     abs(vpos.x()-x0), abs(vpos.y()-y0))
            self.update(); return

        # ── drag handle / move ────────────────────────────────────────────────
        if self._drag_handle and self._drag_origin_v and self._drag_rect0_l is not None:
            # delta in logical space
            vlx0, vly0 = self.vt.v2l(self._drag_origin_v.x(), self._drag_origin_v.y())
            vlx1, vly1 = self.vt.v2l(vpos.x(), vpos.y())
            dlx = vlx1 - vlx0; dly = vly1 - vly0

            r = QRectF(self._drag_rect0_l); h = self._drag_handle
            if   h == "move": r.translate(dlx, dly)
            elif h == "tl":   r.setTopLeft(r.topLeft()       + QPointF(dlx, dly))
            elif h == "tr":   r.setTopRight(r.topRight()     + QPointF(dlx, dly))
            elif h == "bl":   r.setBottomLeft(r.bottomLeft() + QPointF(dlx, dly))
            elif h == "br":   r.setBottomRight(r.bottomRight()+ QPointF(dlx, dly))
            elif h == "tm":   r.setTop(r.top()    + dly)
            elif h == "bm":   r.setBottom(r.bottom() + dly)
            elif h == "lm":   r.setLeft(r.left()  + dlx)
            elif h == "rm":   r.setRight(r.right() + dlx)
            r = r.normalized()

            # Minimum logical size: MIN_BOX_PX logical pixels
            if r.width() >= MIN_BOX_PX and r.height() >= MIN_BOX_PX:
                ss, sc, fl, fu = self._logical_rect_to_sigmf(r)
                idx = self._drag_ann_idx
                if idx is not None:
                    self.dataset.annotations[idx].update({
                        "core:sample_start":    ss, "core:sample_count": sc,
                        "core:freq_lower_edge": fl, "core:freq_upper_edge": fu})
                    self.annotations[idx].ann = self.dataset.annotations[idx]
                self.update()
                self.status_msg.emit(
                    f"ss={ss:,}  sc={sc:,}  fl={fl/1e6:.3f}MHz  fu={fu/1e6:.3f}MHz")
            return

        # ── hover cursor ──────────────────────────────────────────────────────
        if self._mode == self.MODE_SELECT:
            cur = Qt.CursorShape.ArrowCursor
            if self.selected_idx is not None and self.selected_idx < len(self.annotations):
                item = self.annotations[self.selected_idx]
                vr   = self._ann_viewport_rect(item)
                h    = item.hit_handle(vpos, vr)
                if h: cur = item.HANDLE_CURSORS.get(h, Qt.CursorShape.ArrowCursor)
                elif vr.contains(vpos): cur = Qt.CursorShape.SizeAllCursor
            self.setCursor(QCursor(cur))

    def mouseReleaseEvent(self, event):
        if not self.dataset: return
        vpos = QPointF(event.position())

        if self._panning:
            self._panning = False; self._pan_origin_v = None
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor
                                   if self._mode == self.MODE_SELECT
                                   else Qt.CursorShape.CrossCursor))
            return

        if event.button() != Qt.MouseButton.LeftButton: return

        # ── commit new box ────────────────────────────────────────────────────
        if self._mode == self.MODE_CREATE and self._nb_rect_v:
            r_v = self._nb_rect_v.normalized()
            if r_v.width() >= MIN_BOX_PX and r_v.height() >= MIN_BOX_PX:
                # convert viewport rect → logical rect → SigMF
                lr = self.vt.v2l_rect(r_v)
                ss, sc, fl, fu = self._logical_rect_to_sigmf(lr)
                label, ok = QInputDialog.getText(self, "New Annotation", "Label (optional):")
                if ok:
                    ann = {"core:sample_start": ss, "core:sample_count": sc,
                           "core:freq_lower_edge": fl, "core:freq_upper_edge": fu}
                    if label.strip(): ann["core:label"] = label.strip()
                    self.cmd_stack.push(AddAnnotationCmd(self.dataset, ann))
                    self._rebuild_annotations()
                    self.selected_idx = len(self.annotations) - 1
                    self.annotation_changed.emit()
            self._nb_start_v = None; self._nb_rect_v = None; self.update(); return

        # ── commit drag ───────────────────────────────────────────────────────
        if self._drag_handle:
            idx = self._drag_ann_idx
            if idx is not None and self._drag_old_ann is not None:
                new_ann  = self.dataset.annotations[idx]
                keys     = ("core:sample_start","core:sample_count",
                            "core:freq_lower_edge","core:freq_upper_edge")
                new_vals = {k: new_ann[k] for k in keys}
                old_vals = {k: self._drag_old_ann.get(k) for k in keys}
                if new_vals != old_vals:
                    self.dataset.annotations[idx].update(old_vals)
                    self.cmd_stack.push(EditAnnotationCmd(self.dataset, idx, new_vals))
            self._rebuild_annotations()
            self.annotation_changed.emit()
            self._drag_handle = None; self._drag_ann_idx = None
            self._drag_origin_v = None; self._drag_rect0_l = None
            self._drag_old_ann  = None
            self.update()

    def mouseDoubleClickEvent(self, event):
        if not self.dataset: return
        hits = self._hit_annotations_at(QPointF(event.position()))
        if hits: self._edit_label(hits[0])

    def wheelEvent(self, event: QWheelEvent):
        if not self.dataset: return
        delta = event.angleDelta().y()
        if delta == 0: return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Ctrl+scroll → zoom centred on cursor
            factor = 1.15 if delta > 0 else 1.0 / 1.15
            vpos   = event.position()
            self.vt.zoom_at(vpos.x(), vpos.y(), factor)
        else:
            # Plain scroll → pan vertically (proportional to current zoom)
            # delta > 0 = wheel up = move view up (pan_y decreases)
            step = 60.0 / self.vt.zoom   # logical pixels to move
            self.vt.pan_by(0, step if delta > 0 else -step)
        self.transform_changed.emit(); self.update()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self.selected_idx is not None:
                self._delete_annotation(self.selected_idx)
        elif event.key() == Qt.Key.Key_Z and not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self._cycle_overlap()
        elif event.key() == Qt.Key.Key_Home:
            self.reset_zoom()

    def _cycle_overlap(self):
        """Tab through overlapping annotations at the last click point."""
        if len(self._overlap_stack) < 2: return
        self._overlap_ptr = (self._overlap_ptr + 1) % len(self._overlap_stack)
        self.selected_idx = self._overlap_stack[self._overlap_ptr]
        self.update()

    def _context_menu(self, idx, gpos):
        self.selected_idx = idx; self.update()
        menu = QMenu(self)
        a_edt = menu.addAction("Edit Label…"); a_del = menu.addAction("Delete")
        act = menu.exec(gpos)
        if act == a_edt: self._edit_label(idx)
        elif act == a_del: self._delete_annotation(idx)

    def _edit_label(self, idx):
        label, ok = QInputDialog.getText(self, "Edit Label", "Label:",
                                         text=self.annotations[idx].label())
        if ok:
            self.cmd_stack.push(EditAnnotationCmd(
                self.dataset, self.annotations[idx].index, {"core:label": label.strip()}))
            self._rebuild_annotations(); self.selected_idx = idx
            self.annotation_changed.emit(); self.update()

    def _delete_annotation(self, idx):
        self.cmd_stack.push(DeleteAnnotationCmd(
            self.dataset, self.annotations[idx].index))
        self._rebuild_annotations(); self.selected_idx = None
        if idx in self._overlap_stack: self._overlap_stack.clear()
        self.annotation_changed.emit(); self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.dataset:
            # Logical width tracks the viewport width so freq axis stays correct
            self._logical_w = self.width()
            self.vt.logical_w = self._logical_w
            self.transform_changed.emit()

    def closeEvent(self, event):
        self._abort_threads(); super().closeEvent(event)


# ═══════════════════════════════════════════════════════════════════════════════
#  DETECTOR BACKGROUND WORKER  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class DetectorWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    error    = pyqtSignal(str)
    CHUNK_SAMPLES = 4_000_000

    def __init__(self, detector_cls, dataset, sample_start, sample_count, fft_size, kwargs):
        super().__init__()
        self.det_cls=detector_cls; self.dataset=dataset
        self.sample_start=sample_start; self.sample_count=sample_count
        self.fft_size=fft_size; self.kwargs=kwargs; self._abort=False

    def abort(self): self._abort = True

    def run(self):
        try:
            det      = self.det_cls()
            fs       = self.fft_size
            sr       = self.dataset.sample_rate
            cf       = self.dataset.center_freq
            meta     = self.dataset.meta
            chunk_sz = max(fs, (self.CHUNK_SAMPLES // fs) * fs)
            total    = self.sample_count; done = 0
            results: list[DetectionResult] = []
            pos = self.sample_start; end = self.sample_start + total
            while pos < end and not self._abort:
                count   = min(chunk_sz, end - pos)
                samples = self.dataset.read_samples(pos, count)
                if samples.size == 0: break
                ctx = DetectionContext(samples=samples, sample_rate=sr, center_freq=cf,
                                       sample_start=pos, fft_size=fs, sigmf_meta=meta)
                results.extend(det.run(ctx, **self.kwargs))
                done += len(samples)
                self.progress.emit(min(99, int(done / total * 100)))
                pos += len(samples)
            if not self._abort:
                self.progress.emit(100); self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  DETECTOR PANEL  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class DetectorPanel(QWidget):
    run_visible = pyqtSignal(type, dict)
    run_all     = pyqtSignal(type, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(220); self.setMaximumWidth(320)
        self._detectors  = get_all_detectors()
        self._param_widgets: dict[str, QWidget] = {}
        self._current_cls: type[BaseDetector] | None = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self); root.setContentsMargins(6,6,6,6)
        root.addWidget(QLabel("Energy Detection", font=QFont("Arial",10,QFont.Weight.Bold)))
        grp = QGroupBox("Algorithm"); gl = QVBoxLayout(grp)
        self.det_combo = QComboBox()
        for cls in self._detectors: self.det_combo.addItem(cls.NAME, cls)
        self.det_combo.currentIndexChanged.connect(self._on_detector_changed)
        gl.addWidget(self.det_combo); root.addWidget(grp)
        self.param_group  = QGroupBox("Parameters")
        self.param_layout = QFormLayout(self.param_group)
        self.param_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        root.addWidget(self.param_group)
        btn_vis = QPushButton("▶  Run on Visible")
        btn_vis.setToolTip("Run detector on currently visible tiles only")
        btn_vis.clicked.connect(self._emit_visible); root.addWidget(btn_vis)
        btn_all = QPushButton("▶▶  Run on Entire Dataset")
        btn_all.setToolTip("Run detector on all samples in the file")
        btn_all.clicked.connect(self._emit_all); root.addWidget(btn_all)
        prog_row = QHBoxLayout()
        self.progress_bar = QProgressBar(); self.progress_bar.setRange(0,100)
        self.progress_bar.setValue(0); self.progress_bar.setVisible(False)
        prog_row.addWidget(self.progress_bar)
        self.btn_abort = QPushButton("✕"); self.btn_abort.setFixedWidth(28)
        self.btn_abort.setToolTip("Abort running detector"); self.btn_abort.setVisible(False)
        prog_row.addWidget(self.btn_abort); root.addLayout(prog_row)
        self.status_lbl = QLabel(""); self.status_lbl.setWordWrap(True)
        root.addWidget(self.status_lbl); root.addStretch()
        if self._detectors: self._on_detector_changed(0)

    def _on_detector_changed(self, idx):
        cls = self.det_combo.itemData(idx); self._current_cls = cls
        self._param_widgets.clear()
        while self.param_layout.rowCount(): self.param_layout.removeRow(0)
        for p in cls.PARAMS:
            w = self._make_widget(p)
            if w:
                if p.tooltip: w.setToolTip(p.tooltip)
                self._param_widgets[p.name] = w
                self.param_layout.addRow(p.label + ":", w)

    def _make_widget(self, p):
        if p.kind=="float":
            w=QDoubleSpinBox(); w.setRange(p.minimum,p.maximum)
            w.setSingleStep(p.step); w.setDecimals(p.decimals); w.setValue(float(p.default)); return w
        if p.kind=="int":
            w=QSpinBox(); w.setRange(int(p.minimum),int(p.maximum))
            w.setSingleStep(int(p.step)); w.setValue(int(p.default)); return w
        if p.kind=="str": w=QLineEdit(); w.setText(str(p.default)); return w
        if p.kind=="bool": w=QCheckBox(); w.setChecked(bool(p.default)); return w
        if p.kind=="choice":
            w=QComboBox(); w.addItems(p.choices)
            if p.default in p.choices: w.setCurrentText(str(p.default))
            return w
        return None

    def _collect_kwargs(self):
        cls = self._current_cls
        if not cls: return {}
        out = {}
        for p in cls.PARAMS:
            w = self._param_widgets.get(p.name)
            if w is None: out[p.name]=p.default; continue
            if p.kind=="float": out[p.name]=w.value()
            elif p.kind=="int": out[p.name]=w.value()
            elif p.kind=="str": out[p.name]=w.text()
            elif p.kind=="bool": out[p.name]=w.isChecked()
            elif p.kind=="choice": out[p.name]=w.currentText()
        return out

    def _emit_visible(self):
        if self._current_cls: self.run_visible.emit(self._current_cls, self._collect_kwargs())
    def _emit_all(self):
        if self._current_cls: self.run_all.emit(self._current_cls, self._collect_kwargs())

    def set_running(self, running, abort_cb=None):
        self.progress_bar.setVisible(running); self.btn_abort.setVisible(running)
        if running:
            self.progress_bar.setValue(0)
            if abort_cb:
                try: self.btn_abort.clicked.disconnect()
                except: pass
                self.btn_abort.clicked.connect(abort_cb)

    def set_progress(self, v): self.progress_bar.setValue(v)
    def set_status(self, msg): self.status_lbl.setText(msg)


# ═══════════════════════════════════════════════════════════════════════════════
#  ANNOTATION LIST PANEL  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class AnnotationPanel(QWidget):
    jump_to = pyqtSignal(int)
    def __init__(self, p=None):
        super().__init__(p)
        lay = QVBoxLayout(self); lay.setContentsMargins(4,4,4,4)
        lbl = QLabel("Annotations", font=QFont("Arial",10,QFont.Weight.Bold))
        lay.addWidget(lbl)
        self.lw = QListWidget()
        self.lw.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.lw.itemDoubleClicked.connect(
            lambda li: self.jump_to.emit(li.data(Qt.ItemDataRole.UserRole) or 0))
        lay.addWidget(self.lw)
        self.setMinimumWidth(190); self.setMaximumWidth(270)

    def refresh(self, annotations, fft_size):
        self.lw.clear()
        for item in annotations:
            ss  = item.ann.get("core:sample_start", 0)
            sc  = item.ann.get("core:sample_count", 0)
            lbl = item.label() or "(no label)"
            li  = QListWidgetItem(f"[{ss:,}–{ss+sc:,}]  {lbl}")
            li.setData(Qt.ItemDataRole.UserRole, ss // fft_size)
            self.lw.addItem(li)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SigMF Waterfall Annotator")
        self.resize(1380, 800)
        self.dataset: SigMFDataset | None = None
        self._det_thread: QThread | None = None
        self._det_worker: DetectorWorker | None = None

        self.cmd_stack = CommandStack()
        self.cmd_stack.changed = self._on_stack_changed
        self._scrollbar_syncing = False   # must exist before _build_ui

        self._build_ui()
        self._bind_shortcuts()

    def _build_ui(self):
        # ── toolbar ──────────────────────────────────────────────────────────
        tb = QToolBar(); tb.setMovable(False); self.addToolBar(tb)

        def tbtn(label, slot, checkable=False, enabled=True):
            b = QPushButton(label); b.setFixedHeight(28)
            b.setCheckable(checkable); b.setEnabled(enabled)
            b.clicked.connect(slot); tb.addWidget(b); return b

        tbtn("📂 Open SigMF…", self.open_file)
        tb.addSeparator()
        self.btn_save = tbtn("💾 Save", self.save_file, enabled=False)
        tb.addSeparator()
        tb.addWidget(QLabel("  Mode: "))
        self.btn_sel = tbtn("🔲 Select", lambda: self._set_mode("select"), True)
        self.btn_sel.setChecked(True)
        self.btn_drw = tbtn("✏️ Draw",   lambda: self._set_mode("create"), True)
        tb.addSeparator()
        self.btn_undo = tbtn("↩ Undo", self._undo, enabled=False)
        self.btn_redo = tbtn("↪ Redo", self._redo, enabled=False)
        tb.addSeparator()
        # Reset zoom button
        self.btn_zoom_reset = tbtn("⌖ Reset Zoom", self._reset_zoom)
        tb.addSeparator()
        tb.addWidget(QLabel("  FFT: "))
        self.fft_spin = QSpinBox(); self.fft_spin.setRange(64, 8192)
        self.fft_spin.setValue(FFT_SIZE); self.fft_spin.setSingleStep(64)
        self.fft_spin.setFixedWidth(80)
        self.fft_spin.editingFinished.connect(self._on_fft)
        tb.addWidget(self.fft_spin); tb.addSeparator()
        tb.addWidget(QLabel("  Colormap: "))
        self.cmap_cb = QComboBox(); self.cmap_cb.addItems(list(COLORMAPS))
        self.cmap_cb.currentTextChanged.connect(lambda n: self.viewport.set_colormap(n))
        tb.addWidget(self.cmap_cb); tb.addSeparator()
        tb.addWidget(QLabel("  dB min: "))
        self.vmin_sp = QDoubleSpinBox(); self.vmin_sp.setRange(-200, 0)
        self.vmin_sp.setValue(-60); self.vmin_sp.setFixedWidth(70)
        self.vmin_sp.editingFinished.connect(self._on_vrange); tb.addWidget(self.vmin_sp)
        tb.addWidget(QLabel("  max: "))
        self.vmax_sp = QDoubleSpinBox(); self.vmax_sp.setRange(-200, 0)
        self.vmax_sp.setValue(-20); self.vmax_sp.setFixedWidth(70)
        self.vmax_sp.editingFinished.connect(self._on_vrange); tb.addWidget(self.vmax_sp)

        # ── central layout ────────────────────────────────────────────────────
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        # Use a QSplitter but give the waterfall section strong stretch priority
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left section: time axis + [freq axis / waterfall viewport] ────────
        left = QWidget()
        lh   = QHBoxLayout(left); lh.setContentsMargins(0,0,0,0); lh.setSpacing(0)

        self.time_axis = TimeAxisWidget()
        lh.addWidget(self.time_axis)

        rcol = QWidget(); rv = QVBoxLayout(rcol)
        rv.setContentsMargins(0,0,0,0); rv.setSpacing(0)

        self.freq_axis = FreqAxisWidget()
        rv.addWidget(self.freq_axis)

        # The viewport is now a plain widget (no QScrollArea wrapper).
        # Pan is done by dragging; the time axis reflects vt.pan_y.
        self.viewport = WaterfallViewport(self.cmd_stack)
        self.viewport.status_msg.connect(lambda m: self.statusBar().showMessage(m))
        self.viewport.annotation_changed.connect(self._on_ann_changed)
        self.viewport.transform_changed.connect(self._on_transform_changed)
        self.viewport.transform_changed.connect(self._sync_scrollbar_from_vt)

        # Place viewport + a standalone scrollbar side-by-side.
        # The scrollbar drives vt.pan_y; the viewport handles its own rendering.
        from PyQt6.QtWidgets import QScrollBar
        vp_row = QHBoxLayout()
        vp_row.setContentsMargins(0, 0, 0, 0); vp_row.setSpacing(0)
        vp_row.addWidget(self.viewport, stretch=1)
        self._scrollbar = QScrollBar(Qt.Orientation.Vertical)
        self._scrollbar.setRange(0, 0)
        self._scrollbar.valueChanged.connect(self._on_scrollbar_moved)
        self._scrollbar_syncing = False
        vp_row.addWidget(self._scrollbar)
        rv.addLayout(vp_row, stretch=1)

        lh.addWidget(rcol, stretch=1)
        splitter.addWidget(left)

        # ── Middle: annotation list ───────────────────────────────────────────
        self.ann_panel = AnnotationPanel()
        self.ann_panel.jump_to.connect(self._jump_to_row)
        splitter.addWidget(self.ann_panel)

        # ── Right: detector panel ─────────────────────────────────────────────
        self.det_panel = DetectorPanel()
        self.det_panel.run_visible.connect(self._run_detector_visible)
        self.det_panel.run_all.connect(self._run_detector_all)
        splitter.addWidget(self.det_panel)

        # Stretch factors: waterfall gets all extra space; side panels are fixed-ish
        splitter.setStretchFactor(0, 1)   # waterfall
        splitter.setStretchFactor(1, 0)   # annotation list
        splitter.setStretchFactor(2, 0)   # detector panel
        # Prevent side panels from being dragged to crush the waterfall
        splitter.setCollapsible(0, False)
        splitter.setSizes([860, 230, 250])

        root.addWidget(splitter)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            "Ready — open a .sigmf-meta file.  "
            "Scroll to pan · Ctrl+Scroll to zoom · Middle-drag or Ctrl+drag to pan · Z to cycle overlapping annotations.")

    def _bind_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Z"),       self).activated.connect(self._undo)
        QShortcut(QKeySequence("Ctrl+Y"),       self).activated.connect(self._redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self).activated.connect(self._redo)
        QShortcut(QKeySequence("Ctrl+S"),       self).activated.connect(self.save_file)
        QShortcut(QKeySequence("Home"),         self).activated.connect(self._reset_zoom)

    # ── undo/redo ─────────────────────────────────────────────────────────────
    def _undo(self):
        if not self.cmd_stack.can_undo: return
        self.cmd_stack.undo()
        self.viewport._rebuild_annotations(); self.viewport.selected_idx = None
        self.viewport.update(); self._on_ann_changed()

    def _redo(self):
        if not self.cmd_stack.can_redo: return
        self.cmd_stack.redo()
        self.viewport._rebuild_annotations(); self.viewport.selected_idx = None
        self.viewport.update(); self._on_ann_changed()

    def _on_stack_changed(self):
        self.btn_undo.setEnabled(self.cmd_stack.can_undo)
        self.btn_redo.setEnabled(self.cmd_stack.can_redo)
        ul = self.cmd_stack.undo_label; rl = self.cmd_stack.redo_label
        self.btn_undo.setToolTip(f"Undo: {ul}" if ul else "Nothing to undo")
        self.btn_redo.setToolTip(f"Redo: {rl}" if rl else "Nothing to redo")

    # ── zoom ──────────────────────────────────────────────────────────────────
    def _reset_zoom(self):
        self.viewport.reset_zoom()

    # ── transform changed → update axes ──────────────────────────────────────
    def _on_transform_changed(self):
        if not self.dataset: return
        vt = self.viewport.vt
        self.freq_axis.set_params(
            self.dataset.center_freq, self.dataset.sample_rate,
            self.viewport.logical_w(), vt)
        self.time_axis.update_params(
            self.dataset.sample_rate, self.viewport.fft_size,
            self.viewport.row_h, vt, self.viewport.height())

    def _sync_scrollbar_from_vt(self):
        """Push vt.pan_y into the scrollbar without triggering a feedback loop."""
        if self._scrollbar_syncing or not self.dataset: return
        self._scrollbar_syncing = True
        vt     = self.viewport.vt
        # Range is in logical pixels; page = how many logical px fit in the viewport
        page   = self.viewport.height() / vt.zoom
        total  = vt.logical_h
        self._scrollbar.setRange(0, max(0, int(total - page)))
        self._scrollbar.setPageStep(max(1, int(page)))
        self._scrollbar.setSingleStep(max(1, int(60 / vt.zoom)))
        self._scrollbar.setValue(int(vt.pan_y))
        self._scrollbar_syncing = False

    def _on_scrollbar_moved(self, value: int):
        """Scrollbar dragged by user → update vt.pan_y."""
        if self._scrollbar_syncing or not self.dataset: return
        self.viewport.vt.pan_y = float(value)
        self.viewport.vt._clamp()
        self.viewport.transform_changed.emit()
        self.viewport.update()

    # ── detector ──────────────────────────────────────────────────────────────
    def _run_detector_visible(self, cls, kwargs):
        if not self.dataset: return
        ss, sc = self.viewport.visible_sample_range()
        self._launch_detector(cls, kwargs, ss, sc)

    def _run_detector_all(self, cls, kwargs):
        if not self.dataset: return
        self._launch_detector(cls, kwargs, 0, self.dataset.total_samples)

    def _launch_detector(self, cls, kwargs, ss, sc):
        if self._det_thread and self._det_thread.isRunning():
            QMessageBox.information(self, "Busy", "A detector is already running.")
            return
        self._det_worker = DetectorWorker(
            cls, self.dataset, ss, sc, self.viewport.fft_size, kwargs)
        self._det_thread = QThread()
        self._det_worker.moveToThread(self._det_thread)
        self._det_thread.started.connect(self._det_worker.run)
        self._det_worker.progress.connect(self.det_panel.set_progress)
        self._det_worker.finished.connect(self._on_detection_done)
        self._det_worker.error.connect(self._on_detection_error)
        self._det_worker.finished.connect(self._det_thread.quit)
        self._det_worker.error.connect(self._det_thread.quit)
        self.det_panel.set_running(True, abort_cb=self._abort_detector)
        self.det_panel.set_status("Running…"); self._det_thread.start()

    def _abort_detector(self):
        if self._det_worker: self._det_worker.abort(); self.det_panel.set_status("Aborted.")

    def _on_detection_done(self, results):
        self.det_panel.set_running(False); n = len(results)
        self.det_panel.set_status(f"Done — {n} annotation(s) added.")
        self.statusBar().showMessage(f"Detection complete: {n} result(s).")
        if results:
            self.viewport.add_detection_results(results); self._on_ann_changed()

    def _on_detection_error(self, msg):
        self.det_panel.set_running(False); self.det_panel.set_status(f"Error: {msg}")
        QMessageBox.critical(self, "Detector Error", msg)

    # ── file ──────────────────────────────────────────────────────────────────
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SigMF Metadata File", "", "SigMF Metadata (*.sigmf-meta)")
        if not path: return
        if not path.lower().endswith(".sigmf-meta"):
            QMessageBox.warning(self, "Invalid File", "Please select a .sigmf-meta file.")
            return
        try: ds = SigMFDataset(path)
        except Exception as e: QMessageBox.critical(self, "Load Error", str(e)); return
        self.dataset = ds
        self.cmd_stack.clear()
        self.viewport.set_fft_size(self.fft_spin.value())
        self.viewport.set_colormap(self.cmap_cb.currentText())
        vmin, vmax = self._estimate_vrange(ds)
        self.vmin_sp.setValue(vmin); self.vmax_sp.setValue(vmax)
        self.viewport.set_vrange(vmin, vmax)
        self.viewport.load_dataset(ds)
        # Reset scrollbar range for new dataset
        self._sync_scrollbar_from_vt()
        # Sync axes immediately after load
        self._on_transform_changed()
        self._on_ann_changed()
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
            QMessageBox.critical(self, "Save Error", str(e))

    # ── helpers ───────────────────────────────────────────────────────────────
    def _set_mode(self, mode):
        self.viewport.set_mode(mode)
        self.btn_sel.setChecked(mode == "select")
        self.btn_drw.setChecked(mode == "create")

    def _on_fft(self):
        if self.dataset: self.viewport.set_fft_size(self.fft_spin.value())

    def _on_vrange(self):
        self.viewport.set_vrange(self.vmin_sp.value(), self.vmax_sp.value())

    def _on_ann_changed(self):
        if self.dataset:
            self.ann_panel.refresh(self.viewport.annotations, self.viewport.fft_size)

    def _jump_to_row(self, row):
        """Pan the viewport so the given logical row appears at the top."""
        vt = self.viewport.vt
        vt.pan_y = row * self.viewport.row_h
        vt._clamp()
        self.viewport.transform_changed.emit()
        self.viewport.update()

    def _estimate_vrange(self, ds, probe_rows=64):
        fs = self.fft_spin.value()
        samples = ds.read_samples(0, probe_rows * fs)
        if samples.size < fs: return -60., -20.
        pad = (-len(samples)) % fs
        if pad: samples = np.concatenate([samples, np.zeros(pad, dtype=np.complex64)])
        rows = len(samples) // fs; mat = samples.reshape(rows, fs)
        win = np.hanning(fs).astype(np.float32); wg = np.sum(win**2) / fs
        spec = np.fft.fftshift(np.fft.fft(mat*win, axis=1), axes=1)
        pwr  = 10 * np.log10(np.abs(spec/fs)**2 / wg + 1e-12)
        p2, p98 = float(np.percentile(pwr, 2)), float(np.percentile(pwr, 98))
        m = max((p98 - p2) * 0.15, 2.)
        return round(p2 - m, 1), round(p98 + m, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Axes always match the viewport width — nothing to sync manually;
        # WaterfallViewport.resizeEvent fires transform_changed which updates axes.

    def closeEvent(self, event):
        self.viewport._abort_threads(); super().closeEvent(event)


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    _script_dir = Path(__file__).parent
    _loaded = load_detectors_from_dir(_script_dir / "detectors")
    if _loaded: print(f"[detectors] Loaded: {', '.join(_loaded)}")

    app = QApplication(sys.argv); app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor(35,35,45))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor(220,220,220))
    pal.setColor(QPalette.ColorRole.Base,            QColor(25,25,35))
    pal.setColor(QPalette.ColorRole.Text,            QColor(220,220,220))
    pal.setColor(QPalette.ColorRole.Button,          QColor(50,50,65))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor(220,220,220))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor(42,130,218))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(0,0,0))
    app.setPalette(pal)
    win = MainWindow(); win.show(); sys.exit(app.exec())

if __name__ == "__main__":
    main()
