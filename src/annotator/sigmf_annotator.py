"""
SigMF Waterfall Annotator
Requires: pip install PyQt6 numpy

Layout:
  - X axis: frequency  (left = CF - SR/2, right = CF + SR/2), after fftshift
  - Y axis: time       (top = sample 0, increases downward)
  - Each image row = one FFT window of `fft_size` samples
"""

import sys
import json
import math
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QScrollArea, QMessageBox,
    QInputDialog, QStatusBar, QSplitter, QFrame, QSizePolicy,
    QToolBar, QSpinBox, QDoubleSpinBox, QComboBox,
    QListWidget, QListWidgetItem, QAbstractItemView, QMenu
)
from PyQt6.QtCore import Qt, QRect, QPoint, QPointF, QRectF, QSize, QThread, pyqtSignal, QObject
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont, QPixmap, QImage,
    QCursor, QPalette
)

# ─────────────────────────────────────────────────────────────────────────────
TILE_ROWS      = 512
FFT_SIZE       = 1024
HANDLE_SIZE    = 8
MIN_BOX_PX     = 6
ANN_COLOR      = QColor(255, 220, 0)
SEL_COLOR      = QColor(0, 220, 255)

COLORMAPS: dict[str, np.ndarray] = {}

def _build_lut(name: str) -> np.ndarray:
    t = np.linspace(0, 1, 256)
    if name == "Viridis":
        r = np.interp(t, [0,.25,.5,.75,1], [68, 59, 33, 94,253])
        g = np.interp(t, [0,.25,.5,.75,1], [1,  82,145,201,231])
        b = np.interp(t, [0,.25,.5,.75,1], [84,139,140, 98, 37])
    elif name == "Inferno":
        r = np.interp(t, [0,.25,.5,.75,1], [0,  87,188,249,252])
        g = np.interp(t, [0,.25,.5,.75,1], [0,  16, 55,142,255])
        b = np.interp(t, [0,.25,.5,.75,1], [4, 110, 85,  8,164])
    elif name == "Plasma":
        r = np.interp(t, [0,.25,.5,.75,1], [13, 84,163,229,240])
        g = np.interp(t, [0,.25,.5,.75,1], [8,   2, 44,126,249])
        b = np.interp(t, [0,.25,.5,.75,1], [135,163,164, 56, 33])
    elif name == "Jet":
        r = np.interp(t, [0,.35,.66,.89,1],       [0,  0,255,255,128])
        g = np.interp(t, [0,.12,.38,.64,.91,1],   [0,  0,255,255,  0,  0])
        b = np.interp(t, [0,.11,.34,.65,1],        [128,255,255,  0,  0])
    else:  # Greys
        r = g = b = t * 255
    return np.stack([r, g, b], axis=1).astype(np.uint8)

for _n in ["Viridis","Inferno","Plasma","Jet","Greys"]:
    COLORMAPS[_n] = _build_lut(_n)

# ─────────────────────────────────────────────────────────────────────────────
# SigMF loader
# ─────────────────────────────────────────────────────────────────────────────
class SigMFDataset:
    def __init__(self, meta_path: str):
        self.meta_path = Path(meta_path)
        self.data_path = self.meta_path.with_suffix(".sigmf-data")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Meta not found: {self.meta_path}")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found: {self.data_path}")
        with open(self.meta_path) as f:
            self.meta = json.load(f)
        g = self.meta.get("global", {})
        dt = g.get("core:datatype", "")
        if "cf32" not in dt and "complex64" not in dt:
            raise ValueError(f"Only complex64 (cf32) supported. Got: '{dt}'")
        caps = self.meta.get("captures", [{}])
        self.sample_rate  = float(g.get("core:sample_rate", 1.0))
        self.center_freq  = float(caps[0].get("core:frequency",
                                  g.get("core:frequency", 0.0)))
        self.total_samples = self.data_path.stat().st_size // 8
        self.annotations   = list(self.meta.get("annotations", []))

    def read_samples(self, start: int, count: int) -> np.ndarray:
        start = max(0, min(start, self.total_samples - 1))
        count = min(count, self.total_samples - start)
        if count <= 0:
            return np.array([], dtype=np.complex64)
        with open(self.data_path, "rb") as f:
            f.seek(start * 8)
            raw = f.read(count * 8)
        return np.frombuffer(raw, dtype=np.complex64)

    def save(self):
        self.meta["annotations"] = self.annotations
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f, indent=2)

    def add_annotation(self, sample_start, sample_count,
                       freq_lower, freq_upper, label="") -> dict:
        ann = {
            "core:sample_start":    int(sample_start),
            "core:sample_count":    int(sample_count),
            "core:freq_lower_edge": float(freq_lower),
            "core:freq_upper_edge": float(freq_upper),
        }
        if label:
            ann["core:label"] = label
        self.annotations.append(ann)
        return ann

    def delete_annotation(self, index: int):
        if 0 <= index < len(self.annotations):
            self.annotations.pop(index)

    def update_annotation(self, index: int, **kw):
        if 0 <= index < len(self.annotations):
            self.annotations[index].update(kw)


# ─────────────────────────────────────────────────────────────────────────────
# Background tile renderer
# ─────────────────────────────────────────────────────────────────────────────
class TileWorker(QObject):
    tile_ready = pyqtSignal(int, QImage)   # (first_row, image)

    def __init__(self, ds, fft_size, cmap, vmin, vmax, row_start, row_count):
        super().__init__()
        self.ds = ds; self.fft_size = fft_size; self.cmap = cmap
        self.vmin = vmin; self.vmax = vmax
        self.row_start = row_start; self.row_count = row_count

    def run(self):
        fs = self.fft_size
        samples = self.ds.read_samples(self.row_start * fs, self.row_count * fs)
        if samples.size == 0:
            return
        pad = (-len(samples)) % fs
        if pad:
            samples = np.concatenate([samples, np.zeros(pad, dtype=np.complex64)])
        rows = len(samples) // fs
        mat  = samples.reshape(rows, fs)
        win  = np.hanning(fs).astype(np.float32)
        wg   = np.sum(win**2) / fs
        # fftshift so DC (centre freq) is in the middle column
        spec  = np.fft.fftshift(np.fft.fft(mat * win, axis=1), axes=1)
        power = 10 * np.log10(np.abs(spec / fs)**2 / wg + 1e-12)
        idx   = np.clip((power - self.vmin) / (self.vmax - self.vmin), 0, 1)
        idx   = (idx * 255).astype(np.uint8)
        lut   = COLORMAPS[self.cmap]
        rgb   = lut[idx]                        # (rows, fs, 3)
        # Each row = one FFT; row width = fft_size pixels (frequency axis)
        h, w  = rgb.shape[:2]
        img   = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        self.tile_ready.emit(self.row_start, img.copy())

class TileThread(QThread):
    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.worker.moveToThread(self)
        self.started.connect(self.worker.run)


# ─────────────────────────────────────────────────────────────────────────────
# Annotation coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────
class AnnotationItem:
    HANDLE_CURSORS = {
        "tl": Qt.CursorShape.SizeFDiagCursor, "br": Qt.CursorShape.SizeFDiagCursor,
        "tr": Qt.CursorShape.SizeBDiagCursor, "bl": Qt.CursorShape.SizeBDiagCursor,
        "tm": Qt.CursorShape.SizeVerCursor,   "bm": Qt.CursorShape.SizeVerCursor,
        "lm": Qt.CursorShape.SizeHorCursor,   "rm": Qt.CursorShape.SizeHorCursor,
    }

    def __init__(self, ann: dict, index: int):
        self.ann   = ann
        self.index = index

    def label(self) -> str:
        return self.ann.get("core:label", "")

    # ── coordinate conversions ────────────────────────────────────────────────
    @staticmethod
    def freq_to_x(freq: float, cf: float, sr: float, w: int) -> float:
        """Absolute frequency → pixel x on canvas of width w (fftshift layout)."""
        return ((freq - cf) / sr + 0.5) * w

    @staticmethod
    def x_to_freq(x: float, cf: float, sr: float, w: int) -> float:
        return cf + (x / w - 0.5) * sr

    @staticmethod
    def row_to_y(row: float, row_h: int) -> float:
        return row * row_h

    @staticmethod
    def y_to_row(y: float, row_h: int) -> float:
        return y / row_h

    def to_rect(self, fft_size: int, row_h: int,
                cf: float, sr: float, canvas_w: int) -> QRectF:
        a  = self.ann
        ss = a.get("core:sample_start", 0)
        sc = a.get("core:sample_count", fft_size)
        fl = a.get("core:freq_lower_edge", cf - sr / 2)
        fu = a.get("core:freq_upper_edge", cf + sr / 2)

        # Y: sample index → FFT row → pixel
        y_top    = self.row_to_y(ss / fft_size, row_h)
        y_bottom = self.row_to_y((ss + sc) / fft_size, row_h)

        # X: frequency → pixel
        x_left  = self.freq_to_x(fl, cf, sr, canvas_w)
        x_right = self.freq_to_x(fu, cf, sr, canvas_w)

        return QRectF(x_left, y_top, x_right - x_left, y_bottom - y_top)

    def handles(self, r: QRectF) -> dict:
        cx, cy = r.center().x(), r.center().y()
        return {
            "tl": QPointF(r.left(),  r.top()),
            "tr": QPointF(r.right(), r.top()),
            "bl": QPointF(r.left(),  r.bottom()),
            "br": QPointF(r.right(), r.bottom()),
            "tm": QPointF(cx,        r.top()),
            "bm": QPointF(cx,        r.bottom()),
            "lm": QPointF(r.left(),  cy),
            "rm": QPointF(r.right(), cy),
        }

    def handle_rect(self, pt: QPointF) -> QRectF:
        s = HANDLE_SIZE
        return QRectF(pt.x() - s/2, pt.y() - s/2, s, s)

    def hit_handle(self, pos: QPointF, r: QRectF) -> str | None:
        for name, pt in self.handles(r).items():
            if self.handle_rect(pt).contains(pos):
                return name
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Frequency axis ruler (drawn above the canvas)
# ─────────────────────────────────────────────────────────────────────────────
class FreqAxisWidget(QWidget):
    HEIGHT = 36

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self.HEIGHT)
        self.cf = 0.0
        self.sr = 1.0

    def set_params(self, cf: float, sr: float):
        self.cf = cf
        self.sr = sr
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(25, 25, 35))
        p.setPen(QColor(200, 200, 200))
        p.setFont(QFont("Courier", 7))
        w = self.width()
        n = 9
        for i in range(n):
            frac = i / (n - 1)
            freq = (self.cf - self.sr / 2) + frac * self.sr
            x    = int(frac * w)
            if abs(freq) >= 1e9:
                lbl = f"{freq/1e9:.3f}G"
            elif abs(freq) >= 1e6:
                lbl = f"{freq/1e6:.3f}M"
            elif abs(freq) >= 1e3:
                lbl = f"{freq/1e3:.1f}k"
            else:
                lbl = f"{freq:.0f}"
            p.drawLine(x, 0, x, 6)
            fm  = p.fontMetrics()
            tw  = fm.horizontalAdvance(lbl)
            tx  = max(0, min(x - tw // 2, w - tw))
            p.drawText(tx, self.HEIGHT - 4, lbl)


# ─────────────────────────────────────────────────────────────────────────────
# Time axis ruler (left of the canvas, covers the scroll viewport)
# ─────────────────────────────────────────────────────────────────────────────
class TimeAxisWidget(QWidget):
    WIDTH = 72

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(self.WIDTH)
        self.sr       = 1.0
        self.fft_size = FFT_SIZE
        self.row_h    = 1
        self.scroll_y = 0
        self.view_h   = 400

    def update_params(self, sr, fft_size, row_h, scroll_y, view_h):
        self.sr = sr; self.fft_size = fft_size
        self.row_h = row_h; self.scroll_y = scroll_y; self.view_h = view_h
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(25, 25, 35))
        p.setPen(QColor(200, 200, 200))
        p.setFont(QFont("Courier", 7))
        n = 8
        for i in range(n + 1):
            frac   = i / n
            px_y   = int(frac * self.view_h)
            abs_y  = self.scroll_y + px_y
            row    = abs_y / self.row_h
            t_sec  = row * self.fft_size / self.sr
            if t_sec >= 1.0:
                lbl = f"{t_sec:.3f}s"
            elif t_sec >= 1e-3:
                lbl = f"{t_sec*1e3:.2f}ms"
            else:
                lbl = f"{t_sec*1e6:.1f}µs"
            p.drawLine(self.WIDTH - 5, px_y, self.WIDTH, px_y)
            p.drawText(QRect(0, px_y - 8, self.WIDTH - 7, 16),
                       Qt.AlignmentFlag.AlignRight |
                       Qt.AlignmentFlag.AlignVCenter, lbl)


# ─────────────────────────────────────────────────────────────────────────────
# Waterfall canvas
# ─────────────────────────────────────────────────────────────────────────────
class WaterfallCanvas(QWidget):
    status_msg         = pyqtSignal(str)
    annotation_changed = pyqtSignal()

    MODE_SELECT = "select"
    MODE_CREATE = "create"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset: SigMFDataset | None = None
        self.fft_size   = FFT_SIZE
        self.cmap_name  = "Viridis"
        self.vmin       = -60.0
        self.vmax       = -20.0
        self.row_h      = 1
        self.total_rows = 0

        self._tile_cache:   dict[int, QPixmap] = {}
        self._pending:      set[int]           = set()
        self._threads:      list[TileThread]   = []

        self.annotations:   list[AnnotationItem] = []
        self.selected_idx:  int | None = None

        # drag state
        self._drag_handle:  str | None    = None
        self._drag_ann_idx: int | None    = None
        self._drag_origin:  QPointF | None = None
        self._drag_rect0:   QRectF | None  = None   # rect at drag start

        # new-box draw state
        self._mode          = self.MODE_SELECT
        self._nb_start:     QPointF | None = None
        self._nb_rect:      QRectF | None  = None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    # ── public API ────────────────────────────────────────────────────────────
    def load_dataset(self, ds: SigMFDataset):
        self._abort_threads()
        self._tile_cache.clear()
        self._pending.clear()
        self.dataset    = ds
        self.total_rows = math.ceil(ds.total_samples / self.fft_size)
        self._rebuild_annotations()
        self.setFixedSize(self.fft_size, max(self.total_rows * self.row_h, 1))
        self.update()

    def set_fft_size(self, n: int):
        self.fft_size = n
        if self.dataset:
            self.load_dataset(self.dataset)

    def set_colormap(self, name: str):
        self.cmap_name = name
        self._tile_cache.clear(); self._pending.clear(); self.update()

    def set_vrange(self, vmin: float, vmax: float):
        self.vmin = vmin; self.vmax = vmax
        self._tile_cache.clear(); self._pending.clear(); self.update()

    def set_mode(self, mode: str):
        self._mode    = mode
        self._nb_start = None
        self._nb_rect  = None
        cur = (Qt.CursorShape.CrossCursor if mode == self.MODE_CREATE
               else Qt.CursorShape.ArrowCursor)
        self.setCursor(QCursor(cur))

    # ── tiles ─────────────────────────────────────────────────────────────────
    def _tile_for_row(self, row: int) -> int:
        return (row // TILE_ROWS) * TILE_ROWS

    def _request_tile(self, tile_row: int):
        if tile_row in self._tile_cache or tile_row in self._pending:
            return
        if not self.dataset:
            return
        rows = min(TILE_ROWS, self.total_rows - tile_row)
        if rows <= 0:
            return
        self._pending.add(tile_row)
        w = TileWorker(self.dataset, self.fft_size, self.cmap_name,
                       self.vmin, self.vmax, tile_row, rows)
        w.tile_ready.connect(self._on_tile_ready)
        t = TileThread(w)
        t.finished.connect(lambda th=t: self._threads.remove(th)
                           if th in self._threads else None)
        self._threads.append(t)
        t.start()

    def _on_tile_ready(self, row_start: int, img: QImage):
        self._pending.discard(row_start)
        self._tile_cache[row_start] = QPixmap.fromImage(img)
        self.update()

    def _abort_threads(self):
        for t in self._threads:
            t.quit(); t.wait(300)
        self._threads.clear(); self._pending.clear()

    # ── annotations ───────────────────────────────────────────────────────────
    def _rebuild_annotations(self):
        self.annotations = [
            AnnotationItem(a, i)
            for i, a in enumerate(self.dataset.annotations)
        ] if self.dataset else []

    def _ann_rect(self, item: AnnotationItem) -> QRectF:
        if not self.dataset:
            return QRectF()
        return item.to_rect(self.fft_size, self.row_h,
                            self.dataset.center_freq, self.dataset.sample_rate,
                            self.width())

    def _rect_to_sigmf(self, r: QRectF) -> tuple[int, int, float, float]:
        """Pixel rect → (sample_start, sample_count, freq_lower, freq_upper)."""
        ds   = self.dataset
        r    = r.normalized()
        # Y → samples (centre each edge on the nearest FFT row)
        row_top    = r.top()    / self.row_h
        row_bottom = r.bottom() / self.row_h
        ss = int(round(row_top    * self.fft_size))
        se = int(round(row_bottom * self.fft_size))
        sc = max(self.fft_size, se - ss)
        # X → frequency
        fl = AnnotationItem.x_to_freq(r.left(),  ds.center_freq, ds.sample_rate, self.width())
        fu = AnnotationItem.x_to_freq(r.right(), ds.center_freq, ds.sample_rate, self.width())
        fl, fu = min(fl, fu), max(fl, fu)
        return ss, sc, fl, fu

    def _hit_annotation(self, pos: QPointF) -> int | None:
        for i, item in enumerate(self.annotations):
            if self._ann_rect(item).contains(pos):
                return i
        return None

    # ── paint ─────────────────────────────────────────────────────────────────
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not self.dataset:
            painter.fillRect(self.rect(), QColor(20, 20, 30))
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No dataset loaded")
            return

        clip = event.rect()
        w    = self.width()

        # Draw tiles
        row_top    = max(0, clip.top() // self.row_h)
        row_bottom = min(self.total_rows,
                         math.ceil((clip.bottom() + 1) / self.row_h))
        tile_row   = self._tile_for_row(row_top)
        while tile_row < row_bottom:
            tile_h_rows = min(TILE_ROWS, self.total_rows - tile_row)
            tile_y      = tile_row * self.row_h
            tile_h_px   = tile_h_rows * self.row_h
            if tile_row in self._tile_cache:
                pm = self._tile_cache[tile_row]
                painter.drawPixmap(QRect(0, tile_y, w, tile_h_px), pm, pm.rect())
            else:
                painter.fillRect(QRect(0, tile_y, w, tile_h_px), QColor(15, 15, 25))
                painter.setPen(QColor(60, 60, 80))
                painter.drawText(QRect(0, tile_y, w, min(20, tile_h_px)),
                                 Qt.AlignmentFlag.AlignCenter, "Loading…")
                self._request_tile(tile_row)
            tile_row += TILE_ROWS

        # Draw annotations
        for i, item in enumerate(self.annotations):
            r = self._ann_rect(item)
            if not QRectF(clip).intersects(r):
                continue
            sel   = (i == self.selected_idx)
            col   = SEL_COLOR if sel else ANN_COLOR
            pen   = QPen(col, 1.5, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), 30)))
            painter.drawRect(r)

            lbl = item.label()
            if lbl:
                painter.setPen(col)
                fm  = painter.fontMetrics()
                tr  = fm.boundingRect(lbl)
                tx  = r.left() + 3
                ty  = r.top() - 3
                bg  = QRectF(tx - 1, ty - tr.height(), tr.width() + 4, tr.height() + 2)
                painter.fillRect(bg, QColor(0, 0, 0, 160))
                painter.drawText(QPointF(tx, ty), lbl)

            if sel:
                painter.setPen(QPen(SEL_COLOR, 1))
                painter.setBrush(QBrush(SEL_COLOR))
                for pt in item.handles(r).values():
                    painter.drawRect(item.handle_rect(pt))

        # In-progress new box
        if self._nb_rect and not self._nb_rect.isNull():
            painter.setPen(QPen(QColor(0, 255, 128), 1.5, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(QColor(0, 255, 128, 25)))
            painter.drawRect(self._nb_rect.normalized())

    # ── mouse ─────────────────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        if not self.dataset or event.button() != Qt.MouseButton.LeftButton:
            if event.button() == Qt.MouseButton.RightButton and self.dataset:
                hit = self._hit_annotation(QPointF(event.position()))
                if hit is not None:
                    self._context_menu(hit, event.globalPosition().toPoint())
            return

        pos = QPointF(event.position())

        # ── CREATE mode ──
        if self._mode == self.MODE_CREATE:
            self._nb_start = pos
            self._nb_rect  = QRectF(pos, pos)
            return

        # ── SELECT mode ──
        # 1. Check handles on currently selected annotation
        if self.selected_idx is not None:
            item = self.annotations[self.selected_idx]
            r    = self._ann_rect(item)
            h    = item.hit_handle(pos, r)
            if h:
                self._drag_handle  = h
                self._drag_ann_idx = self.selected_idx
                self._drag_origin  = pos
                self._drag_rect0   = QRectF(r)
                return

        # 2. Hit-test body of any annotation
        hit = self._hit_annotation(pos)
        if hit is not None:
            self.selected_idx  = hit
            item               = self.annotations[hit]
            self._drag_handle  = "move"
            self._drag_ann_idx = hit
            self._drag_origin  = pos
            self._drag_rect0   = QRectF(self._ann_rect(item))
        else:
            self.selected_idx = None

        self.update()

    def mouseMoveEvent(self, event):
        if not self.dataset:
            return
        pos = QPointF(event.position())

        # ── drawing new box ──
        if self._mode == self.MODE_CREATE and self._nb_start:
            x0, y0 = self._nb_start.x(), self._nb_start.y()
            self._nb_rect = QRectF(min(x0, pos.x()), min(y0, pos.y()),
                                   abs(pos.x()-x0), abs(pos.y()-y0))
            self.update()
            return

        # ── dragging handle / moving annotation ──
        if self._drag_handle and self._drag_origin and self._drag_rect0 is not None:
            dx = pos.x() - self._drag_origin.x()
            dy = pos.y() - self._drag_origin.y()
            r  = QRectF(self._drag_rect0)
            h  = self._drag_handle

            if   h == "move": r.translate(dx, dy)
            elif h == "tl":   r.setTopLeft(r.topLeft() + QPointF(dx, dy))
            elif h == "tr":   r.setTopRight(r.topRight() + QPointF(dx, dy))
            elif h == "bl":   r.setBottomLeft(r.bottomLeft() + QPointF(dx, dy))
            elif h == "br":   r.setBottomRight(r.bottomRight() + QPointF(dx, dy))
            elif h == "tm":   r.setTop(r.top() + dy)
            elif h == "bm":   r.setBottom(r.bottom() + dy)
            elif h == "lm":   r.setLeft(r.left() + dx)
            elif h == "rm":   r.setRight(r.right() + dx)

            r = r.normalized()
            if r.width() >= MIN_BOX_PX and r.height() >= MIN_BOX_PX:
                ss, sc, fl, fu = self._rect_to_sigmf(r)
                idx = self._drag_ann_idx
                if idx is not None:
                    self.dataset.update_annotation(idx, **{
                        "core:sample_start":    ss,
                        "core:sample_count":    sc,
                        "core:freq_lower_edge": fl,
                        "core:freq_upper_edge": fu,
                    })
                    # update in-memory item without full rebuild
                    self.annotations[idx].ann = self.dataset.annotations[idx]
                self.update()
                self.status_msg.emit(
                    f"ss={ss:,}  sc={sc:,}  "
                    f"fl={fl/1e6:.3f} MHz  fu={fu/1e6:.3f} MHz"
                )
            return

        # ── hover cursor update ──
        if self._mode == self.MODE_SELECT:
            cur = Qt.CursorShape.ArrowCursor
            if self.selected_idx is not None:
                item = self.annotations[self.selected_idx]
                r    = self._ann_rect(item)
                h    = item.hit_handle(pos, r)
                if h:
                    cur = item.HANDLE_CURSORS.get(h, Qt.CursorShape.ArrowCursor)
                elif r.contains(pos):
                    cur = Qt.CursorShape.SizeAllCursor
            self.setCursor(QCursor(cur))

    def mouseReleaseEvent(self, event):
        if not self.dataset or event.button() != Qt.MouseButton.LeftButton:
            return
        pos = QPointF(event.position())

        # ── finish drawing ──
        if self._mode == self.MODE_CREATE and self._nb_rect:
            r = self._nb_rect.normalized()
            if r.width() >= MIN_BOX_PX and r.height() >= MIN_BOX_PX:
                ss, sc, fl, fu = self._rect_to_sigmf(r)
                label, ok = QInputDialog.getText(
                    self, "New Annotation", "Label (optional):")
                if ok:
                    self.dataset.add_annotation(ss, sc, fl, fu, label.strip())
                    self._rebuild_annotations()
                    self.selected_idx = len(self.annotations) - 1
                    self.annotation_changed.emit()
            self._nb_start = None
            self._nb_rect  = None
            self.update()
            return

        # ── finish drag ──
        if self._drag_handle:
            # commit final position without losing selection
            self._rebuild_annotations()
            self.annotation_changed.emit()
            # restore selection (rebuild resets it)
            # selected_idx stays valid because annotation list order is unchanged
            self._drag_handle  = None
            self._drag_ann_idx = None
            self._drag_origin  = None
            self._drag_rect0   = None
            self.update()

    def mouseDoubleClickEvent(self, event):
        if not self.dataset:
            return
        hit = self._hit_annotation(QPointF(event.position()))
        if hit is not None:
            self._edit_label(hit)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self.selected_idx is not None:
                self._delete_annotation(self.selected_idx)

    # ── context menu / label edit / delete ───────────────────────────────────
    def _context_menu(self, idx: int, gpos: QPoint):
        self.selected_idx = idx
        self.update()
        menu  = QMenu(self)
        a_edt = menu.addAction("Edit Label…")
        a_del = menu.addAction("Delete")
        act   = menu.exec(gpos)
        if act == a_edt:
            self._edit_label(idx)
        elif act == a_del:
            self._delete_annotation(idx)

    def _edit_label(self, idx: int):
        label, ok = QInputDialog.getText(
            self, "Edit Label", "Label:",
            text=self.annotations[idx].label())
        if ok:
            self.dataset.update_annotation(
                self.annotations[idx].index,
                **{"core:label": label.strip()})
            self._rebuild_annotations()
            self.selected_idx = idx
            self.annotation_changed.emit()
            self.update()

    def _delete_annotation(self, idx: int):
        self.dataset.delete_annotation(self.annotations[idx].index)
        self._rebuild_annotations()
        self.selected_idx = None
        self.annotation_changed.emit()
        self.update()

    def closeEvent(self, event):
        self._abort_threads()
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# Annotation list panel
# ─────────────────────────────────────────────────────────────────────────────
class AnnotationPanel(QWidget):
    jump_to = pyqtSignal(int)   # FFT row to scroll to

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        lbl = QLabel("Annotations")
        lbl.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(lbl)
        self.lw = QListWidget()
        self.lw.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.lw.itemDoubleClicked.connect(self._on_double)
        layout.addWidget(self.lw)
        self.setMinimumWidth(200)
        self.setMaximumWidth(280)

    def refresh(self, annotations: list[AnnotationItem], fft_size: int):
        self.lw.clear()
        for item in annotations:
            ss  = item.ann.get("core:sample_start", 0)
            sc  = item.ann.get("core:sample_count", 0)
            lbl = item.label() or "(no label)"
            li  = QListWidgetItem(f"[{ss:,}–{ss+sc:,}]  {lbl}")
            li.setData(Qt.ItemDataRole.UserRole, ss // fft_size)
            self.lw.addItem(li)

    def _on_double(self, li: QListWidgetItem):
        row = li.data(Qt.ItemDataRole.UserRole)
        if row is not None:
            self.jump_to.emit(row)


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SigMF Waterfall Annotator")
        self.resize(1200, 780)
        self.dataset: SigMFDataset | None = None
        self._build_ui()

    def _build_ui(self):
        # ── toolbar ──
        tb = QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)

        def tbtn(label, slot, checkable=False):
            b = QPushButton(label)
            b.setFixedHeight(28)
            b.setCheckable(checkable)
            b.clicked.connect(slot)
            tb.addWidget(b)
            return b

        tbtn("📂 Open SigMF…", self.open_file)
        tb.addSeparator()
        self.btn_save = tbtn("💾 Save", self.save_file)
        self.btn_save.setEnabled(False)
        tb.addSeparator()
        tb.addWidget(QLabel("  Mode: "))
        self.btn_sel = tbtn("🔲 Select", lambda: self._set_mode("select"), True)
        self.btn_sel.setChecked(True)
        self.btn_drw = tbtn("✏️ Draw",   lambda: self._set_mode("create"), True)
        tb.addSeparator()
        tb.addWidget(QLabel("  FFT: "))
        self.fft_spin = QSpinBox()
        self.fft_spin.setRange(64, 8192); self.fft_spin.setValue(FFT_SIZE)
        self.fft_spin.setSingleStep(64);  self.fft_spin.setFixedWidth(80)
        self.fft_spin.editingFinished.connect(self._on_fft)
        tb.addWidget(self.fft_spin)
        tb.addSeparator()
        tb.addWidget(QLabel("  Colormap: "))
        self.cmap_cb = QComboBox()
        self.cmap_cb.addItems(list(COLORMAPS)); self.cmap_cb.currentTextChanged.connect(self._on_cmap)
        tb.addWidget(self.cmap_cb)
        tb.addSeparator()
        tb.addWidget(QLabel("  dB min: "))
        self.vmin_sp = QDoubleSpinBox(); self.vmin_sp.setRange(-200,0); self.vmin_sp.setValue(-60); self.vmin_sp.setFixedWidth(70)
        self.vmin_sp.editingFinished.connect(self._on_vrange); tb.addWidget(self.vmin_sp)
        tb.addWidget(QLabel("  max: "))
        self.vmax_sp = QDoubleSpinBox(); self.vmax_sp.setRange(-200,0); self.vmax_sp.setValue(-20); self.vmax_sp.setFixedWidth(70)
        self.vmax_sp.editingFinished.connect(self._on_vrange); tb.addWidget(self.vmax_sp)

        # ── central ──
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left pane: time-axis + (freq-axis header / scroll-area)
        left = QWidget()
        left_h = QHBoxLayout(left)
        left_h.setContentsMargins(0, 0, 0, 0)
        left_h.setSpacing(0)

        # Time axis (left ruler)
        self.time_axis = TimeAxisWidget()
        left_h.addWidget(self.time_axis)

        # Right of time axis: freq axis header + scrollable canvas
        right_col = QWidget()
        right_v = QVBoxLayout(right_col)
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.setSpacing(0)

        self.freq_axis = FreqAxisWidget()
        right_v.addWidget(self.freq_axis)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.verticalScrollBar().valueChanged.connect(self._on_scroll)

        self.canvas = WaterfallCanvas()
        self.canvas.status_msg.connect(lambda m: self.statusBar().showMessage(m))
        self.canvas.annotation_changed.connect(self._on_ann_changed)
        self.scroll.setWidget(self.canvas)
        right_v.addWidget(self.scroll)

        left_h.addWidget(right_col)
        splitter.addWidget(left)

        # Right pane: annotation list
        self.ann_panel = AnnotationPanel()
        self.ann_panel.jump_to.connect(self._jump_to_row)
        splitter.addWidget(self.ann_panel)
        splitter.setSizes([920, 240])

        root.addWidget(splitter)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready — open a .sigmf-meta file.")

    # ── slots ─────────────────────────────────────────────────────────────────
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SigMF Metadata File", "",
            "SigMF Metadata (*.sigmf-meta)")
        if not path:
            return
        if not path.lower().endswith(".sigmf-meta"):
            QMessageBox.warning(self, "Invalid File",
                                "Please select a .sigmf-meta file.")
            return
        try:
            ds = SigMFDataset(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

        self.dataset = ds
        self.canvas.set_fft_size(self.fft_spin.value())
        self.canvas.set_colormap(self.cmap_cb.currentText())
        vmin, vmax = self._estimate_vrange(ds)
        self.vmin_sp.setValue(vmin)
        self.vmax_sp.setValue(vmax)
        self.canvas.set_vrange(vmin, vmax)
        self.canvas.load_dataset(ds)
        self.freq_axis.set_params(ds.center_freq, ds.sample_rate)
        self._sync_canvas_width()
        self._on_ann_changed()
        self.btn_save.setEnabled(True)
        self.statusBar().showMessage(
            f"Loaded: {Path(path).name}  |  {ds.total_samples:,} samples  |  "
            f"SR={ds.sample_rate/1e6:.3f} MHz  |  CF={ds.center_freq/1e6:.3f} MHz  |  "
            f"{len(ds.annotations)} annotation(s)")

    def save_file(self):
        if not self.dataset:
            return
        try:
            self.dataset.save()
            self.statusBar().showMessage(f"Saved → {self.dataset.meta_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _set_mode(self, mode: str):
        self.canvas.set_mode(mode)
        self.btn_sel.setChecked(mode == "select")
        self.btn_drw.setChecked(mode == "create")

    def _on_fft(self):
        if self.dataset:
            self.canvas.set_fft_size(self.fft_spin.value())

    def _on_cmap(self, name: str):
        self.canvas.set_colormap(name)

    def _on_vrange(self):
        self.canvas.set_vrange(self.vmin_sp.value(), self.vmax_sp.value())

    def _on_scroll(self, val: int):
        if self.dataset:
            self.time_axis.update_params(
                self.dataset.sample_rate,
                self.canvas.fft_size,
                self.canvas.row_h,
                val,
                self.scroll.viewport().height()
            )

    def _on_ann_changed(self):
        if self.dataset:
            self.ann_panel.refresh(self.canvas.annotations, self.canvas.fft_size)

    def _jump_to_row(self, row: int):
        self.scroll.verticalScrollBar().setValue(row * self.canvas.row_h)

    def _sync_canvas_width(self):
        vw = self.scroll.viewport().width()
        if self.dataset and vw > 0:
            # Canvas pixel width == fft_size; scale to fill viewport
            self.canvas.setFixedWidth(vw)
            # Keep height as total_rows (1 px/row minimum)
            self.canvas.setFixedHeight(max(self.canvas.total_rows, 1))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_canvas_width()

    def _estimate_vrange(self, ds: SigMFDataset, probe_rows: int = 64):
        fs      = self.fft_spin.value()
        samples = ds.read_samples(0, probe_rows * fs)
        if samples.size < fs:
            return -60.0, -20.0
        pad = (-len(samples)) % fs
        if pad:
            samples = np.concatenate([samples, np.zeros(pad, dtype=np.complex64)])
        rows = len(samples) // fs
        mat  = samples.reshape(rows, fs)
        win  = np.hanning(fs).astype(np.float32)
        wg   = np.sum(win**2) / fs
        spec  = np.fft.fftshift(np.fft.fft(mat * win, axis=1), axes=1)
        power = 10 * np.log10(np.abs(spec / fs)**2 / wg + 1e-12)
        p2, p98 = float(np.percentile(power, 2)), float(np.percentile(power, 98))
        m = max((p98 - p2) * 0.15, 2.0)
        return round(p2 - m, 1), round(p98 + m, 1)

    def closeEvent(self, event):
        self.canvas._abort_threads()
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,        QColor(35, 35, 45))
    pal.setColor(QPalette.ColorRole.WindowText,    QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.Base,          QColor(25, 25, 35))
    pal.setColor(QPalette.ColorRole.Text,          QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.Button,        QColor(50, 50, 65))
    pal.setColor(QPalette.ColorRole.ButtonText,    QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.Highlight,     QColor(42, 130, 218))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
