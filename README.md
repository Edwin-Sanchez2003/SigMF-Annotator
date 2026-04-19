# SigMF Waterfall Annotator

An interactive PyQt6 tool for visualising and annotating [SigMF](https://sigmf.org/) datasets as a scrollable waterfall plot. Supports manual annotation editing, undo/redo, and pluggable energy-detection algorithms that generate annotations automatically.

## Installation

```bash
pip install -e .
```

## Running

```bash
sigmf-annotator
# or
python src/annotator/sigmf_annotator.py
```

## Capabilities

**Waterfall viewer**
- Loads any valid SigMF datatype (`cf32`, `ci16`, `cu8`, `rf64`, etc.) — all formats are converted to complex64 at load time.
- Renders the dataset as a chunked, scrollable waterfall plot (frequency on the X axis, time on the Y axis). Only the visible portion is held in memory at any given time, so arbitrarily large files are supported.
- Adjustable FFT size, colormap, and dB display range.

**Annotation editing**
- **Draw mode** — drag a rectangle on the waterfall to create a new bounding-box annotation; a label prompt appears on release.
- **Select mode** — click an annotation to select it; drag the body to move it or drag any of the eight edge/corner handles to resize it. Double-click or right-click to edit the label or delete it.
- Annotations are stored in SigMF convention (`core:sample_start`, `core:sample_count`, `core:freq_lower_edge`, `core:freq_upper_edge`).
- **Save** overwrites the `.sigmf-meta` file in place.

**Undo / Redo**
- Every annotation action (draw, move, resize, label edit, delete, bulk detection) is undoable.
- Keyboard shortcuts: `Ctrl+Z` to undo, `Ctrl+Y` / `Ctrl+Shift+Z` to redo.
- The toolbar buttons show a tooltip describing the action that will be undone/redone.

**Energy detection**
- Select an algorithm from the *Energy Detection* panel on the right.
- Each algorithm exposes its own tunable parameters directly in the GUI (thresholds, labels, etc.).
- **Run on Visible** — runs the detector only over the rows currently on screen.
- **Run on Entire Dataset** — streams through the whole file in fixed-size chunks (~32 MB each) so memory usage stays bounded regardless of file size.
- Results are added as annotations and are immediately undoable as a single bulk action.
- A progress bar and an abort button are shown while detection is running.

---

## Adding a Custom Detector

Drop a `.py` file into `src/annotator/detectors/`. It is auto-imported at startup — no registration step required.

### Minimal template

```python
# src/annotator/detectors/my_detector.py

from annotator.sigmf_annotator import (
    BaseDetector, DetectorParam, DetectionContext, DetectionResult
)

class MyDetector(BaseDetector):
    NAME = "My Detector"          # shown in the GUI dropdown
    PARAMS = [
        DetectorParam(
            name="label",
            label="Annotation Label",
            kind="str",
            default="signal",
        ),
        DetectorParam(
            name="threshold_db",
            label="Threshold (dB)",
            kind="float",
            default=-45.0,
            minimum=-120.0,
            maximum=0.0,
            step=1.0,
            decimals=1,
            tooltip="Rows with mean power above this are marked as signal.",
        ),
        DetectorParam(
            name="mode",
            label="Mode",
            kind="choice",
            default="peak",
            choices=["peak", "mean", "median"],
        ),
    ]

    def run(self, ctx: DetectionContext, **kw) -> list[DetectionResult]:
        """
        Called once per chunk of IQ data.  For large files the annotator
        calls this multiple times with successive chunks — do not assume
        ctx.samples covers the whole file.

        Parameters
        ----------
        ctx.samples       complex64 ndarray for this chunk
        ctx.sample_rate   Hz
        ctx.center_freq   Hz
        ctx.sample_start  absolute sample index where this chunk begins
        ctx.fft_size      FFT size currently used by the viewer
        ctx.sigmf_meta    full parsed metadata dict (read-only)

        kw                one key per DetectorParam, already cast to the
                          correct Python type (float, int, str, bool, str)
        """
        results = []

        # --- your detection logic here ---
        # example: flag the whole chunk if mean power exceeds threshold
        import numpy as np
        power_db = 10 * np.log10(np.mean(np.abs(ctx.samples) ** 2) + 1e-12)
        if power_db >= float(kw["threshold_db"]):
            results.append(DetectionResult(
                sample_start = ctx.sample_start,
                sample_count = len(ctx.samples),
                freq_lower   = ctx.center_freq - ctx.sample_rate / 2,
                freq_upper   = ctx.center_freq + ctx.sample_rate / 2,
                label        = kw["label"],
            ))

        return results
```

### `DetectorParam` field reference

| `kind` | GUI widget | Extra fields |
|---|---|---|
| `"float"` | `QDoubleSpinBox` | `minimum`, `maximum`, `step`, `decimals` |
| `"int"` | `QSpinBox` | `minimum`, `maximum`, `step` |
| `"str"` | `QLineEdit` | — |
| `"bool"` | `QCheckBox` | — |
| `"choice"` | `QComboBox` | `choices` (list of strings) |

All params support an optional `tooltip` string shown on hover.
