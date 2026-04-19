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