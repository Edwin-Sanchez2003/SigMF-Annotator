"""
    An example Detector, which will be imported by sigmf_annotator.py

    Use this as a baseline for writing a custom Detector to generate
    annotations.
"""

from annotator.sigmf_annotator import BaseDetector, DetectorParam, DetectionContext, DetectionResult

class ExampleDetector(BaseDetector):
    NAME = "Example Detector"
    PARAMS = [
        DetectorParam("label",     "Label",           "str",   "my_signal"),
        DetectorParam("threshold", "Power Threshold", "float", -40.0,
                      minimum=-120, maximum=0, step=0.5, decimals=1),
        DetectorParam("mode",      "Detection Mode",  "choice", "peak",
                      choices=["peak", "mean", "median"]),
    ]

    def run(self, ctx: DetectionContext, **kw) -> list[DetectionResult]:
        # ctx.samples      → complex64 ndarray (the IQ chunk)
        # ctx.sample_rate  → Hz
        # ctx.center_freq  → Hz
        # ctx.sample_start → absolute sample offset in the file
        # ctx.fft_size     → current viewer FFT size
        # ctx.sigmf_meta   → full metadata dict
        # kw["threshold"], kw["label"], kw["mode"] → from the GUI widgets
        results = []
        # ... your logic ...
        offset_samples = 0
        duration_samples = ctx.samples.size
        cf = ctx.center_freq
        bw = ctx.sample_rate
        results.append(DetectionResult(
            sample_start=ctx.sample_start + offset_samples,
            sample_count=duration_samples,
            freq_lower=cf - bw/2,
            freq_upper=cf + bw/2,
            label=kw["label"]
        ))
        
        return results