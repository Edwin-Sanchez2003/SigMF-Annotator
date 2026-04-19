# sigmf_annotator.spec
#
# PyInstaller spec file for SigMF Waterfall Annotator.
#
# Build with:
#   pyinstaller sigmf_annotator.spec
#
# Output: dist/sigmf_annotator/sigmf_annotator.exe  (one-folder build)
#   or use --onefile below for a single .exe (slower startup, same result).

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(SPECPATH)          # directory containing this .spec file
SRC     = ROOT / "src" / "annotator"
DETECTORS = SRC / "detectors"

# ── Data files to bundle ───────────────────────────────────────────────────────
# Include the detectors folder so users get the built-in examples.
# Format: list of (source_path, dest_folder_inside_bundle)
datas = [
    (str(DETECTORS), "detectors"),
]

# Pull in any Qt data files PyInstaller might miss (translations, etc.)
datas += collect_data_files("PyQt6")

# ── Hidden imports ─────────────────────────────────────────────────────────────
# PyQt6 platform plugin and numpy internals are commonly missed by the analyser.
hiddenimports = [
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "PyQt6.sip",
    "numpy",
    "numpy.core._multiarray_umath",
    "numpy.core._multiarray_tests",
]

# ── Analysis ───────────────────────────────────────────────────────────────────
a = Analysis(
    [str(SRC / "sigmf_annotator.py")],   # entry point
    pathex=[str(ROOT / "src")],           # so 'annotator.*' resolves
    binaries=collect_dynamic_libs("PyQt6"),
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",      # not used; saves ~3 MB
        "matplotlib",
        "scipy",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

# ── One-folder EXE (recommended: faster startup) ───────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,      # binaries go in the COLLECT step
    name="sigmf_annotator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                   # compress with UPX if available (optional)
    console=False,              # no console window (set True to debug crashes)
    # icon="icon.ico",          # uncomment and provide a .ico file if desired
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="sigmf_annotator",
)

# ── Uncomment below instead of the EXE+COLLECT above for a true single .exe ───
# WARNING: single-file mode extracts to a temp folder on each launch (~2-3s slower)
#
# exe = EXE(
#     pyz,
#     a.scripts,
#     a.binaries,
#     a.datas,
#     [],
#     name="sigmf_annotator",
#     debug=False,
#     strip=False,
#     upx=True,
#     console=False,
#     onefile=True,
# )
