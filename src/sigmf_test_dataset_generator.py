#!/usr/bin/env python3
"""
SigMF Test Dataset Generator
==============================
Generates a broad corpus of SigMF datasets (.sigmf-meta / .sigmf-data / .sigmf archives)
to stress-test a C++ SigMF reader/writer library.

Covers:
  - Every core SigMF datatype (all widths, endianness, real/complex, int/uint/float)
  - Conforming datasets (spec-valid)
  - Non-conforming datasets (structurally broken, in a controlled/labeled way)
  - Variety across REQUIRED / SHOULD / OPTIONAL fields in global, capture, annotation objects
  - Multi-recording-set archives, collections, non-conforming dataset flag, multi-channel

Reference: SigMF spec v1.x (https://github.com/sigmf/SigMF)
"""

import os
import json
import struct
import random
import shutil
import hashlib
import tarfile
import datetime
import numpy as np
from pathlib import Path

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

OUT_DIR = Path("sigmf_test_corpus")
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)

SIGMF_VERSION = "1.2.0"

# --------------------------------------------------------------------------
# Datatype table: sigmf dtype string -> (numpy dtype, is_complex, struct fmt char, itemsize)
# Covers full SigMF R| C| type grid across widths & endianness
# --------------------------------------------------------------------------

def build_dtype_table():
    table = {}

    # (sigmf_base, numpy_base, complex, size_bytes_per_component)
    bases = [
        ("i8",   np.int8,    1),
        ("u8",   np.uint8,   1),
        ("i16",  np.int16,   2),
        ("u16",  np.uint16,  2),
        ("i32",  np.int32,   4),
        ("u32",  np.uint32,  4),
        ("f32",  np.float32, 4),
        ("f64",  np.float64, 8),
    ]

    for base, npbase, size in bases:
        for complex_flag, prefix in [(False, "r"), (True, "c")]:
            for endian, esuf in [("le", "_le"), ("be", "_be")]:
                # 8-bit types have no endianness distinction in SigMF (no suffix)
                if size == 1:
                    if esuf == "_be":
                        continue
                    esuf = ""
                sigmf_name = f"{prefix}{base}{esuf}"
                table[sigmf_name] = {
                    "npbase": npbase,
                    "complex": complex_flag,
                    "endian": endian,
                    "itemsize": size,
                }
    return table

DTYPE_TABLE = build_dtype_table()

# Some datatypes SigMF spec explicitly disallows/rare-cases we want to test as
# "edge case but technically valid" or "invalid" entries
INVALID_DATATYPE_STRINGS = [
    "cu4",           # not a real sigmf type
    "ri16_le_bad",   # malformed suffix
    "f16_le",        # half precision not in spec
    "",              # empty datatype
    "ci64_le",       # not defined in core spec (64-bit int complex uncommon but let's mark invalid)
]

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def rand_hash(data: bytes) -> str:
    return hashlib.sha512(data).hexdigest()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_json(path: Path, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# --------------------------------------------------------------------------
# Data generation for a given sigmf datatype string
# --------------------------------------------------------------------------

def generate_samples(sigmf_dtype: str, num_samples: int) -> bytes:
    """Generate raw bytes for num_samples of the given sigmf datatype."""
    info = DTYPE_TABLE.get(sigmf_dtype)
    if info is None:
        # Unknown/invalid dtype requested -> just emit random garbage bytes
        return os.urandom(num_samples * 4)

    npbase = info["npbase"]
    is_complex = info["complex"]
    endian = info["endian"]
    n_components = num_samples * (2 if is_complex else 1)

    if np.issubdtype(npbase, np.floating):
        arr = np.random.uniform(-1.0, 1.0, n_components).astype(npbase)
    elif np.issubdtype(npbase, np.signedinteger):
        info_i = np.iinfo(npbase)
        arr = np.random.randint(info_i.min, info_i.max, n_components, dtype=npbase)
    else:  # unsigned
        info_i = np.iinfo(npbase)
        arr = np.random.randint(0, info_i.max, n_components, dtype=npbase)

    # Byte order
    if arr.dtype.itemsize > 1:
        byteorder = "<" if endian == "le" else ">"
        arr = arr.astype(arr.dtype.newbyteorder(byteorder))

    return arr.tobytes()

def sample_size_bytes(sigmf_dtype: str) -> int:
    info = DTYPE_TABLE.get(sigmf_dtype)
    if info is None:
        return 4
    mult = 2 if info["complex"] else 1
    return info["itemsize"] * mult

# --------------------------------------------------------------------------
# Metadata builders — vary REQUIRED vs SHOULD vs OPTIONAL fields
# --------------------------------------------------------------------------

LICENSES = ["CC0-1.0", "CC-BY-4.0", "MIT", None]
AUTHORS = ["Jane Doe <jane@example.com>", "RF Test Lab", None]
HW_LIST = ["USRP B210", "HackRF One", "RTL-SDR v3", None]

def build_global(sigmf_dtype: str, num_samples: int, field_level: str,
                  is_complex_meta_override=None, extra_ns=False,
                  multichannel=False, num_channels=1,
                  non_conforming_flag=False):
    """
    field_level: 'minimal' (REQUIRED only), 'typical' (REQUIRED+SHOULD),
                 'full' (REQUIRED+SHOULD+OPTIONAL+extension namespaces)
    """
    g = {
        "core:datatype": sigmf_dtype,   # REQUIRED
        "core:version": SIGMF_VERSION,  # REQUIRED
    }

    if field_level in ("typical", "full"):
        g["core:sample_rate"] = round(random.uniform(1e3, 40e6), 3)  # SHOULD
        g["core:description"] = f"Auto-generated test dataset ({sigmf_dtype})"  # SHOULD
        g["core:author"] = random.choice(AUTHORS)  # SHOULD
        g["core:num_channels"] = num_channels if multichannel else 1  # SHOULD (default 1)

    if field_level == "full":
        g["core:license"] = random.choice(LICENSES)          # OPTIONAL
        g["core:hw"] = random.choice(HW_LIST)                 # OPTIONAL
        g["core:version_freetext"] = "generated-by-testgen"   # not std but tests extra key tolerance
        g["core:extensions"] = [
            {
                "name": "antenna",
                "version": "1.0.0",
                "optional": True
            }
        ]
        g["core:dataset"] = None  # placeholder, set by caller for archive/non-conforming cases
        g["core:trailing_bytes"] = 0
        g["core:metadata_only"] = False
        g["core:geolocation"] = {
            "type": "Point",
            "coordinates": [round(random.uniform(-180, 180), 5),
                             round(random.uniform(-90, 90), 5)]
        }
        if extra_ns:
            g["antenna:type"] = "dipole"
            g["antenna:gain"] = round(random.uniform(0, 20), 1)

    if non_conforming_flag:
        # SHOULD field per spec for datasets that don't conform to the core namespace strictly
        g["core:dataset"] = "nonconforming-payload.bin"

    return g

def build_captures(num_samples: int, field_level: str, num_segments: int = 1,
                    multichannel=False, num_channels=1):
    caps = []
    if num_segments <= 0:
        return caps  # invalid on purpose (no captures array element -> non-conforming test)

    boundaries = sorted(random.sample(range(1, max(num_samples, 2)),
                                       min(num_segments - 1, max(num_samples - 1, 0)))) if num_segments > 1 else []
    starts = [0] + boundaries

    for i, start in enumerate(starts):
        cap = {"core:sample_start": start}  # REQUIRED
        if field_level in ("typical", "full"):
            cap["core:frequency"] = round(random.uniform(70e6, 6e9), 1)  # SHOULD
            cap["core:datetime"] = now_iso()  # SHOULD
        if field_level == "full":
            cap["core:header_bytes"] = random.choice([0, 0, 16])  # OPTIONAL
            cap["core:global_index"] = start
            if multichannel:
                cap["core:frequency"] = [round(random.uniform(70e6, 6e9), 1)
                                          for _ in range(num_channels)]
        caps.append(cap)
    return caps

ANNOTATION_LABELS = ["burst", "noise", "carrier", "unknown_signal", "interference"]

def build_annotations(num_samples: int, field_level: str, num_annotations: int = 2,
                       overlap_test=False, out_of_bounds_test=False):
    anns = []
    for i in range(num_annotations):
        if out_of_bounds_test and i == num_annotations - 1:
            start = num_samples + 1000  # deliberately out of bounds -> non-conforming
            length = 500
        else:
            start = random.randint(0, max(num_samples - 10, 0))
            length = random.randint(1, max(min(200, num_samples - start), 1))

        ann = {
            "core:sample_start": start,  # REQUIRED
        }
        if field_level in ("typical", "full") or overlap_test:
            ann["core:sample_count"] = length  # SHOULD
        if field_level in ("typical", "full"):
            ann["core:label"] = random.choice(ANNOTATION_LABELS)  # SHOULD
            ann["core:comment"] = "auto-generated annotation"     # SHOULD
        if field_level == "full":
            ann["core:freq_lower_edge"] = round(random.uniform(70e6, 3e9), 1)   # OPTIONAL
            ann["core:freq_upper_edge"] = round(random.uniform(3e9, 6e9), 1)    # OPTIONAL
            ann["core:generator"] = "sigmf_test_gen.py"
            ann["core:uuid"] = f"{random.getrandbits(128):032x}"
        anns.append(ann)

    if overlap_test and num_annotations >= 2:
        # force two annotations to overlap in sample index space
        anns[1]["core:sample_start"] = anns[0]["core:sample_start"]
        anns[1]["core:sample_count"] = max(anns[0].get("core:sample_count", 10), 10)

    return anns

# --------------------------------------------------------------------------
# Dataset writer
# --------------------------------------------------------------------------

class DatasetSpec:
    def __init__(self, name, sigmf_dtype, num_samples=4096, field_level="typical",
                 num_captures=1, num_annotations=2, multichannel=False, num_channels=1,
                 non_conforming=False, non_conforming_kind=None, archive=False,
                 collection=False, metadata_only=False, extra_ns=False):
        self.name = name
        self.sigmf_dtype = sigmf_dtype
        self.num_samples = num_samples
        self.field_level = field_level
        self.num_captures = num_captures
        self.num_annotations = num_annotations
        self.multichannel = multichannel
        self.num_channels = num_channels
        self.non_conforming = non_conforming
        self.non_conforming_kind = non_conforming_kind
        self.archive = archive
        self.collection = collection
        self.metadata_only = metadata_only
        self.extra_ns = extra_ns


def write_dataset(spec: DatasetSpec, out_root: Path):
    ds_dir = out_root / spec.name
    ensure_dir(ds_dir)

    data_path = ds_dir / f"{spec.name}.sigmf-data"
    meta_path = ds_dir / f"{spec.name}.sigmf-meta"

    # ---- Generate raw sample data ----
    dtype_for_gen = spec.sigmf_dtype
    if spec.non_conforming_kind == "invalid_datatype_string":
        dtype_for_gen = random.choice(["r32f_le_junk"])  # garbage, generator falls back

    raw = b"" if spec.metadata_only else generate_samples(dtype_for_gen, spec.num_samples)

    # --- Non-conforming: truncate data (fewer bytes than header implies) ---
    if spec.non_conforming_kind == "truncated_data":
        raw = raw[: max(len(raw) // 2, 1)]

    # --- Non-conforming: extra trailing garbage bytes ---
    if spec.non_conforming_kind == "trailing_garbage":
        raw += os.urandom(37)

    # --- Non-conforming: empty data file but metadata claims samples ---
    if spec.non_conforming_kind == "empty_data_nonzero_meta":
        raw = b""

    if not spec.metadata_only:
        with open(data_path, "wb") as f:
            f.write(raw)

    # ---- Build metadata ----
    effective_dtype = spec.sigmf_dtype
    if spec.non_conforming_kind == "invalid_datatype_string":
        effective_dtype = random.choice(INVALID_DATATYPE_STRINGS)

    global_obj = build_global(
        effective_dtype, spec.num_samples, spec.field_level,
        extra_ns=spec.extra_ns, multichannel=spec.multichannel,
        num_channels=spec.num_channels,
        non_conforming_flag=(spec.non_conforming_kind == "non_conforming_dataset_field"),
    )

    if spec.metadata_only:
        global_obj["core:metadata_only"] = True
        global_obj["core:dataset"] = f"{spec.name}.sigmf-data"  # points to nonexistent/absent file

    if spec.non_conforming_kind == "missing_required_datatype":
        global_obj.pop("core:datatype", None)

    if spec.non_conforming_kind == "missing_required_version":
        global_obj.pop("core:version", None)

    if spec.non_conforming_kind == "wrong_type_sample_rate":
        global_obj["core:sample_rate"] = "not-a-number"  # should be numeric

    captures = build_captures(
        spec.num_samples, spec.field_level, spec.num_captures,
        multichannel=spec.multichannel, num_channels=spec.num_channels
    )

    if spec.non_conforming_kind == "empty_captures_array":
        captures = []

    if spec.non_conforming_kind == "capture_missing_sample_start":
        if captures:
            captures[0].pop("core:sample_start", None)

    if spec.non_conforming_kind == "captures_not_sorted":
        if len(captures) >= 2:
            captures[0]["core:sample_start"], captures[-1]["core:sample_start"] = \
                captures[-1]["core:sample_start"], captures[0]["core:sample_start"]

    annotations = build_annotations(
        spec.num_samples, spec.field_level, spec.num_annotations,
        overlap_test=(spec.non_conforming_kind == "overlapping_annotations"),
        out_of_bounds_test=(spec.non_conforming_kind == "annotation_out_of_bounds"),
    )

    if spec.non_conforming_kind == "annotation_missing_sample_start":
        if annotations:
            annotations[0].pop("core:sample_start", None)

    meta = {
        "global": global_obj,
        "captures": captures,
        "annotations": annotations,
    }

    if spec.collection:
        meta["collection"] = {
            "core:version": SIGMF_VERSION,
            "core:description": "Synthetic collection wrapper",
        }

    if spec.non_conforming_kind == "malformed_json":
        # Write intentionally broken JSON (trailing comma) — raw text write
        with open(meta_path, "w") as f:
            f.write(json.dumps(meta, indent=2)[:-1] + ",}\n")
    else:
        write_json(meta_path, meta)

    # ---- Archive packaging (.sigmf tar) ----
    if spec.archive:
        archive_path = out_root / f"{spec.name}.sigmf"
        with tarfile.open(archive_path, "w") as tar:
            arcbase = spec.name
            tar.add(meta_path, arcname=f"{arcbase}/{spec.name}.sigmf-meta")
            if not spec.metadata_only and data_path.exists():
                tar.add(data_path, arcname=f"{arcbase}/{spec.name}.sigmf-data")
        # keep loose files too for reader flexibility testing

    return ds_dir


# --------------------------------------------------------------------------
# Corpus generation plan
# --------------------------------------------------------------------------

def build_corpus_plan():
    specs = []

    # 1) Full datatype sweep — conforming, typical fields, small sample count
    for dtype_name in DTYPE_TABLE.keys():
        specs.append(DatasetSpec(
            name=f"dtype_{dtype_name}_conforming",
            sigmf_dtype=dtype_name,
            num_samples=random.choice([256, 1024, 4096]),
            field_level=random.choice(["minimal", "typical", "full"]),
            num_captures=random.choice([1, 2, 3]),
            num_annotations=random.choice([0, 1, 3]),
        ))

    # 2) Field-level variety matrix (minimal / typical / full) on a few core dtypes
    core_dtypes = ["cf32_le", "ci16_le", "ru8", "rf64_be"]
    for dtype_name in core_dtypes:
        for level in ["minimal", "typical", "full"]:
            specs.append(DatasetSpec(
                name=f"fieldlevel_{level}_{dtype_name}",
                sigmf_dtype=dtype_name,
                num_samples=2048,
                field_level=level,
                num_captures=2,
                num_annotations=2,
                extra_ns=(level == "full"),
            ))

    # 3) Multi-capture / segmented recordings
    for n in [1, 2, 5, 10]:
        specs.append(DatasetSpec(
            name=f"multicapture_{n}segments",
            sigmf_dtype="ci16_le",
            num_samples=8192,
            field_level="full",
            num_captures=n,
            num_annotations=4,
        ))

    # 4) Multi-channel datasets
    for ch in [1, 2, 4]:
        specs.append(DatasetSpec(
            name=f"multichannel_{ch}ch",
            sigmf_dtype="cf32_le",
            num_samples=4096,
            field_level="full",
            multichannel=True,
            num_channels=ch,
        ))

    # 5) Metadata-only dataset (core:metadata_only = true, no data file)
    specs.append(DatasetSpec(
        name="metadata_only_example",
        sigmf_dtype="cf32_le",
        num_samples=0,
        field_level="full",
        metadata_only=True,
    ))

    # 6) Archive packaging variants (.sigmf tar)
    for dtype_name in ["cf32_le", "ru16_be", "ci8"]:
        specs.append(DatasetSpec(
            name=f"archive_{dtype_name}",
            sigmf_dtype=dtype_name,
            num_samples=2048,
            field_level="typical",
            archive=True,
        ))

    # 7) Collection-wrapped dataset
    specs.append(DatasetSpec(
        name="collection_example",
        sigmf_dtype="cf32_le",
        num_samples=1024,
        field_level="typical",
        collection=True,
    ))

    # 8) Non-conforming dataset flag (core:dataset points to companion non-sigmf-typed payload)
    specs.append(DatasetSpec(
        name="nonconforming_dataset_field",
        sigmf_dtype="cf32_le",
        num_samples=1024,
        field_level="full",
        non_conforming=True,
        non_conforming_kind="non_conforming_dataset_field",
    ))

    # 9) Explicit malformed / broken datasets for negative testing
    broken_kinds = [
        "truncated_data",
        "trailing_garbage",
        "empty_data_nonzero_meta",
        "invalid_datatype_string",
        "missing_required_datatype",
        "missing_required_version",
        "wrong_type_sample_rate",
        "empty_captures_array",
        "capture_missing_sample_start",
        "captures_not_sorted",
        "overlapping_annotations",
        "annotation_out_of_bounds",
        "annotation_missing_sample_start",
        "malformed_json",
    ]
    for kind in broken_kinds:
        specs.append(DatasetSpec(
            name=f"broken_{kind}",
            sigmf_dtype="cf32_le",
            num_samples=1024,
            field_level="typical",
            num_captures=3,
            num_annotations=3,
            non_conforming=True,
            non_conforming_kind=kind,
        ))

    # 10) Edge cases: zero samples, single sample, huge sample count
    specs.append(DatasetSpec(name="edge_zero_samples", sigmf_dtype="ci16_le",
                              num_samples=0, field_level="typical"))
    specs.append(DatasetSpec(name="edge_one_sample", sigmf_dtype="ci16_le",
                              num_samples=1, field_level="typical"))
    specs.append(DatasetSpec(name="edge_large_dataset", sigmf_dtype="ci8",
                              num_samples=2_000_000, field_level="minimal"))

    return specs


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    ensure_dir(OUT_DIR)

    plan = build_corpus_plan()
    manifest = []

    for spec in plan:
        try:
            write_dataset(spec, OUT_DIR)
            manifest.append({"name": spec.name, "status": "ok",
                              "non_conforming_kind": spec.non_conforming_kind})
        except Exception as e:
            manifest.append({"name": spec.name, "status": f"generator_error: {e}",
                              "non_conforming_kind": spec.non_conforming_kind})

    write_json(OUT_DIR / "MANIFEST.json", {
        "generated": now_iso(),
        "seed": SEED,
        "sigmf_version": SIGMF_VERSION,
        "count": len(manifest),
        "datasets": manifest,
    })

    print(f"Generated {len(manifest)} dataset cases in {OUT_DIR.resolve()}")
    print("See MANIFEST.json for a full index of conforming/non-conforming cases.")


if __name__ == "__main__":
    main()