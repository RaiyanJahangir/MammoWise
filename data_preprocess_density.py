from datasets import Dataset, Features, Value
from pathlib import Path
from PIL import Image
import json, re

image_dir = Path("...")
report_dir = Path("...")
load_images = False
progress_every = 10

# Optional numeric fallback mapping if some reports use 1-4
NUM_TO_DENS = {1: "A", 2: "B", 3: "C", 4: "D"}

def _extract_density(v):
    """Return 'A'/'B'/'C'/'D' or None."""
    if v is None:
        return None
    # numeric → map 1-4 to A-D
    if isinstance(v, (int, float)):
        v = int(v)
        return NUM_TO_DENS.get(v)
    # dict/object with possible keys
    if isinstance(v, dict):
        for k in ("density", "breast_density", "density_char"):
            if k in v:
                return _extract_density(v[k])
        return None
    # strings like "Density C - ..." or just "C"
    if isinstance(v, str):
        s = v.strip()
        # common explicit pattern "Density X"
        m = re.search(r"\b([A-Da-d])\b", s)
        if m:
            return m.group(1).upper()
        # fallback: last char if it looks like a density
        last = s[-1:].upper()
        if last in {"A", "B", "C", "D"}:
            return last
    return None

png_files = sorted(image_dir.glob("*.png"))
samples, bad = [], 0

for idx, p in enumerate(png_files, 1):
    jp = report_dir / f"{p.stem}.json"
    if not jp.exists():
        continue

    try:
        d = json.load(open(jp))
    except json.JSONDecodeError:
        continue

    # extract density from common keys
    dens = (
        d.get("breast_density")
        or d.get("density")
        or d
    )

    if dens is None:
        bad += 1
        continue

    # --------------------------------------------------------
    # NEW: Save clean JSON file containing ONLY breast_density
    # --------------------------------------------------------
    out_json = {
        "breast_density": str(dens)
    }

    out_path = report_dir / f"{p.stem}_clean.json"
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=4)

    # keep dataset sample (optional)
    samples.append({
        "image": str(p) if not load_images else Image.open(p).convert("RGB"),
        "density": dens,
    })

    if progress_every and idx % progress_every == 0:
        print(f"[progress] {idx}/{len(png_files)}  kept={len(samples)}  bad_density={bad}")

features = Features({
    "image": Value("string"),   # use datasets.Image() if you prefer decoding at load time
    "density": Value("string"), # keep as string label 'A'...'D'
})

ds = Dataset.from_list(samples, features=features)
split = ds.train_test_split(test_size=0.1, seed=42)
split.save_to_disk("...")
