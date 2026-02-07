from datasets import Dataset, Features, Value
from pathlib import Path
from PIL import Image
import json, re

image_dir = Path("...")
report_dir = Path("...")
load_images = False
progress_every = 10

TARGET_CLASSES = {"healthy", "benign", "suspicious"}

# Optional fallback: map BI-RADS → suspicion if needed
# 1 -> healthy; 2/3 -> benign; 4/5 -> suspicious; others -> None
def _suspicion_from_birads(v):
    if v is None:
        return None
    try:
        b = int(v)
    except Exception:
        # Try to extract a single digit 0–6 if v is a string like "BIRADS 3"
        if isinstance(v, str):
            m = re.search(r"\b([0-6])\b", v)
            if not m:
                return None
            b = int(m.group(1))
        else:
            return None
    if b == 1:
        return "healthy"
    if b in (2, 3):
        return "benign"
    if b in (4, 5):
        return "suspicious"
    return None

def _normalize_suspicion_str(s: str):
    """Normalize free-text to one of TARGET_CLASSES (case-insensitive)."""
    if not s:
        return None
    t = s.strip().lower()
    # accept common variants/synonyms
    if t in TARGET_CLASSES:
        return t
    if t in {"malign", "suspicion", "cancer", "suspicious."}:
        return "suspicious"
    # try to find a keyword inside longer text
    if re.search(r"\bhealthy\b", t):
        return "healthy"
    if re.search(r"\bbenign\b", t):
        return "benign"
    if re.search(r"\bmalignan\w*\b", t) or re.search(r"\bcancer\b", t):
        return "suspicious"
    return None

def _extract_suspicion(v):
    """
    Return one of: 'healthy' | 'benign' | 'suspicious' | None.
    Accepts exact strings, booleans/numbers not used here, and dicts with nested keys.
    """
    if v is None:
        return None
    # Direct string
    if isinstance(v, str):
        return _normalize_suspicion_str(v)
    # Dict with nested keys
    if isinstance(v, dict):
        for k in ("suspicion", "label", "status"):
            if k in v:
                out = _extract_suspicion(v[k])
                if out is not None:
                    return out
        return None
    # Anything else -> not supported (avoid guessing)
    return None

png_files = sorted(image_dir.glob("*.png"))
samples, bad = [], 0

for idx, p in enumerate(png_files, 1):
    jp = report_dir / f"{p.stem}.json"
    if not jp.exists():
        continue
    try:
        with open(jp, "r") as fh:
            d = json.load(fh)
    except json.JSONDecodeError:
        continue

    # 1) Try the explicit 'suspicion' key
    suspicion = _extract_suspicion(d.get("suspicion"))

    # 2) Fallback from BI-RADS if needed (optional)
    if suspicion is None:
        suspicion = _suspicion_from_birads(d.get("birads"))

    if suspicion is None:
        bad += 1
        continue

    samples.append({
        "image": str(p) if not load_images else Image.open(p).convert("RGB"),
        "suspicion": suspicion,  # 'healthy' | 'benign' | 'suspicious'
    })

    if progress_every and idx % progress_every == 0:
        print(f"[progress] {idx}/{len(png_files)}  kept={len(samples)}  bad_suspicion={bad}")

features = Features({
    "image": Value("string"),     # or datasets.Image() if you want to decode later
    "suspicion": Value("string") # keep as string class label
})

ds = Dataset.from_list(samples, features=features)
split = ds.train_test_split(test_size=0.1, seed=42)
split.save_to_disk("...")
