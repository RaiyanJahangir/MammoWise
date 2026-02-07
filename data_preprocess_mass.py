from datasets import Dataset, Features, Value
from pathlib import Path
from PIL import Image
import json, re

image_dir = Path("...")
report_dir = Path("...")
load_images = False
progress_every = 10

def _extract_mass_from_findings(findings):
    """
    Return 1 if the 'findings' text mentions a mass/masses, else 0.
    Uses regex on the findings string.
    """
    if findings is None:
        # If there is no findings text, treat as no mass
        return 0

    s = str(findings).lower()
    # Look for 'mass' or 'masses' as a whole word
    if re.search(r"\bmass(es)?\b", s):
        return 1
    return 0

png_files = sorted(image_dir.glob("*.png"))
samples = []
bad = 0

for idx, p in enumerate(png_files, 1):
    jp = report_dir / f"{p.stem}.json"
    if not jp.exists():
        continue

    try:
        with open(jp, "r") as fh:
            d = json.load(fh)
    except json.JSONDecodeError:
        continue

    # Get mass label from 'findings' via regex
    findings = d.get("findings", "")
    mval = _extract_mass_from_findings(findings)

    # mval is guaranteed 0 or 1 here, but keep the pattern in case you later change it
    if mval is None:
        bad += 1
        continue

    samples.append({
        "image": str(p) if not load_images else Image.open(p).convert("RGB"),
        "mass": int(mval),  # guaranteed 0 or 1
    })

    if progress_every and idx % progress_every == 0:
        print(f"[progress] {idx}/{len(png_files)}  kept={len(samples)}  bad_mass={bad}")

features = Features({
    "image": Value("string"),  # use datasets.Image() if you want decoding at load time
    "mass": Value("int64"),
})

ds = Dataset.from_list(samples, features=features)
split = ds.train_test_split(test_size=0.1, seed=42)
split.save_to_disk("...")
