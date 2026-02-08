import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np

import config


# ---------------------------------------------------------------------------
# ✅ Paths/params now come from config.py
# ---------------------------------------------------------------------------

IMAGES_DIR: Path  = Path(config.AUG_VINDR_IN_IMAGES_DIR)
REPORTS_DIR: Path = Path(config.AUG_VINDR_IN_REPORTS_DIR)

OUT_IMAGES_DIR: Path  = Path(config.AUG_VINDR_OUT_IMAGES_DIR)
OUT_REPORTS_DIR: Path = Path(config.AUG_VINDR_OUT_REPORTS_DIR)

TARGET_PER_CLASS: Dict[int, int] = dict(config.AUG_TARGET_PER_CLASS)
IMAGE_EXTS: List[str] = list(getattr(config, "AUG_IMAGE_EXTS", [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm"]))

DRY_RUN: bool = bool(getattr(config, "AUG_DRY_RUN", False))
AUG_SEED: int = int(getattr(config, "AUG_SEED", 1337))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BIRADS_DIGIT = re.compile(r"([0-6])")

def normalize_birads(value) -> Optional[int]:
    """Return an int BI-RADS value if possible, else None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = int(value)
        return v if 0 <= v <= 6 else None
    if isinstance(value, str):
        m = BIRADS_DIGIT.search(value)
        if m:
            v = int(m.group(1))
            return v if 0 <= v <= 6 else None
    return None

def find_image_for_report(stem: str, images_dir: Path, image_exts: List[str]) -> Optional[Path]:
    for ext in image_exts:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def safe_read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# --- Left/right swap in free text (case-insensitive, whole-word) ---
def swap_left_right(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    patt_left  = re.compile(r"\b(left)\b", re.IGNORECASE)
    patt_right = re.compile(r"\b(right)\b", re.IGNORECASE)
    LEFT_PH, RIGHT_PH = "<<__LEFT__>>", "<<__RIGHT__>>"
    tmp = patt_left.sub(LEFT_PH, text)
    tmp = patt_right.sub(RIGHT_PH, tmp)
    tmp = tmp.replace(LEFT_PH, "right").replace(RIGHT_PH, "left")
    return tmp

def update_report_for_flip(report: dict) -> dict:
    rep = json.loads(json.dumps(report))  # deep copy
    findings = rep.get("findings")
    if isinstance(findings, str):
        rep["findings"] = swap_left_right(findings)
    elif isinstance(findings, list):
        rep["findings"] = [swap_left_right(x) if isinstance(x, str) else x for x in findings]
    elif isinstance(findings, dict):
        rep["findings"] = {k: (swap_left_right(v) if isinstance(v, str) else v) for k, v in findings.items()}
    return rep

# --- Image I/O & augmentations ---
def load_image(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    return img

def save_image(img: Image.Image, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

def aug_flip(img: Image.Image) -> Image.Image:
    return ImageOps.mirror(img)

def aug_small_rotate(img: Image.Image, degrees: float = 2.0) -> Image.Image:
    return img.rotate(degrees, resample=Image.BICUBIC, expand=False, fillcolor=0)

def aug_small_rotate_rand(img: Image.Image, max_degrees: float = 5.0) -> Image.Image:
    deg = random.uniform(-max_degrees, max_degrees)
    return aug_small_rotate(img, deg)


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------

def collect_pairs_by_birads() -> Dict[int, List[Tuple[Path, Path, dict]]]:
    """
    Returns: {birads: [(img_path, report_path, report_dict), ...]}
    Only pairs with both image and valid report & birads.
    """
    by_class: Dict[int, List[Tuple[Path, Path, dict]]] = {}
    report_files = sorted(REPORTS_DIR.glob("*.json"))
    missing_img, bad_json, missing_birads = 0, 0, 0

    for rpath in report_files:
        stem = rpath.stem
        data = safe_read_json(rpath)
        if data is None:
            bad_json += 1
            continue

        birads_raw = data.get("birads", data.get("BI-RADS", data.get("birads_category")))
        b = normalize_birads(birads_raw)
        if b is None:
            missing_birads += 1
            continue

        ipath = find_image_for_report(stem, IMAGES_DIR, IMAGE_EXTS)
        if ipath is None:
            missing_img += 1
            continue

        by_class.setdefault(b, []).append((ipath, rpath, data))

    for b in by_class:
        by_class[b].sort(key=lambda t: t[0].name)

    print(f"Collected pairs. Missing image: {missing_img}, bad JSON: {bad_json}, missing birads: {missing_birads}")
    return by_class

def write_pair(img: Image.Image, img_name: str, out_img_dir: Path,
               report: dict, out_rep_dir: Path, new_image_id: str):
    """Write image and report JSON with updated image_id and file names."""
    img_out_path = out_img_dir / img_name
    rep_out_path = out_rep_dir / (Path(img_name).stem + ".json")
    if DRY_RUN:
        return
    save_image(img, img_out_path)
    rep = json.loads(json.dumps(report))
    rep["image_id"] = new_image_id
    write_json(rep_out_path, rep)

def run():
    # reproducible
    random.seed(AUG_SEED)

    # sanity checks
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"IMAGES_DIR not found: {IMAGES_DIR}")
    if not REPORTS_DIR.exists():
        raise FileNotFoundError(f"REPORTS_DIR not found: {REPORTS_DIR}")

    OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    pairs_by_b = collect_pairs_by_birads()

    for b, target in TARGET_PER_CLASS.items():
        out_img_dir = OUT_IMAGES_DIR / f"BIRADS_{b}"
        out_rep_dir = OUT_REPORTS_DIR / f"BIRADS_{b}"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_rep_dir.mkdir(parents=True, exist_ok=True)

        pool: List[Tuple[str, Image.Image, dict]] = []
        used_names = set()

        src_pairs = pairs_by_b.get(b, [])

        # 1) Copy originals (up to target)
        for ipath, _, rep in src_pairs[:target]:
            img = load_image(ipath)
            base = ipath.stem
            ext  = ipath.suffix or ".png"
            name = f"{base}{ext}"
            if name in used_names:
                name = f"{base}_orig{ext}"
            used_names.add(name)
            write_pair(img, name, out_img_dir, {**rep, "image_id": name}, out_rep_dir, name)
            pool.append((name, img, {**rep, "image_id": name}))

        count = len(pool)
        print(f"BI-RADS {b}: copied originals: {count}/{target}")

        if b in (1, 2):
            # No augmentation for BIRADS 1 & 2
            print(f"BI-RADS {b}: final count written: {count}/{target}")
            continue

        def next_name(stem: str, ext: str, suffix: str) -> str:
            candidate = f"{stem}_{suffix}{ext}"
            idx = 2
            while candidate in used_names:
                candidate = f"{stem}_{suffix}{idx}{ext}"
                idx += 1
            used_names.add(candidate)
            return candidate

        # --- step A: rotate all originals by <5° ---
        if count < target and len(pool) > 0:
            added = 0
            base_snapshot = list(pool)
            for orig_name, orig_img, orig_rep in base_snapshot:
                if count >= target:
                    break
                stem, ext = Path(orig_name).stem, Path(orig_name).suffix
                img_rot = aug_small_rotate_rand(orig_img, max_degrees=5.0)
                new_name = next_name(stem, ext, "rot")
                rep_new  = {**orig_rep, "image_id": new_name}
                write_pair(img_rot, new_name, out_img_dir, rep_new, out_rep_dir, new_name)
                pool.append((new_name, img_rot, rep_new))
                count += 1
                added += 1
            print(f"BI-RADS {b}: rotated-all added: {added}, total: {count}/{target}")

        # --- step B: flip entire pool once (with left↔right swap in findings) ---
        if count < target and len(pool) > 0:
            added = 0
            base_snapshot = list(pool)
            for orig_name, orig_img, orig_rep in base_snapshot:
                if count >= target:
                    break
                stem, ext = Path(orig_name).stem, Path(orig_name).suffix
                img_flip = aug_flip(orig_img)
                rep_flip = update_report_for_flip(orig_rep)
                new_name = next_name(stem, ext, "flip")
                rep_flip = {**rep_flip, "image_id": new_name}
                write_pair(img_flip, new_name, out_img_dir, rep_flip, out_rep_dir, new_name)
                pool.append((new_name, img_flip, rep_flip))
                count += 1
                added += 1
            print(f"BI-RADS {b}: flipped pool added: {added}, total: {count}/{target}")

        # --- step C: if still short → add small-rotation chunks until target ---
        while count < target and len(pool) > 0:
            need = target - count
            added = 0
            base_snapshot = list(pool)
            for orig_name, orig_img, orig_rep in base_snapshot:
                if added >= need or count >= target:
                    break
                stem, ext = Path(orig_name).stem, Path(orig_name).suffix
                img_rot = aug_small_rotate_rand(orig_img, max_degrees=5.0)
                new_name = next_name(stem, ext, "rotx")
                rep_new  = {**orig_rep, "image_id": new_name}
                write_pair(img_rot, new_name, out_img_dir, rep_new, out_rep_dir, new_name)
                pool.append((new_name, img_rot, rep_new))
                count += 1
                added += 1
            print(f"BI-RADS {b}: extra rotate chunk added: {added}, total: {count}/{target}")

        print(f"BI-RADS {b}: final count written (including augs): {count}/{target}")

    if DRY_RUN:
        print("\nDRY_RUN=True → nothing was written.")
    else:
        print("\nDone. Outputs written to:")
        print(f"  Images : {OUT_IMAGES_DIR}")
        print(f"  Reports: {OUT_REPORTS_DIR}")


if __name__ == "__main__":
    run()
