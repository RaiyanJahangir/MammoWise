#!/usr/bin/env python3
import os
import json
import ast
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import pandas as pd
from PIL import Image

import config


# -----------------------------
# Human-readable descriptions
# -----------------------------
LATERALITY_FULL = {"R": "right", "L": "left"}
VIEW_FULL = {"CC": "cranio-caudal (CC)", "MLO": "medio-lateral oblique (MLO)"}

BIRADS_DESCRIPTIONS = {
    "0": "Incomplete - Additional imaging evaluation and/or comparison to prior mammograms (or other imaging tests) is needed.",
    "1": "Negative. Healthy Breast.",
    "2": "Benign (non-cancerous) finding",
    "3": "Probably benign finding. Follow-up in a short time frame is suggested",
    "4": "Suspicious abnormality. Biopsy should be considered",
    "5": "Highly suggestive of malignancy. Appropriate action should be taken",
    "6": "Known biopsy-proven malignancy. Appropriate action should be taken",
}

DENSITY_DESCRIPTIONS = {
    "A": "Almost all fatty tissue.",
    "B": "Scattered areas of dense glandular and fibrous tissue (seen as white areas on the mammogram).",
    "C": "Heterogeneously Dense. More of the breast is made of dense glandular and fibrous tissue. This can make it hard to see small masses in or around the dense tissue, which also appear as white areas.",
    "D": "Extremely Dense. Hard to see masses or other findings that may appear as white areas on the mammogram.",
}

REQUIRED_VIEWS = [("L", "CC"), ("L", "MLO"), ("R", "CC"), ("R", "MLO")]


# -----------------------------
# Helpers
# -----------------------------
def safe_last_alnum(s: Any) -> str:
    """Return the last alphanumeric char in a value (e.g., 'Density C' -> 'C', 'BIRADS 4' -> '4')."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    t = str(s).strip()
    if not t:
        return ""
    for ch in reversed(t):
        if ch.isalnum():
            return ch
    return ""


def normalize_lat(x: Any) -> str:
    t = str(x).strip().upper() if x is not None else ""
    if not t:
        return ""
    # Accept "LEFT", "L", "Left" -> "L"; same for R
    if t.startswith("L"):
        return "L"
    if t.startswith("R"):
        return "R"
    return t[:1]  # fallback


def normalize_view(x: Any) -> str:
    t = str(x).strip().upper() if x is not None else ""
    if not t:
        return ""
    # Common variants could exist; keep it simple but robust
    if "MLO" in t:
        return "MLO"
    if t.endswith("CC") or "CC" in t:
        return "CC"
    return t


def birads_to_suspicion(birads_digit: str) -> str:
    """
    User requested only: healthy / benign / suspicious.

    Mapping:
      1 -> healthy
      2,3 -> benign
      4,5 -> suspicious
      0,6,missing -> suspicious (forced into the requested 3-class label space)
    """
    if birads_digit == "1":
        return "healthy"
    if birads_digit in {"2", "3"}:
        return "benign"
    # includes 4,5 and any other (0,6,"")
    return "suspicious"


def parse_categories(raw_cats: Any) -> List[str]:
    """
    VinDr finding_categories can appear as a stringified list like:
      "['Mass', 'Asymmetry']"
    or something messier. This tries safe parsing first.
    """
    if raw_cats is None or (isinstance(raw_cats, float) and pd.isna(raw_cats)):
        return []
    s = str(raw_cats).strip()
    if not s:
        return []
    try:
        out = ast.literal_eval(s)
        if isinstance(out, list):
            return [str(x).strip() for x in out if str(x).strip()]
        # sometimes it might be a single string
        return [str(out).strip()] if str(out).strip() else []
    except Exception:
        # fallback: strip brackets and split by comma
        s2 = s.strip("[]").replace("'", "").replace('"', "")
        return [x.strip() for x in s2.split(",") if x.strip()]


def index_all_pngs(root: Path) -> Dict[str, Path]:
    """
    Recursively index all PNG files by their stem.
    If duplicates exist, the first one encountered wins.
    """
    idx: Dict[str, Path] = {}
    for p in root.rglob("*.png"):
        if not p.is_file():
            continue
        stem = p.stem
        if stem not in idx:
            idx[stem] = p
    return idx


def pad_to_size(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    Pads image to target size, centering it.
    """
    if img.size == (target_w, target_h):
        return img
    canvas = Image.new("L" if img.mode == "L" else "RGB", (target_w, target_h), 0)
    x = (target_w - img.size[0]) // 2
    y = (target_h - img.size[1]) // 2
    canvas.paste(img, (x, y))
    return canvas


def merge_four_views(
    paths_by_view: Dict[Tuple[str, str], Path],
    out_path: Path,
) -> None:
    """
    Merge 4 views into a 2x2 grid:
      row 0: CC (L, R)
      row 1: MLO (L, R)
    """
    # Load images (convert to grayscale to keep consistent)
    imgs = {}
    for (lat, view) in REQUIRED_VIEWS:
        img = Image.open(paths_by_view[(lat, view)]).convert("L")
        imgs[(lat, view)] = img

    # Determine max width/height to pad each view
    max_w = max(im.size[0] for im in imgs.values())
    max_h = max(im.size[1] for im in imgs.values())

    # Pad all to same size
    for k in list(imgs.keys()):
        imgs[k] = pad_to_size(imgs[k], max_w, max_h)

    # Create grid canvas
    merged = Image.new("L", (max_w * 2, max_h * 2), 0)

    # Placement: (L,CC)=top-left, (R,CC)=top-right, (L,MLO)=bottom-left, (R,MLO)=bottom-right
    merged.paste(imgs[("L", "CC")], (0, 0))
    merged.paste(imgs[("R", "CC")], (max_w, 0))
    merged.paste(imgs[("L", "MLO")], (0, max_h))
    merged.paste(imgs[("R", "MLO")], (max_w, max_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(out_path)


def build_findings_sentence_and_flags(finding_df: pd.DataFrame, study_id: str) -> Tuple[str, int, int, int]:
    """
    For a given study_id, aggregates all finding rows to:
      - a human sentence mentioning view/laterality
      - mass/calcification/asymmetry flags (0/1) if present anywhere
    """
    frows = finding_df.loc[finding_df["study_id"] == study_id]

    findings_sentence = "Healthy Breast. No Findings"
    mass_flag = 0
    calcification_flag = 0
    asymmetry_flag = 0

    if frows.empty:
        return findings_sentence, mass_flag, calcification_flag, asymmetry_flag

    grouped: Dict[Tuple[str, str], Dict[str, int]] = {}
    all_no_find = True

    for _, row in frows.iterrows():
        lat = normalize_lat(row.get("laterality", ""))
        view = normalize_view(row.get("view_position", ""))

        cats = parse_categories(row.get("finding_categories", "[]"))
        for cat in cats:
            cat_l = cat.lower().strip()
            if not cat_l:
                continue

            # binary flags across entire merged study
            if "mass" in cat_l:
                mass_flag = 1
            if "calcification" in cat_l:
                calcification_flag = 1
            if "asymmetry" in cat_l:
                asymmetry_flag = 1

            if cat_l != "no finding":
                all_no_find = False

            key = (lat, view)
            grouped.setdefault(key, {}).setdefault(cat, 0)
            grouped[key][cat] += 1

    if all_no_find:
        return findings_sentence, mass_flag, calcification_flag, asymmetry_flag

    parts: List[str] = []
    for (lat, view), cat_counts in grouped.items():
        filtered = {cat: cnt for cat, cnt in cat_counts.items() if cat.lower() != "no finding"}
        if not filtered:
            continue

        lat_full = LATERALITY_FULL.get(lat, lat.lower())
        view_full = VIEW_FULL.get(view, view)

        if len(filtered) > 1:
            cats_list = [c.lower() for c in filtered.keys()]
            phrase = " and ".join(cats_list)
            parts.append(f"{phrase} found in {lat_full} {view_full}")
        else:
            cat, cnt = next(iter(filtered.items()))
            phrase = cat.lower()
            if cnt > 1:
                parts.append(f"{cnt} {phrase}s found in {lat_full} {view_full}")
            else:
                parts.append(f"{phrase} found in {lat_full} {view_full}")

    if parts:
        findings_sentence = " and ".join(parts)

    return findings_sentence, mass_flag, calcification_flag, asymmetry_flag


def main():
    # Load dataframes
    breast_df = pd.read_csv(config.BREAST_CSV)
    finding_df = pd.read_csv(config.FINDING_CSV)

    # Ensure output dirs exist
    config.MERGED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.MERGED_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Index all PNGs (recursive)
    png_index = index_all_pngs(config.VIN_DR_IMAGES_ROOT)
    print(f"Indexed {len(png_index)} PNGs under {config.VIN_DR_IMAGES_ROOT}")

    # Basic schema check
    needed_breast_cols = {"study_id", "image_id", "laterality", "view_position", "breast_density", "breast_birads"}
    missing = needed_breast_cols - set(breast_df.columns)
    if missing:
        raise ValueError(f"breast CSV missing columns: {sorted(missing)}")

    if "study_id" not in finding_df.columns:
        raise ValueError("finding CSV must have 'study_id' column")
    if "finding_categories" not in finding_df.columns:
        raise ValueError("finding CSV must have 'finding_categories' column")

    # Group breast rows by study_id
    studies = list(breast_df.groupby("study_id"))
    total = len(studies)
    print(f"Found {total} studies in breast CSV")

    written = 0
    skipped_incomplete_views = 0
    skipped_missing_files = 0

    for i, (study_id, sdf) in enumerate(studies, start=1):
        # Build mapping (lat, view) -> image_path for this study
        paths_by_view: Dict[Tuple[str, str], Path] = {}
        for _, r in sdf.iterrows():
            lat = normalize_lat(r.get("laterality"))
            view = normalize_view(r.get("view_position"))
            img_id = str(r.get("image_id")).strip()
            img_stem = Path(img_id).stem  # handle if csv has extension

            if (lat, view) in paths_by_view:
                # keep first; ignore duplicates
                continue

            p = png_index.get(img_stem)
            if p is not None:
                paths_by_view[(lat, view)] = p

        # Check we have all 4 required views
        if any(k not in paths_by_view for k in REQUIRED_VIEWS):
            skipped_incomplete_views += 1
            # If breast CSV has all 4 but file missing, count separately
            # (heuristic: if breast rows contain all 4 but paths missing)
            breast_has_all = True
            breast_keys = {(normalize_lat(r.get("laterality")), normalize_view(r.get("view_position"))) for _, r in sdf.iterrows()}
            for k in REQUIRED_VIEWS:
                if k not in breast_keys:
                    breast_has_all = False
                    break
            if breast_has_all:
                skipped_missing_files += 1

            if i % 50 == 0:
                print(f"[{i}/{total}] written={written} skipped_incomplete={skipped_incomplete_views} skipped_missing_files={skipped_missing_files}")
            continue

        # Decide merged output image path
        merged_img_path = config.MERGED_IMAGES_DIR / f"{study_id}.png"

        # Merge images into one
        try:
            merge_four_views(paths_by_view, merged_img_path)
        except Exception as e:
            skipped_missing_files += 1
            print(f"Warning: failed merging study_id={study_id} due to: {e}")
            continue

        # Aggregate density + birads from breast rows (use first non-empty density, max birads)
        density_char = ""
        birads_max_digit = ""

        for _, r in sdf.iterrows():
            d = safe_last_alnum(r.get("breast_density"))
            b = safe_last_alnum(r.get("breast_birads"))
            if not density_char and d:
                density_char = d.upper()
            if b.isdigit():
                if not birads_max_digit or int(b) > int(birads_max_digit):
                    birads_max_digit = b

        density_desc = DENSITY_DESCRIPTIONS.get(density_char, "")
        density_text = (
            f"Density {density_char} - {density_desc}"
            if density_char and density_desc
            else (f"Density {density_char}" if density_char else "")
        )

        birads_desc = BIRADS_DESCRIPTIONS.get(birads_max_digit, "")
        birads_text = (
            f"BIRADS {birads_max_digit} - {birads_desc}"
            if birads_max_digit and birads_desc
            else (f"BIRADS {birads_max_digit}" if birads_max_digit else "BIRADS unknown")
        )

        suspicion_label = birads_to_suspicion(birads_max_digit)

        # Findings sentence + flags (from finding_df using study_id)
        findings_sentence, mass_flag, calc_flag, asym_flag = build_findings_sentence_and_flags(finding_df, study_id)

        # Build report
        report = {
            "image_id": str(merged_img_path),
            "breast_density": density_text,
            "birads": birads_text,
            "findings": findings_sentence,
            "mass": int(mass_flag),
            "calcification": int(calc_flag),
            "asymmetry": int(asym_flag),
            "suspicion": suspicion_label,  # renamed key + requested value space
            "dataset": "vindr",
            "study_id": str(study_id),
        }

        # Write JSON
        out_path = config.MERGED_REPORTS_DIR / f"{study_id}.json"
        with out_path.open("w", encoding="utf-8") as jf:
            json.dump(report, jf, indent=2)

        written += 1

        if i % 50 == 0:
            print(f"[{i}/{total}] written={written} skipped_incomplete={skipped_incomplete_views} skipped_missing_files={skipped_missing_files}")

    print("Done.")
    print(f"Written merged studies: {written}")
    print(f"Skipped (missing any required view): {skipped_incomplete_views}")
    print(f"Skipped (likely missing image files / merge errors): {skipped_missing_files}")


if __name__ == "__main__":
    main()
