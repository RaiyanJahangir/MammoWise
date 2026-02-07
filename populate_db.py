#!/usr/bin/env python3
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from PIL import Image
import open_clip

import chromadb


# -----------------------
# Import your config paths
# -----------------------
import config


# -----------------------
# Utilities
# -----------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(images_dir: Path, exts=(".png", ".jpg", ".jpeg")) -> List[Path]:
    out = []
    for ext in exts:
        out.extend(images_dir.rglob(f"*{ext}"))
    return sorted([p for p in out if p.is_file()])


def make_train_test_split(items: List[Path], train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[Path], List[Path]]:
    items = list(items)
    rnd = random.Random(seed)
    rnd.shuffle(items)
    n_train = int(len(items) * train_ratio)
    return items[:n_train], items[n_train:]


def save_split(out_dir: Path, train_items: List[Path], test_items: List[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_ids.txt").write_text("\n".join([p.stem for p in train_items]) + "\n", encoding="utf-8")
    (out_dir / "test_ids.txt").write_text("\n".join([p.stem for p in test_items]) + "\n", encoding="utf-8")


def load_json_if_exists(json_path: Path) -> Dict:
    if json_path.exists() and json_path.is_file():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def chunked(lst: List, batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def normalize_metadata(md: Dict) -> Dict:
    """
    Keep only simple JSON-serializable primitives for Chroma metadata.
    (Chroma metadata must be dict[str, str|int|float|bool].)
    """
    keep = {}
    for k, v in md.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            keep[k] = v
        else:
            # fallback to string
            keep[k] = str(v)
    return keep


# -----------------------
# Embedding + Indexing
# -----------------------
@torch.no_grad()
def embed_batch(model, preprocess, device: str, paths: List[Path]) -> List[List[float]]:
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        imgs.append(preprocess(im))
    batch = torch.stack(imgs, dim=0).to(device)
    feats = model.encode_image(batch)
    feats = feats.float().cpu().numpy()
    return [f.flatten().tolist() for f in feats]


def build_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    model,
    preprocess,
    device: str,
    train_images: List[Path],
    reports_dir: Optional[Path],
    batch_size: int = 32,
):
    """
    Index only training images into a Chroma collection.
    If reports_dir is provided, each image's metadata is loaded from <stem>.json in that directory.
    """
    col = client.get_or_create_collection(name=collection_name)

    # Optional: avoid re-adding ids that already exist (best-effort)
    # We'll keep it simple: you can delete collection manually if you want a fresh index.
    total = len(train_images)
    print(f"[{collection_name}] Indexing {total} training images...")

    written = 0
    for batch_paths in chunked(train_images, batch_size):
        ids = [p.stem for p in batch_paths]
        uris = [str(p) for p in batch_paths]

        # metadata
        metadatas = []
        if reports_dir is not None:
            for p in batch_paths:
                md = load_json_if_exists(reports_dir / f"{p.stem}.json")
                md = normalize_metadata(md)
                # always include dataset id and image id for tracing
                md.setdefault("image_id", p.stem)
                metadatas.append(md)
        else:
            for p in batch_paths:
                metadatas.append({"image_id": p.stem})

        # embeddings
        embs = embed_batch(model, preprocess, device, batch_paths)

        col.add(ids=ids, uris=uris, embeddings=embs, metadatas=metadatas)
        written += len(batch_paths)

        if written % (batch_size * 10) == 0 or written == total:
            print(f"[{collection_name}] {written}/{total} indexed")

    print(f"[{collection_name}] Done. Collection count: {col.count()}")
    return col


def main():
    # -----------------------
    # Settings
    # -----------------------
    SEED = 42
    TRAIN_RATIO = 0.8
    BATCH_SIZE = 32

    set_seed(SEED)

    # -----------------------
    # Chroma client (persistent)
    # -----------------------
    chroma_root = getattr(config, "CHROMA_ROOT", Path("chroma"))
    chroma_root = Path(chroma_root)
    chroma_root.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_root))

    # -----------------------
    # OpenCLIP model
    # -----------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu",
        pretrained="openai",
    )
    model = model.to(device).eval()
    print(f"OpenCLIP loaded on {device}. Chroma root: {chroma_root}")

    # -----------------------
    # Dataset 1: VinDr merged
    # -----------------------
    vindr_images_dir = Path(getattr(config, "VINDR_MERGED_IMAGES_DIR", config.MERGED_IMAGES_DIR))
    vindr_reports_dir = Path(getattr(config, "VINDR_MERGED_REPORTS_DIR", config.MERGED_REPORTS_DIR))

    vindr_imgs = list_images(vindr_images_dir, exts=(".png",))
    if not vindr_imgs:
        print(f"Warning: No VinDr merged images found in {vindr_images_dir}")
    else:
        vindr_train, vindr_test = make_train_test_split(vindr_imgs, TRAIN_RATIO, SEED)
        save_split(chroma_root / "splits" / "vindr", vindr_train, vindr_test)

        build_collection(
            client=client,
            collection_name="vindr_train_db",
            model=model,
            preprocess=preprocess,
            device=device,
            train_images=vindr_train,
            reports_dir=vindr_reports_dir,
            batch_size=BATCH_SIZE,
        )

    # -----------------------
    # Dataset 2: DMID
    # -----------------------
    dmid_images_dir = Path(getattr(config, "DMID_IMAGES_DIR", ""))
    dmid_reports_dir = Path(getattr(config, "DMID_JSON_OUTPUT_DIR", ""))

    dmid_imgs = list_images(dmid_images_dir, exts=(".png", ".jpg", ".jpeg"))
    if not dmid_imgs:
        print(f"Warning: No DMID images found in {dmid_images_dir}")
    else:
        dmid_train, dmid_test = make_train_test_split(dmid_imgs, TRAIN_RATIO, SEED)
        save_split(chroma_root / "splits" / "dmid", dmid_train, dmid_test)

        # If DMID json reports are absent, you can set reports_dir=None
        reports_dir = dmid_reports_dir if dmid_reports_dir.exists() else None

        build_collection(
            client=client,
            collection_name="dmid_train_db",
            model=model,
            preprocess=preprocess,
            device=device,
            train_images=dmid_train,
            reports_dir=reports_dir,
            batch_size=BATCH_SIZE,
        )

    print("All done.")


if __name__ == "__main__":
    main()
