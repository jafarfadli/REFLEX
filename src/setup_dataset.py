import shutil
import random
from pathlib import Path

# ── Split ratios ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED        = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"
PROCESSED   = DATASET_DIR / "processed"

# ── Dataset → classifier mapping ─────────────────────────────────────────────
DATASET_MAP = {
    "yawn_clf": {
        "source_dir": DATASET_DIR / "yawn_dataset",
        "classes": {
            "yawn":    "yawn",
            "no yawn": "no_yawn",
        },
    }
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_IMAGES_PER_CLASS = 20

# ─────────────────────────────────────────────────────────────────────────────

def is_image(path: Path) -> bool:
    """True if path is an image file and NOT a dotfile."""
    return (path.is_file()
            and path.suffix.lower() in IMG_EXTS
            and not path.name.startswith("."))


def gather_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if is_image(p)])


def remove_dotfiles(root: Path) -> int:
    """Recursively delete dotfiles (.DS_Store etc.) under root. Returns count."""
    if not root.exists():
        return 0
    count = 0
    for path in root.rglob(".*"):
        if path.is_file():
            try:
                path.unlink()
                count += 1
            except OSError:
                pass
    return count


def split_and_copy(images: list[Path], splits: dict[str, Path], seed: int = SEED):
    random.seed(seed)
    random.shuffle(images)

    n       = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    buckets = {
        "train": images[:n_train],
        "val":   images[n_train : n_train + n_val],
        "test":  images[n_train + n_val :],
    }

    for split_name, bucket in buckets.items():
        dst = splits[split_name]
        dst.mkdir(parents=True, exist_ok=True)
        for img in bucket:
            shutil.copy2(img, dst / img.name)
        print(f"      {split_name:5s}: {len(bucket):5d} images  →  {dst.relative_to(BASE_DIR)}")


def main():
    print("\n" + "=" * 60)
    print("  Fatigue Detection — Dataset Setup")
    print("=" * 60)

    # ── Pre-clean dotfiles from source + previous processed dirs ──────────────
    cleaned = 0
    for path in (DATASET_DIR,):
        cleaned += remove_dotfiles(path)
    if cleaned:
        print(f"\n  Cleaned {cleaned} dotfiles (.DS_Store etc.) from dataset/")

    # ── Wipe stale processed/ so re-runs don't accumulate ─────────────────────
    if PROCESSED.exists():
        shutil.rmtree(PROCESSED)
        print(f"  Cleared previous processed/ tree")

    total_copied = 0

    for clf_name, config in DATASET_MAP.items():
        src_root = config["source_dir"]
        print(f"\n[{clf_name}]  source: {src_root.relative_to(BASE_DIR)}")

        if not src_root.exists():
            print(f"  ⚠  Directory not found, skipping: {src_root}")
            continue

        for src_cls, dst_cls in config["classes"].items():
            src_dir = src_root / src_cls

            if not src_dir.exists():
                print(f"  ⚠  Class folder not found: {src_dir}  — skipping")
                continue

            images = gather_images(src_dir)
            print(f"\n  Class '{src_cls}' → '{dst_cls}'  ({len(images)} images)")

            if len(images) < MIN_IMAGES_PER_CLASS:
                print(f"  ⚠  Too few images ({len(images)} < {MIN_IMAGES_PER_CLASS}). Skipping.")
                continue

            splits = {
                split: PROCESSED / clf_name / split / dst_cls
                for split in ("train", "val", "test")
            }
            split_and_copy(images, splits)
            total_copied += len(images)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Done!  {total_copied} images organized.")
    print("=" * 60)
    print("\n  Processed structure:")

    for clf_name in DATASET_MAP:
        clf_dir = PROCESSED / clf_name
        if not clf_dir.exists():
            continue
        print(f"\n  {clf_dir.relative_to(BASE_DIR)}/")
        for split in ("train", "val", "test"):
            split_dir = clf_dir / split
            if not split_dir.exists():
                continue
            count = sum(1 for p in split_dir.rglob("*") if is_image(p))
            print(f"    {split:5s}/  ({count} images)")
            for cls_dir in sorted(split_dir.iterdir()):
                if cls_dir.name.startswith(".") or not cls_dir.is_dir():
                    continue
                n = sum(1 for p in cls_dir.iterdir() if is_image(p))
                print(f"      {cls_dir.name}/  ({n})")

    print()


if __name__ == "__main__":
    main()