import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
MODELS_DIR    = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

BATCH_SIZE  = 32
EPOCHS      = 15
LR          = 1e-4
IMG_SIZE    = 224
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSIFIERS = ["yawn_clf", "eye_clf"]

# ── Transforms ────────────────────────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

def get_transform(augment: bool = False) -> transforms.Compose:
    base = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if augment:
        # Slightly stronger augmentation since self-collected sets are smaller
        base += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        ]
    base += [
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]
    return transforms.Compose(base)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 2) -> nn.Module:
    """MobileNetV2 with frozen backbone, custom head."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.last_channel
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


# ── Train / Eval loops ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast("cuda"):
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs)
        loss = criterion(out, labels)

        total_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


# ── Main training routine ─────────────────────────────────────────────────────

def train_classifier(clf_name: str, epochs: int = EPOCHS):
    data_dir = PROCESSED_DIR / clf_name
    if not data_dir.exists():
        print(f"  [SKIP] {clf_name}: processed data not found at {data_dir}")
        print("         Run  python setup_dataset.py  first.")
        return None, None

    train_ds = datasets.ImageFolder(data_dir / "train", transform=get_transform(augment=True))
    val_ds   = datasets.ImageFolder(data_dir / "val",   transform=get_transform(augment=False))
    test_ds  = datasets.ImageFolder(data_dir / "test",  transform=get_transform(augment=False))

    print(f"\n  Classes (index): {train_ds.class_to_idx}")
    print(f"  Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model     = build_model(num_classes=len(train_ds.classes)).to(DEVICE)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None

    best_val_acc    = 0.0
    best_model_path = MODELS_DIR / f"{clf_name}.pt"

    print(f"\n  {'Epoch':>5}  {'TrainLoss':>9}  {'TrainAcc':>8}  {'ValLoss':>7}  {'ValAcc':>7}  {'LR':>8}")
    print("  " + "-" * 58)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, scaler)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion)
        scheduler.step(va_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        flag   = "  ←  best" if va_acc > best_val_acc else ""

        print(f"  {epoch:5d}  {tr_loss:9.4f}  {tr_acc:8.4f}  {va_loss:7.4f}  {va_acc:7.4f}  "
              f"{lr_now:8.2e}  {time.time()-t0:.1f}s{flag}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(
                {
                    "model_state":  model.state_dict(),
                    "classes":      train_ds.classes,
                    "class_to_idx": train_ds.class_to_idx,
                    "val_acc":      va_acc,
                },
                best_model_path,
            )

    ckpt = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    _, test_acc = eval_epoch(model, test_loader, criterion)

    print(f"\n  Best val acc : {best_val_acc:.4f}")
    print(f"  Test acc     : {test_acc:.4f}")
    print(f"  Saved        : {best_model_path.relative_to(BASE_DIR)}")

    return best_val_acc, test_acc


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train fatigue/eye CNN classifiers")
    parser.add_argument("--clf",    choices=CLASSIFIERS + ["all"], default="all",
                        help="Which classifier to train (default: all)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of epochs (default: {EPOCHS})")
    args = parser.parse_args()

    targets = CLASSIFIERS if args.clf == "all" else [args.clf]

    print(f"\n{'='*60}")
    print(f"  Fatigue Detection — CNN Training")
    print(f"{'='*60}")
    print(f"  Device   : {DEVICE}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {BATCH_SIZE}")
    print(f"  Targets  : {targets}")

    results = {}
    for clf_name in targets:
        print(f"\n{'='*60}")
        print(f"  Training: {clf_name}")
        print(f"{'='*60}")
        val_acc, test_acc = train_classifier(clf_name, epochs=args.epochs)
        if val_acc is not None:
            results[clf_name] = {"val_acc": val_acc, "test_acc": test_acc}

    print(f"\n{'='*60}")
    print("  Training Summary")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f"  {name:<15}  val={r['val_acc']:.4f}   test={r['test_acc']:.4f}")
    print()


if __name__ == "__main__":
    main()