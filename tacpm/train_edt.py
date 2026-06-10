"""
EDT Training Script

Trains the Episode Diagnostic Transformer on the synthetic dataset produced
by generate_edt_data.py.  Saves the trained model to models/edt/edt_model.pth.

Active-learning hook: if data/edt_active.npz exists (new labelled real episodes),
it is merged with the synthetic dataset before training.

Usage:
    python tacpm/train_edt.py                    # train from scratch
    python tacpm/train_edt.py --epochs 200       # more epochs
    python tacpm/train_edt.py --smoke            # 2 epochs, sanity check
"""

import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "tacpm"))

from edt_model import (
    EpisodeDiagnosticTransformer, FAILURE_MODES, FIX_TYPES,
    N_FEATURES, MAX_LEN, make_pad_mask,
)

DEVICE = torch.device("cpu")   # CPU is fast enough for d_model=64


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    def __init__(self, X: np.ndarray, failures: np.ndarray,
                 fixes: np.ndarray, lengths: np.ndarray):
        self.X        = torch.from_numpy(X)
        self.failures = torch.from_numpy(failures)
        self.fixes    = torch.from_numpy(fixes)
        self.lengths  = torch.from_numpy(lengths)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.failures[idx], self.fixes[idx], self.lengths[idx]


def load_dataset(data_dir: Path) -> TrajectoryDataset:
    """Load synthetic data and optionally merge active-learning additions."""
    syn_path    = data_dir / "edt_train.npz"
    active_path = data_dir / "edt_active.npz"

    if not syn_path.exists():
        raise FileNotFoundError(
            f"No training data at {syn_path}.\n"
            "Run: python tacpm/generate_edt_data.py"
        )

    d = np.load(syn_path)
    X, failures, fixes, lengths = d["X"], d["failures"], d["fixes"], d["lengths"]

    if active_path.exists():
        a = np.load(active_path)
        X        = np.concatenate([X,        a["X"]],        axis=0)
        failures = np.concatenate([failures, a["failures"]], axis=0)
        fixes    = np.concatenate([fixes,    a["fixes"]],    axis=0)
        lengths  = np.concatenate([lengths,  a["lengths"]],  axis=0)
        print(f"  Active-learning examples merged: +{len(a['X'])}")

    print(f"  Dataset: {len(X)} trajectories  "
          f"({X.shape[1]} timesteps × {X.shape[2]} features)")
    return TrajectoryDataset(X, failures, fixes, lengths)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    n_epochs:   int   = 150,
    batch_size: int   = 32,
    lr:         float = 3e-4,
    val_frac:   float = 0.15,
    smoke:      bool  = False,
) -> dict:
    data_dir  = project_root / "data"
    model_dir = project_root / "models" / "edt"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "edt_model.pth"

    if smoke:
        n_epochs = 2

    dataset = load_dataset(data_dir)
    n_val   = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model = EpisodeDiagnosticTransformer().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = {"train_loss": [], "val_failure_acc": [], "val_fix_acc": []}

    print(f"\n  Training EDT  ({n_train} train / {n_val} val)  epochs={n_epochs}\n")

    for epoch in range(1, n_epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for X, fail_lbl, fix_lbl, lengths in train_loader:
            X, fail_lbl, fix_lbl = X.to(DEVICE), fail_lbl.to(DEVICE), fix_lbl.to(DEVICE)
            pad_mask = make_pad_mask(lengths.to(DEVICE), X.size(1))

            fail_logits, fix_logits, _ = model(X, pad_mask)
            loss = loss_fn(fail_logits, fail_lbl) + loss_fn(fix_logits, fix_lbl)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()

        sched.step()
        avg_loss = epoch_loss / len(train_loader)

        # --- Validate ---
        model.eval()
        fail_correct = fix_correct = total = 0
        with torch.no_grad():
            for X, fail_lbl, fix_lbl, lengths in val_loader:
                X, fail_lbl, fix_lbl = X.to(DEVICE), fail_lbl.to(DEVICE), fix_lbl.to(DEVICE)
                pad_mask = make_pad_mask(lengths.to(DEVICE), X.size(1))

                fail_logits, fix_logits, _ = model(X, pad_mask)
                fail_correct += (fail_logits.argmax(1) == fail_lbl).sum().item()
                fix_correct  += (fix_logits.argmax(1)  == fix_lbl).sum().item()
                total        += len(fail_lbl)

        fail_acc = fail_correct / max(total, 1)
        fix_acc  = fix_correct  / max(total, 1)
        history["train_loss"].append(avg_loss)
        history["val_failure_acc"].append(fail_acc)
        history["val_fix_acc"].append(fix_acc)

        if (epoch % 10 == 0) or epoch == 1:
            print(f"  Epoch {epoch:>4}  loss={avg_loss:.4f}  "
                  f"fail_acc={fail_acc:.2%}  fix_acc={fix_acc:.2%}")

        # Save best model
        combined_acc = (fail_acc + fix_acc) / 2
        if combined_acc > best_val_acc:
            best_val_acc = combined_acc
            torch.save(model.state_dict(), model_path)

    print(f"\n  Best combined val acc: {best_val_acc:.2%}")
    print(f"  Model saved -> {model_path}")

    # Save training history alongside model
    history_path = model_dir / "edt_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


# ---------------------------------------------------------------------------
# Quick evaluation on full dataset
# ---------------------------------------------------------------------------

def evaluate_saved_model() -> None:
    model_path = project_root / "models" / "edt" / "edt_model.pth"
    if not model_path.exists():
        print("No saved model found — run train_edt.py first.")
        return

    dataset = load_dataset(project_root / "data")
    loader  = DataLoader(dataset, batch_size=64)

    model = EpisodeDiagnosticTransformer().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    from collections import defaultdict
    per_class_fail = defaultdict(lambda: [0, 0])   # correct, total
    per_class_fix  = defaultdict(lambda: [0, 0])

    with torch.no_grad():
        for X, fail_lbl, fix_lbl, lengths in loader:
            X = X.to(DEVICE)
            pad_mask = make_pad_mask(lengths.to(DEVICE), X.size(1))
            fail_logits, fix_logits, _ = model(X, pad_mask)

            for pred, true in zip(fail_logits.argmax(1).tolist(), fail_lbl.tolist()):
                per_class_fail[true][1] += 1
                if pred == true:
                    per_class_fail[true][0] += 1

            for pred, true in zip(fix_logits.argmax(1).tolist(), fix_lbl.tolist()):
                per_class_fix[true][1] += 1
                if pred == true:
                    per_class_fix[true][0] += 1

    print("\n  Per-class failure mode accuracy:")
    for idx, name in enumerate(FAILURE_MODES):
        c, t = per_class_fail[idx]
        print(f"    {name:<22}  {c}/{t}  ({c/max(t,1):.0%})")

    print("\n  Per-class fix accuracy:")
    for idx, name in enumerate(FIX_TYPES):
        c, t = per_class_fix[idx]
        print(f"    {name:<26}  {c}/{t}  ({c/max(t,1):.0%})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=150)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--smoke",      action="store_true",
                        help="2 epochs — sanity check only")
    parser.add_argument("--eval-only",  action="store_true",
                        help="Evaluate saved model, skip training")
    args = parser.parse_args()

    if args.eval_only:
        evaluate_saved_model()
    else:
        train(
            n_epochs   = args.epochs,
            batch_size = args.batch_size,
            lr         = args.lr,
            smoke      = args.smoke,
        )
        print()
        evaluate_saved_model()
