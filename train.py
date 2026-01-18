"""
Simple training script for AtlasFreeBNT.

Assumption (because 0.7/0.3/0.3 doesn't sum to 1):
- Train = 0.70
- Test = 0.15
- Heldout = 0.15
(all splits are stratified by class label)

Usage:
  python train_atlasfreebnt.py --data_dir /path/to/mats --epochs 30 --batch_size 4

Folder expected to contain:
  label_mat.mat  (with key "label")
  s_{sid}_cluster_index.mat (key "cluster_index_mat")
  s_{sid}_feature.mat       (key "feature_mat")
"""

import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy.io as sio

from sklearn.model_selection import train_test_split
from model import AtlasFreeBNT
from collections import Counter
import json
# ---- import your dataset + model ----
# from dataset import AtlasFreeBNTDataset
# from model import AtlasFreeBNT

# If you already have AtlasFreeBNTDataset in the same file, remove this import line.
from dataset import AtlasFreeBNTDataset

def get_valid_subjects(root_dir, subject_ids):
    valid = []
    for sid in subject_ids:
        f_path = os.path.join(root_dir, f"s_{sid}_feature.mat")
        f = sio.loadmat(f_path)["feature_mat"]

        if f.shape[0] == 400:
            valid.append(sid)
        else:
            print(f"[SKIP] subject {sid}: feature shape {f.shape}")
    return np.array(valid, dtype=np.int64)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for c, f, y in loader:
        c = c.to(device)
        f = f.to(device)
        y = y.to(device)

        logits = model(c, f)  # <-- assumes your model forward accepts (c, f)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * y.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += y.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)



    # ---- Load labels only (for stratification) ----
    label_path = os.path.join(args.data_dir, "label.mat")
    label_mat = sio.loadmat(label_path)
    # class labels -> 0-based

    Y = np.squeeze(label_mat["label"]).astype(np.int64) - 1  # shape [N]

    all_subjects = np.arange(1, len(Y)+1)  # subject IDs are 1-based
    valid_sids = get_valid_subjects(args.data_dir, all_subjects)
    #Y_valid = Y[valid_sids - 1] 

    # ---- Stratified split: train (0.70), remaining (0.30) ----
    train_sids, rem_sids = train_test_split(
        valid_sids,
        test_size=0.30,
        random_state=args.seed,
        shuffle=True,
        stratify=Y[valid_sids - 1],  # stratify by class labels
    )

    # ---- Split remaining into test and heldout equally (0.15/0.15 of full) ----
    # Need labels for rem_sids: label index is sid-1
    rem_labels = Y[rem_sids - 1]
    test_sids, heldout_sids = train_test_split(
        rem_sids,
        test_size=0.50,  # half of 0.30 -> 0.15
        random_state=args.seed,
        shuffle=True,
        stratify=rem_labels,
    )

    print(f"Split sizes: train={len(train_sids)}, test={len(test_sids)}, heldout={len(heldout_sids)}")

    # ---- Build datasets/loaders ----
    train_ds = AtlasFreeBNTDataset(args.data_dir,Y, train_sids)
    test_ds = AtlasFreeBNTDataset(args.data_dir, Y, test_sids)
    heldout_ds = AtlasFreeBNTDataset(args.data_dir,Y, heldout_sids)
    print("datasets created..")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    heldout_loader = DataLoader(
        heldout_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    print("dataloaders created..")
    # ---- Create model ----
    # Replace this with your actual import / constructor

    model = AtlasFreeBNT().to(device)
    print("model initialized..")
    # ---- Train setup ----
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(
                                                optimizer,
                                                step_size=2,   # every 2 epochs
                                                gamma=0.1      # multiply LR by 0.5
                                                )  

    best_test_acc = -1.0
    lowest_loss = 999999.0
    training_losses = []
    testing_losses = []
    print("start training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total, correct = 0, 0

        for c, f, y in train_loader:
            c = c.to(device)
            f = f.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(c, f)  # <-- assumes forward(c, f) -> [B, num_classes]
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * y.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += y.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        test_loss, test_acc = evaluate(model, test_loader, device)

        training_losses.append(train_loss)
        testing_losses.append(test_loss)

        if test_loss < lowest_loss:
            best_test_acc = test_acc
            lowest_loss = test_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_acc": best_test_acc,
                    "args": vars(args),
                    "splits": {
                        "train_sids": train_sids.tolist(),
                        "test_sids": test_sids.tolist(),
                        "heldout_sids": heldout_sids.tolist(),
                    },
                },
                "best_checkpoint.pt",
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"test loss {test_loss:.4f} acc {test_acc:.4f} | "
            f"best test acc {best_test_acc:.4f}"
        )
        scheduler.step()

    with open("loss_log.json", "w") as f:
        json.dump({
            "train": training_losses,
            "test": testing_losses
        }, f)

    # ---- Final evaluation on heldout using best checkpoint ----
    ckpt = torch.load("best_checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    heldout_loss, heldout_acc = evaluate(model, heldout_loader, device)
    print(f"[HELDOUT] loss {heldout_loss:.4f} acc {heldout_acc:.4f}")


if __name__ == "__main__":
    main()
