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
from dataset import AtlasFreeBNTDataset

## Helper function: filter out data that only has 399 features
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

## Help function: set random seed
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# This function runs evaluation on test dataset.
# It returns average test loss and testing accuracy.
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

        logits = model(c, f)
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



    # Load labels
    label_path = os.path.join(args.data_dir, "label.mat")
    label_mat = sio.loadmat(label_path)

    # Make labels zero-based
    Y = np.squeeze(label_mat["label"]).astype(np.int64) - 1  # shape [N]

    # Get valid subjects
    all_subjects = np.arange(1, len(Y)+1)
    valid_sids = get_valid_subjects(args.data_dir, all_subjects)

    #Stratified split: train (0.70), test(0.15), heldout(0.15)
    train_sids, rem_sids = train_test_split(
        valid_sids,
        test_size=0.30,
        random_state=args.seed,
        shuffle=True,
        stratify=Y[valid_sids - 1],  # stratify by class labels
    )

    rem_labels = Y[rem_sids - 1]
    test_sids, heldout_sids = train_test_split(
        rem_sids,
        test_size=0.50,
        random_state=args.seed,
        shuffle=True,
        stratify=rem_labels,
    )

    # Save Heldout subject IDs
    # with open('headout_sids.json', "w") as f:
    #     json.dump(
    #         [int(s) for s in heldout_sids],
    #         f,
    #         indent=2
    #     )
    # assert 1==2
    print(f"Split sizes: train={len(train_sids)}, test={len(test_sids)}, heldout={len(heldout_sids)}")

    # Build datasets
    train_ds = AtlasFreeBNTDataset(args.data_dir,Y, train_sids)
    test_ds = AtlasFreeBNTDataset(args.data_dir, Y, test_sids)
    heldout_ds = AtlasFreeBNTDataset(args.data_dir,Y, heldout_sids)
    print("datasets created..")

    # Build dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    heldout_loader = DataLoader(
        heldout_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)


    print("dataloaders created..")
    
    # Initialize model
    model = AtlasFreeBNT().to(device)
    print("model initialized..")
    
    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(
                                                optimizer,
                                                step_size=2,
                                                gamma=0.1
                                                )  

    # Initialize meta data
    best_test_acc = -1.0
    lowest_loss = 999999.0
    training_losses = []
    testing_losses = []

    # Training Loop
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

            logits = model(c, f)
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

        # Save the checkpoint with lowest test loss
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

    # Log test And training losses
    with open("loss_log.json", "w") as f:
        json.dump({
            "train": training_losses,
            "test": testing_losses
        }, f)

    #Final evaluation on heldout using best checkpoint
    ckpt = torch.load("best_checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    heldout_loss, heldout_acc = evaluate(model, heldout_loader, device)
    print(f"[HELDOUT] loss {heldout_loss:.4f} acc {heldout_acc:.4f}")


if __name__ == "__main__":
    main()
