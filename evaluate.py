import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import scipy.io as sio
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score

from dataset import AtlasFreeBNTDataset
from model import AtlasFreeBNT

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

# This function runs evaluation on the dataset.
# It returns logits and labels for running classification report.
@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    all_logits, all_y = [], []

    for batch in loader:
        if len(batch) != 3:
            raise ValueError(
                f"Dataset must return (c, f, y). Got batch length={len(batch)}."
            )
        c, f, y = batch
        c = c.to(device)
        f = f.to(device)
        y = y.to(device)

        logits = model(c, f)  # [B, C]
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    y_true = torch.cat(all_y, dim=0).numpy()
    return logits, y_true

# This function computes metrics using logits and labels from run_eval function.
def compute_metrics(logits, y_true):
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    y_pred = probs.argmax(axis=1)
    n_classes = probs.shape[1]

    acc = accuracy_score(y_true, y_pred)

    if n_classes == 2:
        prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
        auroc = roc_auc_score(y_true, probs[:, 1])
    else:
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        auroc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")

    return acc, auroc, f1, prec, recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--mode", choices=["all", "heldout"], default="all")
    ap.add_argument("--heldout_json", type=str, default="heldout_sids.json", help="Required if mode=heldout")
    ap.add_argument("--ckpt", type=str, default="best_checkpoint.pt")
    ap.add_argument("--batch_size", type=int, default=4)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load labels
    label_path = os.path.join(args.data_dir, "label.mat")
    label_mat = sio.loadmat(label_path)
    # Make labels zero-based
    Y = np.squeeze(label_mat["label"]).astype(np.int64) - 1

    # Filter out invalid subjects
    all_subjects = np.arange(1, len(Y)+1)
    valid_sids = get_valid_subjects(args.data_dir, all_subjects)

    # Get subject ids based on args mode
    if args.mode == "heldout":
        if args.heldout_json is None:
            raise ValueError("--heldout_json is required when --mode heldout")
        with open(args.heldout_json, "r") as f:
            sids = np.array(json.load(f), dtype=np.int64)
    else:
        sids = valid_sids

    if len(sids) == 0:
        raise RuntimeError("No subjects found for evaluation. Check data_dir / filtering / heldout_json.")

    # Build dataset and dataloader
    dataset = AtlasFreeBNTDataset(args.data_dir, Y, sids)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # Load checkpoint & model
    ckpt = torch.load(args.ckpt, map_location=device)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(
            "Checkpoint must be a dict with keys: model_state_dict (and ideally model_args)."
        )

    model = AtlasFreeBNT().to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # Run evaluation
    logits, y_true = run_eval(model, loader, device)
    acc, auroc, f1, prec, recall = compute_metrics(logits, y_true)

    print(f"Num of Subjects : {len(sids)}")
    print(f"Accuracy : {acc:.4f}")
    print(f"AUROC    : {auroc:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {recall:.4f}\n")


if __name__ == "__main__":
    main()

