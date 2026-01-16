import os
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np


class AtlasFreeBNTDataset(Dataset):
    def __init__(self,
                 folder_path,
                 indices):
        self.folder_path = folder_path
        self.indices = indices

        label_mat = sio.loadmat(os.path.join(self.folder_path, "label.mat"))
        self.Y = np.squeeze(label_mat["label"]).astype(np.int64) - 1

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        sid = int(self.indices[idx])

        c_path = os.path.join(self.folder_path, f"s_{sid}_cluster_index.mat")
        f_path = os.path.join(self.folder_path, f"s_{sid}_feature.mat")

        c = sio.loadmat(c_path)["cluster_index_mat"]
        f = sio.loadmat(f_path)["feature_mat"]

        return torch.from_numpy(c).long(), \
                torch.from_numpy(f).float(), \
                torch.tensor(self.Y[sid-1], dtype=torch.long)



