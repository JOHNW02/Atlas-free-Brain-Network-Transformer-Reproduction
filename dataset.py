import os
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np


class AtlasFreeBNTDataset(Dataset):
    def __init__(self,
                 folder_path,
                 Y,
                 indices):
        self.folder_path = folder_path
        self.indices = indices

        self.Y = Y

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        sid = int(self.indices[idx])

        c_path = os.path.join(self.folder_path, f"s_{sid}_cluster_index.mat")
        f_path = os.path.join(self.folder_path, f"s_{sid}_feature.mat")

        c = sio.loadmat(c_path)["cluster_index_mat"]
        f = sio.loadmat(f_path)["feature_mat"]

        f = torch.from_numpy(f).float()

        bg = torch.zeros(1, f.size(1))
        f = torch.cat([bg, f], dim=0)
        #print(torch.tensor(self.Y[sid-1], dtype=torch.long))
        if f.shape != (401, 1632):
            raise RuntimeError(f"[BAD F] sid={sid}, F.shape={f.shape}, F.dtype={f.dtype}")
        return torch.from_numpy(c).long(), \
                f, \
                torch.tensor(self.Y[sid-1], dtype=torch.long)



