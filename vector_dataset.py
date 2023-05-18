import torch
from torch.utils.data import Dataset


class VectorDataset(Dataset):

    def __init__(self, X_, y_):
        self.datasetDimensionality = X_.shape[1]
        self.X_ = X_
        self.y_ = y_

    def __len__(self):
        return self.X_.shape[0]

    def __getitem__(self, idx):
        sample_X = self.X_[idx]
        sample_y = self.y_[idx]
        return sample_X, sample_y
