import os
from tqdm import tqdm

import numpy as np
import torch
from torch import tensor
from torch.utils.data import Dataset, DataLoader


def load_offline_data(data_dir, vol=None):
    dataset = []
    print('Loading data...')
    for i, episode_dir in tqdm(enumerate(sorted(os.listdir(data_dir)))):
        if vol is not None and i == vol:
            break
        episode_path = os.path.join(data_dir, episode_dir)
        for npz_file in os.listdir(episode_path):
            npz_path = os.path.join(episode_path, npz_file)
            data = np.load(npz_path)
            d = {}
            for k in data:
                if k == 'actions':
                    d[k] = tensor(data[k], dtype=torch.int64)
                else:
                    d[k] = tensor(data[k], dtype=torch.float)
            dataset.append(d)
    return dataset


class TSCDataset(Dataset):
    def __init__(self, data_dir, vol=None):
        super(TSCDataset, self).__init__()
        self.dataset = load_offline_data(data_dir, vol)
    
    def __getitem__(self, index):
        return self.dataset[index]  # A dict
    
    def __len__(self):
        return len(self.dataset)


def build_data_loader(data_dir: str, vol: int = None, shuffle: bool = True) -> DataLoader:
    dataset = TSCDataset(data_dir, vol)
    data_loader = DataLoader(dataset, 8, shuffle)
    return data_loader
