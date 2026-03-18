import os
import numpy as np
from torch.utils.data import Dataset
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm


class MyDataset(Dataset):

    def __init__(self, is_train=True):
        if is_train:
            csv_files = [str(p) for p in Path("data/train").rglob("*.csv")]
        else:
            csv_files = [str(p) for p in Path("data/test").rglob("*.csv")]
        self.data_list = []
        self.label_list = []
        for csv_path in tqdm(csv_files):
            data = pd.read_csv(csv_path).values
            data = data[:, 2:]
            data = data[:, :1024]
            self.data_list.append(data)
            if "无缺陷数据" in csv_path:
                self.label_list.append(0)
            else:
                self.label_list.append(1)

    def __getitem__(self, index):
        #-2047 1847
        data = self.data_list[index]
        label = self.label_list[index]
        data = torch.tensor(data).permute(1, 0).unsqueeze(dim=0) / 20
        label = torch.tensor(label)

        return data.float(), label.long()

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    dataset = MyDataset(is_train=True)
    for data, label in dataset:
        if label > 0:
            print(label)