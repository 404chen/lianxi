import os
import numpy as np
from torch.utils.data import Dataset
import torch
from pathlib import Path
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, is_train=True):
        if is_train:
            files = [str(p) for p in Path("data/train").rglob("*.npy")]
        else:
            files = [str(p) for p in Path("data/test").rglob("*.npy")]

        self.data_list = []
        self.label_list = []

        # ===== 加载基准模板 =====
        self.baseline = np.load("data/baseline.npy")

        for file_path in tqdm(files, desc="加载数据集"):
            data = np.load(file_path)
            self.data_list.append(data)

            if is_train:
                # 训练集全是无缺陷数据，标签为 0
                self.label_list.append(0)
            else:
                # ===== 核心修改：通过文件夹名称判断测试集标签 =====
                # 只要完整路径里包含“无缺陷”这三个字，就是标签 0
                if "无缺陷" in file_path:
                    self.label_list.append(0)
                else:
                    self.label_list.append(1)

    def __getitem__(self, index):
        data = self.data_list[index]
        label = self.label_list[index]

        # ===== 扣除基准，放大残差 =====
        data = data - self.baseline

        # 扣除基准后，幅值从2000降到了几十，所以归一化除数改为 50.0
        data = torch.tensor(data).unsqueeze(dim=0) / 20.0
        label = torch.tensor(label)

        return data.float(), label.long()

    def __len__(self):
        return len(self.data_list)