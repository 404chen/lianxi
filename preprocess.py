import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def process_directory(input_dir, output_dir):
    """
    将包含 B扫原始 csv 的目录，转换为包含 A扫 1D 数组的 npy 目录。
    """
    csv_files = list(Path(input_dir).rglob("*.csv"))
    for csv_path in tqdm(csv_files, desc=f"处理 {input_dir}"):
        # 保持原有目录结构
        rel_path = csv_path.relative_to(input_dir)
        out_folder = Path(output_dir) / rel_path.parent
        out_folder.mkdir(parents=True, exist_ok=True)

        # 读取数据：去掉前两列，截取前 1024 个 pos 点
        data = pd.read_csv(csv_path).values
        data = data[:, 2:1026]  # 形状变为 [32, 1024]

        # 将 32 个阵元（每一行）拆分为单独的 A扫数据
        for i in range(data.shape[0]):
            a_scan = data[i, :]  # 形状为 [1024]
            out_file = out_folder / f"{csv_path.stem}_A{i}.npy"
            # 保存为 numpy 数组文件，加载速度远快于 csv
            np.save(out_file, a_scan.astype(np.float32))


if __name__ == "__main__":
    # 假设你的原始数据存放在 data/raw_train 和 data/raw_test
    # 转换后的数据将保存在 data/train 和 data/test 中，供 MyDataset 读取
    # 请根据你实际的文件夹名称修改路径
    print("开始处理训练集...")
    process_directory("data/raw_train", "data/train")

    print("开始处理测试集...")
    process_directory("data/raw_test", "data/test")