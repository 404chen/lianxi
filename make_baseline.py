import numpy as np
from pathlib import Path

print("正在计算无缺陷数据的平均基准信号...")
# 读取训练集里所有的无缺陷 A 扫
files = list(Path("data/train").rglob("*.npy"))

if len(files) == 0:
    print("未找到训练集数据！")
else:
    all_data = []
    for f in files:
        all_data.append(np.load(f))

    all_data = np.array(all_data)  # 形状 [N, 1024]
    # 对所有无缺陷波形求平均，得到纯净的始发脉冲模板
    baseline = np.mean(all_data, axis=0)  # 形状 [1024]

    # 1. 保存为 npy 格式（供模型快速读取）
    np.save("data/baseline.npy", baseline)

    # 2. 【新增】保存为 txt 格式（供人工查看数值）
    # fmt="%.6f" 表示保留6位小数，每一行将输出一个位置的数值
    np.savetxt("data/baseline.txt", baseline, fmt="%.6f")

    print(f"基准信号已成功保存！")
    print(f" - 模型读取用: data/baseline.npy")
    print(f" - 人工查看用: data/baseline.txt")
    print(f" 形状: {baseline.shape}")