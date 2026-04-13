import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import gaussian_filter  # <=== 新增：引入高斯滤波做平滑处理
from model1d import AnomalyCAE_Final_1D

# ================= 全局加载基准模板 =================
BASELINE = np.load("data/baseline.npy").astype(np.float32)


def load_data_for_bscan(csv_path):
    data = pd.read_csv(csv_path).values[:, 2:1026]

    # 核心：利用广播机制减去 1D 基准模板，抹除始发脉冲
    data = data - BASELINE

    # 除以 20.0 归一化残差
    data = data / 20.0
    data = data.astype(np.float32)
    data_t = torch.from_numpy(data).unsqueeze(1)
    return data_t


def predict_one(model, csv_path, device, threshold=None, save_dir="predict_results"):
    os.makedirs(save_dir, exist_ok=True)
    x = load_data_for_bscan(csv_path).to(device)

    model.eval()
    with torch.no_grad():
        recon = model(x)
        x_2d = x.squeeze(1).permute(1, 0)
        recon_2d = recon.squeeze(1).permute(1, 0)

        # 直接使用平方误差（MSE）计算整张图的异常
        err_map = torch.square(recon_2d - x_2d)
        sample_score = err_map.mean()

    # 乘以 20 还原残差幅度（不再加回基准，直接观察残差图）
    input_np = x_2d.detach().cpu().numpy() * 20.0
    recon_np = recon_2d.detach().cpu().numpy() * 20.0
    err_map_np = err_map.detach().cpu().numpy()
    score = float(sample_score.item())

    pred_label = None
    binary_map = None
    if threshold is not None:
        pred_label = int(score > threshold)
        binary_map = (err_map_np > threshold).astype(np.uint8)

    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    # ================= 保存所有 npy 和 txt 数据 =================
    np.save(os.path.join(save_dir, f"{base_name}_input.npy"), input_np)
    np.save(os.path.join(save_dir, f"{base_name}_recon.npy"), recon_np)
    np.save(os.path.join(save_dir, f"{base_name}_err_map.npy"), err_map_np)
    np.savetxt(os.path.join(save_dir, f"{base_name}_err_map.txt"), err_map_np, fmt='%.4f')

    if binary_map is not None:
        np.save(os.path.join(save_dir, f"{base_name}_binary_map.npy"), binary_map)
        np.savetxt(os.path.join(save_dir, f"{base_name}_binary_map.txt"), binary_map, fmt='%d')

    # ================= 单独的作图 =================
    plt.figure(figsize=(12, 6))
    plt.imshow(input_np, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Input Residual (B-Scan)")
    plt.xlabel("Elements (32)")
    plt.ylabel("Pos (1024)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_input.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.imshow(recon_np, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Reconstruction Residual")
    plt.xlabel("Elements (32)")
    plt.ylabel("Pos (1024)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_recon.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.imshow(err_map_np, aspect='auto', cmap='hot')
    plt.colorbar()
    plt.title(f"Error Map | score={score:.6f}")
    plt.xlabel("Elements (32)")
    plt.ylabel("Pos (1024)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_err_map.png"), dpi=200)
    plt.close()

    if binary_map is not None:
        plt.figure(figsize=(12, 6))
        plt.imshow(binary_map, aspect='auto', cmap='gray')
        plt.colorbar()
        plt.title(f"Binary Anomaly Map | threshold={threshold:.6f}")
        plt.xlabel("Elements (32)")
        plt.ylabel("Pos (1024)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{base_name}_binary_map.png"), dpi=200)
        plt.close()

    # ================= 【新增】带有物理坐标的平滑实际缺陷图 =================
    # 1. 计算物理坐标范围
    x_max_mm = 31 * 0.6  # 32条数据（0到31的间隔），间隔0.6mm
    y_max_mm = 1024 * 1e-8 * (6100 / 2.0) * 1e3  # 根据公式 y * 10^-8 * 6400/2 * 10^3 算出的最大深度

    # 2. 对误差图进行二维高斯平滑处理 (sigma 控制平滑强度，推荐 1.5 到 3 之间)
    smoothed_err_map = gaussian_filter(err_map_np, sigma=2.0)

    # 3. 绘制带有真实物理尺度的图像
    plt.figure(figsize=(6, 8))  # 调整了图像比例，使其看起来更符合实际的深宽比
    # extent=[left, right, bottom, top] 控制坐标轴的真实数值
    # interpolation='bicubic' 让 Matplotlib 在渲染时进一步进行双三次平滑插值
    plt.imshow(smoothed_err_map, aspect='auto', cmap='jet',
               extent=[0, x_max_mm, y_max_mm, 0], interpolation='bicubic')

    plt.colorbar(label='Defect Intensity (Smoothed MSE)')
    #plt.title(f"Physical Defect Map\n(Width: {x_max_mm:.1f}mm, Depth: {y_max_mm:.1f}mm)")
    plt.xlabel("Width (mm)")
    plt.ylabel("Depth (mm)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_physical_map.png"), dpi=300)  # 输出高分辨率 300dpi
    plt.close()

    # ================= 组合图展示 =================
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    im0 = axes[0].imshow(input_np, aspect='auto', cmap='viridis')
    axes[0].set_title("Input Residual (B-Scan)")
    axes[0].set_xlabel("Elements (32)")
    axes[0].set_ylabel("Pos (1024)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(recon_np, aspect='auto', cmap='viridis')
    axes[1].set_title("Reconstruction Residual")
    axes[1].set_xlabel("Elements (32)")
    axes[1].set_ylabel("Pos (1024)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(err_map_np, aspect='auto', cmap='hot')
    title_str = f"Error Map\nscore={score:.6f}" if pred_label is None else f"Error Map\nscore={score:.6f}, pred={pred_label}"
    axes[2].set_title(title_str)
    axes[2].set_xlabel("Elements (32)")
    axes[2].set_ylabel("Pos (1024)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_all.png"), dpi=200)
    plt.close()

    print(f"csv_path   : {csv_path}\nscore      : {score}\n")
    return {"score": score}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AnomalyCAE_Final_1D(in_channels=1, base_channels=8, latent_dim=64, dropout=0.1).to(device)

    # 填入你新的模型权重路径
    ckpt_path = "checkpoints/AnomalyCAE20_100/epoch_best.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # ================= 批量处理逻辑 =================
    target_folder = "data/缺陷数据/5mm孔-10mm深"
    csv_files = glob.glob(os.path.join(target_folder, "*.csv"))

    if len(csv_files) == 0:
        print(f"在 {target_folder} 中没有找到 CSV 文件，请检查路径。")
    else:
        print(f"找到 {len(csv_files)} 个文件，开始批量预测...")
        for csv_path in csv_files:
            csv_name = os.path.basename(csv_path)
            base_name = os.path.splitext(csv_name)[0]

            print(f"正在处理: {csv_name} ...")
            predict_one(
                model=model,
                csv_path=csv_path,
                device=device,
                threshold=0.19098518788814545,
                save_dir=f"predict_results/20-100/5mm孔-10mm深/{base_name}"
            )

        print("所有文件预测完毕！")