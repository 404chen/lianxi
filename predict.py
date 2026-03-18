import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"#0

import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

from model3 import AnomalyCAE_Final


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(csv_path):
    """
    输入:
        csv_path: str

    返回:
        data: np.ndarray, shape = [H, W]，例如 [1024, 32]
              dtype 建议 float32
    """
    data = pd.read_csv(csv_path).values
    data = data[:, 2:]
    data = data[:, :1024]
    data = data / 20
    data = data.astype(np.float32)
    # 如果你的 load_data 已经返回 [1024, 32]，这里直接扩维
    data = torch.from_numpy(data).permute(1, 0).unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
    return data


def postprocess_tensor(x):
    """
    x: torch.Tensor
    return: np.ndarray
    """
    return x.detach().cpu().numpy()


def predict_one(model, csv_path, device, threshold=None, save_dir="predict_results"):
    os.makedirs(save_dir, exist_ok=True)

    # 1. load
    data = load_data(csv_path)                 # [H, W]
    x = data.to(device)      # [1, 1, H, W]

    # 2. inference
    model.eval()
    with torch.no_grad():
        recon = model(x)
        x = x.permute(0, 1, 3, 2)
        recon = recon.permute(0, 1, 3, 2)
        err_map = torch.abs(recon - x)                     # [1, 1, H, W]
        sample_score = err_map.reshape(err_map.size(0), -1).mean(dim=1)  # [1]

    # 3. to numpy
    input_np = postprocess_tensor(x)[0, 0] * 20         # [H, W]
    recon_np = postprocess_tensor(recon)[0, 0] * 20     # [H, W]
    err_map_np = postprocess_tensor(err_map)[0, 0] # [H, W]
    score = float(sample_score.item())

    # 4. threshold
    pred_label = None
    binary_map = None
    if threshold is not None:
        pred_label = int(score > threshold)
        binary_map = (err_map_np > threshold).astype(np.uint8)

    # 5. save raw arrays
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    np.save(os.path.join(save_dir, f"{base_name}_input.npy"), input_np)
    np.save(os.path.join(save_dir, f"{base_name}_recon.npy"), recon_np)
    np.save(os.path.join(save_dir, f"{base_name}_err_map.npy"), err_map_np)

    if binary_map is not None:
        np.save(os.path.join(save_dir, f"{base_name}_binary_map.npy"), binary_map)
        np.savetxt(os.path.join(save_dir, f"{base_name}_binary_map.txt"), binary_map, fmt='%d')

    # 6. plot: input
    plt.figure(figsize=(12, 6))
    plt.imshow(input_np, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Input")
    plt.xlabel("W")
    plt.ylabel("H")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_input.png"), dpi=200)
    plt.close()

    # 7. plot: reconstruction
    plt.figure(figsize=(12, 6))
    plt.imshow(recon_np, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Reconstruction")
    plt.xlabel("W")
    plt.ylabel("H")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_recon.png"), dpi=200)
    plt.close()

    # 8. plot: error map
    plt.figure(figsize=(12, 6))
    plt.imshow(err_map_np, aspect='auto', cmap='hot')
    plt.colorbar()
    plt.title(f"Error Map | score={score:.6f}")
    plt.xlabel("W")
    plt.ylabel("H")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_err_map.png"), dpi=200)
    plt.close()

    # 9. combined plot
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    im0 = axes[0].imshow(input_np, aspect='auto', cmap='viridis')
    axes[0].set_title("Input")
    axes[0].set_xlabel("W")
    axes[0].set_ylabel("H")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(recon_np, aspect='auto', cmap='viridis')
    axes[1].set_title("Reconstruction")
    axes[1].set_xlabel("W")
    axes[1].set_ylabel("H")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(err_map_np, aspect='auto', cmap='hot')
    if pred_label is None:
        axes[2].set_title(f"Error Map\nscore={score:.6f}")
    else:
        axes[2].set_title(f"Error Map\nscore={score:.6f}, pred={pred_label}")
    axes[2].set_xlabel("W")
    axes[2].set_ylabel("H")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_all.png"), dpi=200)
    plt.close()

    # 10. binary anomaly map
    if binary_map is not None:
        plt.figure(figsize=(12, 6))
        plt.imshow(binary_map, aspect='auto', cmap='gray')
        plt.colorbar()
        plt.title(f"Binary Anomaly Map | threshold={threshold:.6f}")
        plt.xlabel("W")
        plt.ylabel("H")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{base_name}_binary_map.png"), dpi=200)
        plt.close()

    print("csv_path   :", csv_path)
    print("score      :", score)
    if threshold is not None:
        print("threshold  :", threshold)
        print("pred_label :", pred_label)
    print("save_dir   :", save_dir)

    return {
        "input": input_np,
        "recon": recon_np,
        "err_map": err_map_np,
        "score": score,
        "pred_label": pred_label,
        "binary_map": binary_map,
    }


if __name__ == "__main__":
    seed_torch(3047)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. model
    model = AnomalyCAE_Final(
        in_channels=1,
        base_channels=8,
        latent_dim=64,
        dropout=0.1
    )
    model = model.to(device)

    # 2. load checkpoint
    ckpt_path = "checkpoints/AnomalyCAE_Final20_100/epoch_best.pth"
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    # 3. 你的输入 csv
    # csv_path = "data/缺陷数据/2-1-10mm/20260204-001.csv"
    csv_path = "data/无缺陷数据/xin-1/20260126-001.csv"
    csv_name = os.path.basename(csv_path)

    # 4. 阈值
    # 这个 threshold 建议填你训练时保存下来的 metrics.txt 里的 threshold
    threshold = 0.8662542104721069

    # 5. predict
    result = predict_one(
        model=model,
        csv_path=csv_path,
        device=device,
        threshold=threshold,
        save_dir=f"predict_results/{csv_name}"
    )