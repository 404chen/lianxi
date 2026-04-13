import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.nn.functional as F
from torch import nn
import copy
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import tqdm
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import MyDataset
from model1d import AnomalyCAE_Final_1D

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def unpack_batch(batch):
    if len(batch) == 2:
        input, label = batch
    elif len(batch) == 3:
        input, _, label = batch
    else:
        raise ValueError("Unexpected batch format.")
    return input, label

def train_model(model, train_dataloader, test_dataloader, optimizer, exp_lr_scheduler,
                num_epochs=25, save_dir="checkpoints"):

    best_model_wts = copy.deepcopy(model.state_dict())
    final_best_acc = 0
    final_best_uar = 0
    results = {"train_loss": [], "test_loss": [], "accuracy": [], "precision": [], "recall": [], "f1": []}

    os.makedirs(save_dir, exist_ok=True)
    log_file_path = os.path.join(save_dir, "log.txt")
    log_file = open(log_file_path, "w")

    for epoch in range(1, 1 + num_epochs):
        # 记录学习率
        for param_group in optimizer.param_groups:
            print("LR", param_group['lr'], file=log_file)

        # --- 训练阶段 ---
        model.train()
        metrics = defaultdict(float)
        epoch_samples = 0

        train_bar = tqdm.tqdm(train_dataloader, file=log_file)
        for batch in train_bar:
            input, label = unpack_batch(batch)
            input, label = input.to(device), label.to(device)

            recon = model(input)
            loss = F.mse_loss(recon, input)

            metrics['loss'] += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_samples += 1
            train_bar.set_description(f"Train [{epoch}|{num_epochs}]: Loss(MSE): {metrics['loss']/epoch_samples:.4f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch}.pth"))

        train_loss = metrics['loss'] / epoch_samples
        exp_lr_scheduler.step()

        # --- 测试阶段 ---
        model.eval()
        metrics = defaultdict(float)
        epoch_samples = 0
        mae_list = []
        label_list = []

        test_bar = tqdm.tqdm(test_dataloader, file=log_file)
        # 1. 纯粹收集测试数据
        for batch in test_bar:
            input, label = unpack_batch(batch)
            input, label = input.to(device), label.to(device)

            with torch.no_grad():
                recon = model(input)
                loss = F.mse_loss(recon, input)

                # 计算逐点平方误差图
                mae = F.mse_loss(recon, input, reduction="none")
                mae = mae.view(mae.size(0), -1).mean(dim=1)

            metrics['loss'] += loss.item()
            mae_list.append(mae)
            label_list.append(label)

            epoch_samples += 1
            test_bar.set_description(f"Test  [{epoch}|{num_epochs}]: Loss(MSE): {metrics['loss']/epoch_samples:.4f}")

        test_loss = metrics['loss'] / max(epoch_samples, 1)

        # 2. 所有批次完成后，统一合并张量
        mae_list_t = torch.cat(mae_list, dim=0)
        label_list_t = torch.cat(label_list, dim=0)

        indexes0 = label_list_t == 0
        indexes1 = label_list_t == 1

        # 3. 安全计算 mae0 和 mae1
        if indexes0.sum() > 0:
            mae0 = mae_list_t[indexes0].mean().item()
        else:
            mae0 = 0.0

        if indexes1.sum() > 0:
            mae1 = mae_list_t[indexes1].mean().item()
        else:
            mae1 = 0.0

        # 4. 搜索更优阈值
        grid_size = 20
        best_uar, best_a0, best_a1 = 0, 0, 1
        for i in range(grid_size + 1):
            a0 = i * (1 / grid_size)
            a1 = 1 - a0
            mae_thresh = mae0 * a0 + mae1 * a1

            predict_list_t = torch.zeros_like(label_list_t)
            predict_list_t[mae_list_t > mae_thresh] = 1

            uar = recall_score(label_list_t.cpu().numpy(), predict_list_t.cpu().numpy(), average="macro", zero_division=0)
            if uar > best_uar:
                best_uar, best_a0, best_a1 = uar, a0, a1

        # 5. 应用最佳阈值计算最终指标
        mae_thresh = mae0 * best_a0 + mae1 * best_a1
        predict_list_t = torch.zeros_like(label_list_t)
        predict_list_t[mae_list_t > mae_thresh] = 1

        uar = recall_score(label_list_t.cpu().numpy(), predict_list_t.cpu().numpy(), average="macro", zero_division=0)
        acc = accuracy_score(label_list_t.cpu().numpy(), predict_list_t.cpu().numpy())
        precision = precision_score(label_list_t.cpu().numpy(), predict_list_t.cpu().numpy(), average="macro", zero_division=0)
        f1 = f1_score(label_list_t.cpu().numpy(), predict_list_t.cpu().numpy(), average="macro", zero_division=0)
        cm = confusion_matrix(label_list_t.cpu().numpy(), predict_list_t.cpu().numpy())

        # 打印到 log 文件
        print("accuracy: ", acc, file=log_file)
        print("precision: ", precision, file=log_file)
        print("recall: ", uar, file=log_file)
        print("f1: ", f1, file=log_file)
        print("cm: \n", cm, file=log_file)
        print("threshold: ", mae_thresh.item() if torch.is_tensor(mae_thresh) else mae_thresh, file=log_file)

        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["accuracy"].append(acc)
        results["precision"].append(precision)
        results["recall"].append(uar)
        results["f1"].append(f1)

        # ================= 恢复所有输出保存 =================
        # save best
        if uar > final_best_uar:
            print("saving best model", file=log_file)
            final_best_acc, final_best_uar = acc, uar
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(save_dir, "epoch_best.pth"))

            with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
                print(f"accuracy: {acc}\nprecision: {precision}\nrecall: {uar}\nf1: {f1}\nthreshold: {mae_thresh}\nconfusion_matrix:\n{cm}", file=f)

            # 保存混淆矩阵图片
            labels_name = [str(i) for i in range(2)]
            plt.figure()
            sns.heatmap((cm).astype("int"), annot=True, fmt='d', cmap='Blues',
                        square=True, xticklabels=labels_name, yticklabels=labels_name)
            plt.savefig(os.path.join(save_dir, "cm.png"), format='png')
            plt.close()

            # 保存误差分布直方图
            plt.figure()
            plt.hist(mae_list_t[indexes0].cpu().numpy(), bins=300, density=True, alpha=0.7, color="r", label="Normal")
            plt.hist(mae_list_t[indexes1].cpu().numpy(), bins=300, density=True, alpha=0.7, color="g", label="Defect")
            plt.xlabel('MSE Error')
            plt.ylabel('Frequency')
            plt.title('Histogram of Reconstruction Error')
            plt.legend()
            plt.savefig(os.path.join(save_dir, "value.png"))
            plt.close()

        # 动态绘制每一轮的指标曲线
        train_loss_arr = np.array(results["train_loss"])
        test_loss_arr = np.array(results["test_loss"])
        uar_arr = np.array(results["recall"])
        x = np.arange(1, epoch + 1)

        plt.figure()
        plt.plot(x, uar_arr, label="uar")
        plt.xlabel("epoch")
        plt.ylabel("uar")
        plt.ylim(0.0, 1.05)
        plt.title('UAR')
        plt.legend()
        plt.savefig(os.path.join(save_dir, "uar.png"))
        plt.close()

        plt.figure()
        plt.plot(x, train_loss_arr, label="train loss")
        plt.plot(x, test_loss_arr, label="test loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title('Loss')
        plt.legend()
        plt.savefig(os.path.join(save_dir, "loss.png"))
        plt.close()

    print("Best acc: ", final_best_acc, file=log_file)
    print("Best uar: ", final_best_uar, file=log_file)

    # 导出完整的 metrics_all.txt
    metrics_all = np.stack(
        [
            np.array(results["train_loss"]),
            np.array(results["test_loss"]),
            np.array(results["accuracy"]),
            np.array(results["precision"]),
            np.array(results["recall"]),
            np.array(results["f1"]),
        ],
        axis=1
    )
    np.savetxt(os.path.join(save_dir, "metrics_all.txt"), metrics_all, fmt="%.4f")

    log_file.close()
    return model

if __name__ == "__main__":
    seed_torch(3047)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AnomalyCAE_Final_1D(in_channels=1, base_channels=8, latent_dim=64, dropout=0.1).to(device)

    train_dataset = MyDataset(is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=False)

    test_dataset = MyDataset(is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False)

    optimizer_ft = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    exp_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=50, eta_min=1e-6)

    model = train_model(
        model, train_dataloader, test_dataloader, optimizer_ft, exp_lr_scheduler,
        num_epochs=100, save_dir=f"checkpoints/AnomalyCAE20_100"
    )