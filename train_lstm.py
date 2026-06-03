import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

# ========== 配置 ==========
BASE = "F:/shiyan3"
CLASSES = ['Normal', 'Yawning', 'Microsleep']
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
FRAME_SKIP = 2  # 每2帧取1帧
SEQ_LENGTH = 16  # 每个视频取16帧的序列
IMG_SIZE = 224  # 输入图像尺寸
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.5

print(f"🔥 设备: {DEVICE}")

# ========== 数据增强 ==========
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ========== 视频数据集 ==========
class VideoDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.transform = transform
        self.samples = []

        split_dir = os.path.join(data_dir, split)
        for cls_idx, cls_name in enumerate(CLASSES):
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            for video_name in os.listdir(cls_dir):
                if video_name.endswith('.mp4'):
                    video_path = os.path.join(cls_dir, video_name)
                    self.samples.append((video_path, cls_idx, video_name))

        print(f"{split} 集: {len(self.samples)} 个视频")

    def __len__(self):
        return len(self.samples)

    def extract_frames(self, video_path):
        """从视频中按间隔提取帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % FRAME_SKIP == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            frame_count += 1
        cap.release()
        return frames

    def sample_sequence(self, frames):
        """从帧列表中采样固定长度的序列"""
        if len(frames) == 0:
            # 如果视频为空，返回黑帧
            frames = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)]

        if len(frames) >= SEQ_LENGTH:
            # 均匀采样
            indices = np.linspace(0, len(frames) - 1, SEQ_LENGTH, dtype=int)
            sampled = [frames[i] for i in indices]
        else:
            # 不足则循环填充
            sampled = frames.copy()
            while len(sampled) < SEQ_LENGTH:
                sampled.extend(frames[:SEQ_LENGTH - len(sampled)])
            sampled = sampled[:SEQ_LENGTH]

        # 应用transform
        if self.transform:
            sampled = [self.transform(f) for f in sampled]

        return torch.stack(sampled)  # (SEQ_LENGTH, C, H, W)

    def __getitem__(self, idx):
        video_path, label, video_name = self.samples[idx]
        frames = self.extract_frames(video_path)
        seq = self.sample_sequence(frames)
        return seq, label, video_name


# ========== CNN + LSTM 模型 ==========
class CNNLSTM(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=256, num_layers=2, dropout=0.5):
        super(CNNLSTM, self).__init__()

        # 使用 ResNet18 作为空间特征提取器（去掉最后的全连接层）
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # 输出 (512, 1, 1)
        self.cnn_feature_dim = 512

        # 冻结 CNN 前几层
        for param in list(self.cnn.parameters())[:30]:
            param.requires_grad = False

        # LSTM 时序建模
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # bidirectional → *2
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # Focal Loss 权重（针对类别不平衡）
        self.alpha = torch.tensor([1.0, 1.0, 2.0]).to(DEVICE)  # Microsleep 权重加倍

    def forward(self, x):
        # x: (B, SEQ_LENGTH, C, H, W)
        batch_size, seq_len = x.shape[0], x.shape[1]

        # CNN 逐帧提取特征
        x = x.view(batch_size * seq_len, x.shape[2], x.shape[3], x.shape[4])
        features = self.cnn(x)  # (B*SEQ, 512, 1, 1)
        features = features.view(batch_size, seq_len, -1)  # (B, SEQ, 512)

        # LSTM 时序建模
        lstm_out, (h_n, c_n) = self.lstm(features)

        # 取最后时间步 + 双向拼接
        last_hidden = lstm_out[:, -1, :]  # (B, hidden*2)

        # 分类
        output = self.classifier(last_hidden)
        return output


# ========== Focal Loss ==========
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


# ========== 训练函数 ==========
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for seqs, labels, _ in pbar:
        seqs, labels = seqs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct / total:.4f}'})

    return total_loss / len(loader), correct / total


# ========== 验证/测试函数 ==========
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_names = []

    for seqs, labels, names in tqdm(loader, desc="Evaluating"):
        seqs, labels = seqs.to(device), labels.to(device)

        outputs = model(seqs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_names.extend(names)

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return total_loss / len(loader), acc, all_preds, all_labels, all_names


# ========== 主函数 ==========
def main():
    # 数据加载
    train_dataset = VideoDataset(BASE, "Train", train_transform)
    val_dataset = VideoDataset(BASE, "Val", val_transform)
    test_dataset = VideoDataset(BASE, "Test", val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 模型
    model = CNNLSTM(num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM,
                    num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)

    # 损失函数（Focal Loss 处理类别不平衡）
    alpha = torch.tensor([1.0, 1.0, 2.0]).to(DEVICE)  # Microsleep 权重 ×2
    criterion = FocalLoss(alpha=alpha, gamma=2.0)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 训练
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{BASE}/best_lstm.pth")
            print(f"✅ 最佳模型已保存 (Val Acc: {val_acc:.4f})")

    # 测试集评估
    print(f"\n{'=' * 50}")
    print("测试集评估...")
    model.load_state_dict(torch.load(f"{BASE}/best_lstm.pth"))
    test_loss, test_acc, test_preds, test_labels, test_names = evaluate(
        model, test_loader, criterion, DEVICE
    )

    print(f"\n🎯 测试集准确率: {test_acc:.4f}")

    # 分类报告
    rep = classification_report(test_labels, test_preds, target_names=CLASSES, digits=4, zero_division=0)
    print("\n" + rep)

    # 保存结果
    with open(f"{BASE}/lstm_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"测试集准确率: {test_acc:.4f}\n\n")
        f.write(rep)

    # 混淆矩阵
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("CNN+LSTM Confusion Matrix (Test Set)")
    plt.savefig(f"{BASE}/lstm_confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("✅ 结果已保存")

    # 保存 pred_test.csv
    import pandas as pd
    pred_df = pd.DataFrame({
        'filename': test_names,
        'true_label': test_labels,
        'pred_label': test_preds
    })
    pred_df.to_csv(f"{BASE}/lstm_pred_test.csv", index=False)

    # 错误分析
    print("\n========== 预测错误的视频 ==========")
    for i, (name, tl, pl) in enumerate(zip(test_names, test_labels, test_preds)):
        if tl != pl:
            print(f"  {name}: 真实={CLASSES[tl]}, 预测={CLASSES[pl]}")


if __name__ == '__main__':
    main()