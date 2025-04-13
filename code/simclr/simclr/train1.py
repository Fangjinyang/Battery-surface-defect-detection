import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


# -------------------- 配置区 --------------------
class Config:
    # 数据配置
    data_dir = "data/train"  # 训练图像目录（无需标签）
    input_size = 224  # 输入尺寸
    batch_size = 128  # 根据GPU显存调整（4060建议64-128）

    # 模型配置
    yolo_version = "yolov8s.yaml"  # 选择模型结构
    feat_dim = 512  # 投影头输出维度

    # 训练参数
    epochs = 100
    temperature = 0.5
    learning_rate = 3e-4
    fp16 = True  # 启用混合精度训练

    # 路径配置
    save_dir = "weights/"
    pretrain_name = "yolov8s_backbone_simclr.pth"


# -------------------- 数据加载 --------------------
class SimCLRDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_paths = [os.path.join(data_dir, f)
                            for f in os.listdir(data_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img), self.transform(img)


def get_augmentations(size):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# -------------------- 模型定义 --------------------
class YOLOv8Backbone(nn.Module):
    """提取YOLOv8的骨干网络"""

    def __init__(self, yolo_cfg):
        super().__init__()
        model = YOLO(yolo_cfg).model  # 初始化YOLO模型
        self.layers = model.model[:6]  # 提取前6层作为骨干（根据实际结构调整）

    def forward(self, x):
        return self.layers(x)


class SimCLRWrapper(nn.Module):
    """SimCLR框架封装"""

    def __init__(self, backbone, feat_dim=128):
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self._get_feat_dim(), 512),
            nn.ReLU(),
            nn.Linear(512, feat_dim))

    def _get_feat_dim(self):
        """动态获取骨干网络输出维度"""
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.input_size, Config.input_size)
            return self.backbone(dummy).shape[1]

    def forward(self, x1, x2):
        h1 = self.backbone(x1)
        h2 = self.backbone(x2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        return z1, z2


# -------------------- 损失函数 --------------------
class NTXentLoss(nn.Module):
    """SimCLR对比损失"""

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temp = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        z = nn.functional.normalize(z, dim=1)

        # 计算相似度矩阵
        sim = torch.mm(z, z.T) / self.temp  # [2B, 2B]

        # 创建正样本掩码
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        pos_mask = mask.roll(shifts=batch_size, dims=0)

        # 计算对比损失
        logits = sim[~mask].view(2 * batch_size, -1)  # 移除对角线
        labels = torch.arange(batch_size, device=z.device).repeat(2)
        loss = self.criterion(logits, labels)
        return loss


# -------------------- 训练流程 --------------------
def main():
    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据加载
    transform = get_augmentations(Config.input_size)
    dataset = SimCLRDataset(Config.data_dir, transform)
    loader = DataLoader(dataset, batch_size=Config.batch_size,
                        shuffle=True, num_workers=4, pin_memory=True)

    # 模型初始化
    backbone = YOLOv8Backbone(Config.yolo_version).to(device)
    model = SimCLRWrapper(backbone, Config.feat_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
    criterion = NTXentLoss(Config.temperature)

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=Config.fp16)

    # 训练循环
    best_loss = float('inf')
    os.makedirs(Config.save_dir, exist_ok=True)

    for epoch in range(Config.epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{Config.epochs}")
        for (view1, view2) in pbar:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            optimizer.zero_grad()

            # 混合精度前向
            with torch.cuda.amp.autocast(enabled=Config.fp16):
                z1, z2 = model(view1, view2)
                loss = criterion(z1, z2)

            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})

        # 保存最佳模型
        avg_loss = epoch_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(backbone.state_dict(),
                       os.path.join(Config.save_dir, Config.pretrain_name))
            print(f"Saved best model with loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()