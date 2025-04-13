import torch
import torch.nn as nn
from model import SimCLR
from loss import nt_xent_loss
from dataloader import SurfaceDefectDataset
from torch.utils.data import DataLoader
from torchvision import transforms

def train():
    # 设置数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0)),  # 随机裁剪
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),  # 高斯模糊
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 载入数据集
    train_dataset = SurfaceDefectDataset(root_dir='data/train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = SimCLR(base_model='resnet50').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader))

    num_epochs = 100
    temperature = 0.5

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for (view1, view2) in train_loader:
            view1, view2 = view1.to(device), view2.to(device)

            optimizer.zero_grad()
            z1, z2 = model(view1, view2)
            loss = nt_xent_loss(z1, z2, temperature=temperature)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    torch.save(model.encoder.state_dict(), 'pretrained_encoder.pth')


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)  # 确保 Windows 下使用 'spawn' 模式

    train()
