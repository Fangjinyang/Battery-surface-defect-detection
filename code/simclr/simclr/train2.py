from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
from model import YOLOv8Backbone
from torchvision import transforms
from torch.nn import functional as F
class Augmentation:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    loss = F.cross_entropy(logits, labels)
    return loss

# 假设无标签数据存储在 path/to/unlabeled_data
dataset = ImageFolder("data", transform=Augmentation())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLOv8Backbone().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):  # 训练 10 个 epoch
    for (x1, x2), _ in dataloader:
        x1, x2 = x1.to(device), x2.to(device)
        optimizer.zero_grad()
        z1 = model(x1)
        z2 = model(x2)
        loss = contrastive_loss(z1, z2)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
torch.save(model.state_dict(), "simclr_pretrained_backbone.pth")