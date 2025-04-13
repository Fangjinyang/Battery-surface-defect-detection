import torch.nn.functional as F
import torch

def nt_xent_loss(z1, z2, temperature=0.5):
    # 合并所有样本的特征
    z = torch.cat([z1, z2], dim=0)  # [2*batch_size, output_dim]
    z = F.normalize(z, dim=1)  # L2归一化

    # 计算相似度矩阵
    sim_matrix = torch.mm(z, z.T) / temperature  # [2N, 2N]

    # 排除自身相似度，构造正样本对
    batch_size = z1.size(0)
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z.device)
    mask = torch.eye(labels.size(0), dtype=torch.bool).to(z.device)
    labels = labels[~mask].view(labels.size(0), -1)  # 移除对角线

    # 计算交叉熵损失
    sim_matrix = sim_matrix[~mask].view(sim_matrix.size(0), -1)
    positives = sim_matrix[labels.bool()].view(labels.size(0), -1)
    negatives = sim_matrix[~labels.bool()].view(sim_matrix.size(0), -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long).to(z.device)

    loss = F.cross_entropy(logits, labels)
    return loss