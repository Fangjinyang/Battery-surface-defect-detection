import os
import math
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn as nn
from torchvision import transforms, models
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =================配置参数=================
INPUT_DIR = "input_images"  # 原始图片目录
OUTPUT_DIR = "optimized_results"  # 输出目录
NUM_CLUSTERS = 3  # 初始聚类数量
SPLIT_CONFIG = [  # 分割配置
    {"mode": "multi_row", "num": 4, "overlap": 0.2},
    {"mode": "multi_col", "num": 4, "overlap": 0.2},
    {"mode": "grid", "rows": 5, "cols": 5, "overlap": 0.2}
]
ENHANCE_PARAMS = {  # 新增优化参数
    "feature_dim": 512,  # 特征维度（ResNet34:512, ResNet50:2048）
    "pca_variance": 0.95,  # PCA保留方差
    "kmeans_n_init": 10,  # K-means初始化次数
    "min_cluster_size": 5  # 最小簇样本数
}
COLORS = [
    '#FF0000',  # 红色
    '#00FF00',  # 绿色
    '#0000FF',  # 蓝色
    '#FFFF00',  # 黄色
    '#FF00FF',  # 品红
    '#00FFFF'   # 青色
]

# =================增强图像预处理=================
augment_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# =================特征提取优化=================
class EnhancedFeatureExtractor(nn.Module):
    """优化后的多尺度特征提取器"""

    def __init__(self):
        super().__init__()
        base_model = models.resnet34(pretrained=True)
        self.layer1 = nn.Sequential(*list(base_model.children())[:5])  # 浅层特征
        self.layer2 = nn.Sequential(*list(base_model.children())[5:-1])  # 深层特征

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        # 多尺度特征融合
        return torch.cat([
            nn.AdaptiveAvgPool2d((1, 1))(x1).flatten(1),
            nn.AdaptiveAvgPool2d((1, 1))(x2).flatten(1)
        ], dim=1)


def advanced_split(image, config):
    """支持多模式分割的核心函数"""
    width, height = image.size
    sub_images = []
    positions = []

    # 多行水平分割
    if config["mode"] == "multi_row":
        num = config["num"]
        sub_h = height / num
        overlap = sub_h * config["overlap"]

        for i in range(num):
            upper = max(0, i * sub_h - overlap / 2)
            lower = min(height, (i + 1) * sub_h + overlap / 2)
            sub = image.crop((0, upper, width, lower))
            sub_images.append(sub)
            positions.append(("row", i, 0, upper, width, lower))

    # 多列垂直分割
    elif config["mode"] == "multi_col":
        num = config["num"]
        sub_w = width / num
        overlap = sub_w * config["overlap"]

        for i in range(num):
            left = max(0, i * sub_w - overlap / 2)
            right = min(width, (i + 1) * sub_w + overlap / 2)
            sub = image.crop((left, 0, right, height))
            sub_images.append(sub)
            positions.append(("col", i, left, 0, right, height))

    # 网格分割
    elif config["mode"] == "grid":
        rows = config["rows"]
        cols = config["cols"]
        sub_w = width / cols
        sub_h = height / rows
        overlap_w = sub_w * config["overlap"]
        overlap_h = sub_h * config["overlap"]

        for y in range(rows):
            for x in range(cols):
                left = max(0, x * sub_w - overlap_w / 2)
                upper = max(0, y * sub_h - overlap_h / 2)
                right = min(width, (x + 1) * sub_w + overlap_w / 2)
                lower = min(height, (y + 1) * sub_h + overlap_h / 2)
                sub = image.crop((left, upper, right, lower))
                sub_images.append(sub)
                positions.append(("grid", x, y, left, upper, right, lower))

    return sub_images, positions
# =================图像处理流程=================
def process_images():
    """优化的图像处理流程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedFeatureExtractor().to(device)
    model.eval()

    all_features = []
    meta_data = []

    with torch.no_grad():
        for fname in os.listdir(INPUT_DIR):
            img_path = os.path.join(INPUT_DIR, fname)
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')

                    # 多种分割处理
                    for cfg in SPLIT_CONFIG:
                        sub_imgs, positions = advanced_split(img, cfg)

                        for sub_img, pos in zip(sub_imgs, positions):
                            # 增强预处理
                            img_tensor = augment_transform(sub_img).unsqueeze(0).to(device)

                            # 提取融合特征
                            feature = model(img_tensor).cpu().numpy().flatten()

                            all_features.append(feature)
                            meta_data.append({
                                'orig_file': fname,
                                'position': pos,
                                'sub_img': sub_img
                            })
            except Exception as e:
                print(f"处理文件 {fname} 出错: {str(e)}")

    return np.array(all_features), meta_data


# =================聚类优化流程=================
def optimized_clustering(features):
    """带后处理的K-means优化流程"""
    # 特征标准化
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # PCA降维
    pca = PCA(n_components=ENHANCE_PARAMS["pca_variance"])
    pca_features = pca.fit_transform(scaled_features)

    # 多次初始化选择最佳聚类
    best_labels = None
    best_score = float('inf')
    for _ in range(ENHANCE_PARAMS["kmeans_n_init"]):
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, init='k-means++')
        labels = kmeans.fit_predict(pca_features)
        score = kmeans.inertia_  # 使用惯性值作为评估
        if score < best_score:
            best_score = score
            best_labels = labels

    # 后处理：移除小聚类
    unique, counts = np.unique(best_labels, return_counts=True)
    for cluster in unique:
        if counts[cluster] < ENHANCE_PARAMS["min_cluster_size"]:
            best_labels[best_labels == cluster] = -1  # 标记为异常

    return best_labels


# =================可视化与保存=================
def visualize_results(labels, meta_data):
    """增强的可视化输出"""
    # 创建带置信度的目录结构
    for i in range(NUM_CLUSTERS):
        os.makedirs(os.path.join(OUTPUT_DIR, f"cluster_{i + 1}"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "uncertain"), exist_ok=True)

    # 生成热力图
    heatmap_dir = os.path.join(OUTPUT_DIR, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    file_groups = {}
    for idx, data in enumerate(meta_data):
        label = labels[idx]
        if label == -1:
            label = "uncertain"
        else:
            label = f"cluster_{label + 1}"

        # 保存子图
        sub_path = os.path.join(OUTPUT_DIR, label, f"sub_{idx}.jpg")
        data['sub_img'].save(sub_path)

        # 记录热力图数据
        orig_file = data['orig_file']
        if orig_file not in file_groups:
            file_groups[orig_file] = []
        file_groups[orig_file].append((data['position'], label))

    # 绘制增强热力图
    for orig_file, defects in file_groups.items():
        orig_path = os.path.join(INPUT_DIR, orig_file)
        with Image.open(orig_path) as img:
            draw = ImageDraw.Draw(img)

            for pos, label in defects:
                if label == "uncertain":
                    color = "#808080"  # 灰色表示不确定
                else:
                    color = COLORS[int(label.split('_')[-1]) - 1 % len(COLORS)]

                # 绘制带透明度的矩形
                if pos[0] == 'row':
                    _, _, left, upper, right, lower = pos
                elif pos[0] == 'col':
                    _, _, left, upper, right, lower = pos
                else:
                    _, _, _, left, upper, right, lower = pos

                draw.rectangle([left, upper, right, lower], outline=color + "80", width=3)

            img.save(os.path.join(heatmap_dir, f"heatmap_{orig_file}"))


# =================主执行流程=================
if __name__ == "__main__":
    # 特征提取
    features, meta_data = process_images()
    print(f"共提取 {len(features)} 个特征")

    # 优化聚类
    labels = optimized_clustering(features)

    # 结果保存与可视化
    visualize_results(labels, meta_data)
    print(f"优化处理完成！结果保存在 {OUTPUT_DIR} 目录")