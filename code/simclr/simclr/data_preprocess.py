"""
表面缺陷检测数据预处理脚本（直接运行版）
功能：自动从预设目录读取原始图像，处理为统一格式后保存
"""

import os
from PIL import Image
from tqdm import tqdm
import logging

# 配置日志

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('defect_data/preprocess.log'), logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# ************** 用户配置区 **************
INPUT_DIR = "Surface-Defect-Detection/Magnetic-Tile-Defect"  # 原始图像存放路径
OUTPUT_DIR = "data/train"  # 处理后的输出路径
TARGET_SIZE = (224, 224)  # 目标尺寸（宽度, 高度）
OVERWRITE = False  # 是否覆盖已存在的文件


# ***************************************

def preprocess_images():
    """
    执行预处理的核心函数
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # 遍历输入目录（包括子目录）
    for root, _, files in os.walk(INPUT_DIR):
        for filename in tqdm(files, desc=f"Processing {os.path.basename(root)}"):
            # 仅处理图像文件
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                skipped_count += 1
                continue

            input_path = os.path.join(root, filename)
            output_filename = f"defect_{processed_count:04d}.jpg"  # 生成规范文件名
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            # 跳过已处理文件（除非开启覆盖）
            if not OVERWRITE and os.path.exists(output_path):
                skipped_count += 1
                continue

            try:
                # 打开图像并转换为RGB
                with Image.open(input_path) as img:
                    img = img.convert('RGB')

                    # 调整尺寸（保持长宽比，最小边缩放至target_size）
                    img.thumbnail((TARGET_SIZE[0] * 2, TARGET_SIZE[1] * 2))  # 放大避免细节丢失

                    # 中心裁剪至目标尺寸
                    width, height = img.size
                    left = (width - TARGET_SIZE[0]) // 2
                    top = (height - TARGET_SIZE[1]) // 2
                    right = left + TARGET_SIZE[0]
                    bottom = top + TARGET_SIZE[1]
                    img = img.crop((left, top, right, bottom))

                    # 保存为JPG（质量85%）
                    img.save(output_path, format='JPEG', quality=85)
                    processed_count += 1

            except Exception as e:
                logger.error(f"处理失败: {input_path} - {str(e)}")
                error_count += 1

    # 生成统计报告
    logger.info("\n===== 预处理完成 =====")
    logger.info(f"输入目录: {INPUT_DIR}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"目标尺寸: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    logger.info(f"成功处理: {processed_count} 张")
    logger.info(f"跳过文件: {skipped_count} 张（非图像/已存在）")
    logger.info(f"错误文件: {error_count} 张")


if __name__ == "__main__":
    print("""
    **************************************
      表面缺陷检测数据预处理脚本
      开始执行前请确认：
      1. 原始图像已放入 data/raw 目录
      2. 输出目录 data/processed/train 可用
    **************************************
    """)
    preprocess_images()
    print("\n处理完成！请查看 data/processed/train 目录")