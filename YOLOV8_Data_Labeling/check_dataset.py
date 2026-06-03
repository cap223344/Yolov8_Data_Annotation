# check_dataset.py
import os
import yaml
from pathlib import Path
import cv2
import matplotlib.pyplot as plt


def check_dataset():
    """检查数据集"""

    dataset_path = "D:/py/YOLOV8 Data Labeling Project/Data"

    print("=== 数据集检查 ===")

    # 检查目录结构
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"错误: 数据集路径不存在: {dataset_path}")
        return

    # 查找所有图像文件
    image_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    for fmt in image_formats:
        images.extend(list(dataset_path.rglob(f"*{fmt}")))
        images.extend(list(dataset_path.rglob(f"*{fmt.upper()}")))

    print(f"找到 {len(images)} 个图像文件")

    # 查找标注文件
    labels = []
    for img_path in images:
        label_path = img_path.parent / f"{img_path.stem}.txt"
        if label_path.exists():
            labels.append(label_path)

    print(f"找到 {len(labels)} 个标注文件")

    # 检查标注内容
    if labels:
        label_path = labels[0]
        with open(label_path, 'r') as f:
            content = f.read().strip()
            print(f"示例标注文件内容 ({label_path.name}):")
            print(content)

    # 显示示例图像
    if images:
        sample_image = images[0]
        print(f"\n显示示例图像: {sample_image}")

        img = cv2.imread(str(sample_image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(f"示例图像: {sample_image.name}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('dataset_sample.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("示例图像已保存为: dataset_sample.png")

    # 统计类别分布
    class_counts = {0: 0, 1: 0}  # rc, vt

    for label_path in labels[:100]:  # 检查前100个标注文件
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id in class_counts:
                            class_counts[class_id] += 1
        except Exception as e:
            print(f"读取标注文件错误 {label_path}: {e}")

    print(f"\n类别分布:")
    print(f"  rc (0): {class_counts[0]}")
    print(f"  vt (1): {class_counts[1]}")


if __name__ == "__main__":
    check_dataset()