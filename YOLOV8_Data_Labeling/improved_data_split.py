# improved_data_split.py
import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import yaml
import json


class ImprovedDataSplitter:
    """改进的数据分割器 - 独立验证集"""

    def __init__(self, dataset_path, val_ratio=0.2, initial_labeled_ratio=0.3):
        self.dataset_path = Path(dataset_path)
        self.val_ratio = val_ratio
        self.initial_labeled_ratio = initial_labeled_ratio
        self.setup_directories()

    def setup_directories(self):
        """创建目录结构"""
        dirs = ['data/training', 'data/validation', 'active_learning']
        for dir_name in dirs:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

    def discover_valid_data(self):
        """发现有效数据（有标注的图像）"""
        from data.dataset_loader import DatasetLoader

        dataset_loader = DatasetLoader(self.dataset_path, ".png")
        images, labels = dataset_loader.discover_dataset()

        # 只保留有标注文件且标注文件存在的样本
        valid_data = []
        for img_path, label_path in zip(images, labels):
            if label_path is not None and label_path.exists():
                # 检查标注文件是否为空
                try:
                    with open(label_path, 'r') as f:
                        content = f.read().strip()
                    if content:  # 标注文件不为空
                        valid_data.append((img_path, label_path))
                except:
                    continue

        print(f"发现 {len(valid_data)} 个有效样本")
        return valid_data

    def create_improved_splits(self):
        """创建改进的数据分割"""
        valid_data = self.discover_valid_data()

        if len(valid_data) < 100:
            raise ValueError(f"有效数据太少: {len(valid_data)}，至少需要100个样本")

        print(f"\n=== 创建改进的数据分割 ===")
        print(f"总有效数据: {len(valid_data)}")

        # 第一步：从整个数据集中划分固定的验证集
        train_val_data, val_data = train_test_split(
            valid_data,
            test_size=self.val_ratio,
            random_state=42,
            shuffle=True
        )

        # 第二步：从剩余数据中划分初始标注集和未标注池
        initial_labeled_size = int(len(train_val_data) * self.initial_labeled_ratio)
        initial_labeled_data, unlabeled_pool = train_test_split(
            train_val_data,
            train_size=initial_labeled_size,
            random_state=42,
            shuffle=True
        )

        print(f"数据分割结果:")
        print(f"  验证集: {len(val_data)} 样本 (整个数据集的{self.val_ratio * 100:.1f}%)")
        print(f"  初始标注集: {len(initial_labeled_data)} 样本")
        print(f"  未标注池: {len(unlabeled_pool)} 样本")
        print(f"  总计: {len(val_data) + len(initial_labeled_data) + len(unlabeled_pool)} 样本")

        # 保存分割信息
        self.save_split_info(val_data, initial_labeled_data, unlabeled_pool)

        return initial_labeled_data, val_data, unlabeled_pool

    def save_split_info(self, val_data, initial_labeled_data, unlabeled_pool):
        """保存分割信息"""
        split_info = {
            'validation_set': [str(img_path.name) for img_path, _ in val_data],
            'initial_labeled_set': [str(img_path.name) for img_path, _ in initial_labeled_data],
            'unlabeled_pool': [str(img_path.name) for img_path, _ in unlabeled_pool],
            'statistics': {
                'total_samples': len(val_data) + len(initial_labeled_data) + len(unlabeled_pool),
                'validation_samples': len(val_data),
                'initial_labeled_samples': len(initial_labeled_data),
                'unlabeled_samples': len(unlabeled_pool)
            }
        }

        with open('results/data_split_info.json', 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)

    def create_training_structure(self, train_data, val_data, output_dir):
        """创建训练目录结构"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建训练集
        train_dir = output_dir / 'train'
        self._create_split_directory(train_data, train_dir)

        # 创建验证集（使用固定的验证集）
        val_dir = output_dir / 'val'
        self._create_split_directory(val_data, val_dir)

        # 创建数据集配置文件
        dataset_config = self._create_dataset_yaml(output_dir)

        return dataset_config

    def _create_split_directory(self, data, split_dir):
        """创建分割目录"""
        split_dir = Path(split_dir)
        split_dir.mkdir(exist_ok=True)

        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)

        for img_path, label_path in data:
            try:
                # 复制图像
                dest_img = images_dir / img_path.name
                if not dest_img.exists():
                    shutil.copy2(img_path, dest_img)

                # 复制标注
                dest_label = labels_dir / label_path.name
                if not dest_label.exists():
                    shutil.copy2(label_path, dest_label)
            except Exception as e:
                print(f"复制文件错误 {img_path}: {e}")

    def _create_dataset_yaml(self, output_dir):
        """创建数据集配置文件"""
        config = {
            'path': str(output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 2,
            'names': ['rc', 'vt']
        }

        config_path = output_dir / 'dataset.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        return config_path


def test_improved_split():
    """测试改进的数据分割"""
    splitter = ImprovedDataSplitter(
        dataset_path="D:/py/启用主动学习和注意力机制的yolov8/Data",
        val_ratio=0.2,
        initial_labeled_ratio=0.3
    )

    initial_labeled, val_data, unlabeled_pool = splitter.create_improved_splits()

    # 创建初始训练结构
    dataset_config = splitter.create_training_structure(
        initial_labeled, val_data, "data/improved_training"
    )

    print(f"\n训练结构创建完成:")
    print(f"  数据集配置: {dataset_config}")
    print(f"  训练样本: {len(initial_labeled)}")
    print(f"  验证样本: {len(val_data)}")
    print(f"  未标注样本: {len(unlabeled_pool)}")


if __name__ == "__main__":
    test_improved_split()