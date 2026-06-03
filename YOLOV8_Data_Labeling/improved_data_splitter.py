# improved_data_splitter.py
import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import yaml
import json
import cv2
import numpy as np
from tqdm import tqdm


class ImprovedDataSplitter:
    """改进的数据分割器 - 独立验证集，支持添加白噪声"""

    def __init__(self, dataset_path, val_ratio=0.2, initial_labeled_ratio=0.3, noise_config=None):
        self.dataset_path = Path(dataset_path)
        self.val_ratio = val_ratio
        self.initial_labeled_ratio = initial_labeled_ratio
        self.noise_config = noise_config or {}
        self.setup_directories()

    def setup_directories(self):
        """创建目录结构"""
        dirs = ['data/training', 'data/validation', 'active_learning', 'results']
        for dir_name in dirs:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

    def validate_image_file(self, image_path):
        """验证图像文件是否可读"""
        try:
            from PIL import Image
            with Image.open(str(image_path)) as img:
                img.verify()  # 验证文件完整性
            return True
        except Exception as e:
            print(f"图像文件验证失败 {image_path}: {e}")
            return False

    def discover_valid_data(self):
        """发现有效数据（有标注的图像）"""
        images = []
        labels = []

        # 搜索所有图像文件
        for img_file in self.dataset_path.rglob("*.png"):
            # 验证图像文件是否可读
            if self.validate_image_file(img_file):
                images.append(img_file)

                # 查找对应的标注文件
                label_file = img_file.parent / f"{img_file.stem}.txt"
                if label_file.exists():
                    # 检查标注文件是否为空
                    try:
                        with open(label_file, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                        if content:  # 标注文件不为空
                            labels.append(label_file)
                        else:
                            labels.append(None)
                    except Exception as e:
                        print(f"读取标注文件错误 {label_file}: {e}")
                        labels.append(None)
                else:
                    labels.append(None)
            else:
                print(f"跳过不可读的图像文件: {img_file}")
                labels.append(None)

        # 只保留有有效标注的样本
        valid_data = []
        for img_path, label_path in zip(images, labels):
            if label_path is not None:
                valid_data.append((img_path, label_path))

        print(f"发现 {len(valid_data)} 个有效样本")
        return valid_data

    def add_white_noise(self, image, snr_db=None):
        """向图像添加白噪声

        Args:
            image: 输入图像 (numpy array)
            snr_db: 信噪比 (dB)，如果为None则使用配置中的值

        Returns:
            添加噪声后的图像
        """
        if snr_db is None:
            snr_db = self.noise_config.get('snr_db', 20)  # 默认20dB

        # 将图像转换为浮点型
        img_float = image.astype(np.float32) / 255.0

        # 计算信号功率
        signal_power = np.mean(img_float ** 2)

        # 根据信噪比计算噪声功率
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # 生成高斯白噪声
        noise = np.random.normal(0, np.sqrt(noise_power), img_float.shape)

        # 添加噪声
        noisy_img = img_float + noise

        # 裁剪到有效范围并转换回uint8
        noisy_img = np.clip(noisy_img, 0, 1)
        noisy_img = (noisy_img * 255).astype(np.uint8)

        return noisy_img

    def add_salt_pepper_noise(self, image, noise_ratio=None):
        """添加椒盐噪声

        Args:
            image: 输入图像
            noise_ratio: 噪声比例，如果为None则使用配置中的值

        Returns:
            添加椒盐噪声后的图像
        """
        if noise_ratio is None:
            noise_ratio = self.noise_config.get('salt_pepper_ratio', 0.01)

        noisy_img = image.copy()

        # 盐噪声 (白色像素)
        salt_mask = np.random.random(image.shape[:2]) < noise_ratio / 2
        noisy_img[salt_mask] = 255

        # 椒噪声 (黑色像素)
        pepper_mask = np.random.random(image.shape[:2]) < noise_ratio / 2
        noisy_img[pepper_mask] = 0

        return noisy_img

    def apply_noise_to_image(self, image_path, output_path):
        """对单个图像应用噪声"""
        try:
            # 使用PIL代替OpenCV读取图像，更好的中文路径支持
            from PIL import Image
            import PIL.ImageOps

            # 使用PIL读取图像
            pil_image = Image.open(str(image_path))

            # 转换为RGB模式（如果是RGBA或其他模式）
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # 转换为numpy数组
            image = np.array(pil_image)

            # 应用白噪声
            if self.noise_config.get('enable_white_noise', False):
                image = self.add_white_noise(image)

            # 应用椒盐噪声
            if self.noise_config.get('enable_salt_pepper', False):
                image = self.add_salt_pepper_noise(image)

            # 使用PIL保存图像，避免中文路径问题
            noisy_pil_image = Image.fromarray(image.astype(np.uint8))
            noisy_pil_image.save(str(output_path))

            return True

        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            # 如果处理失败，尝试直接复制
            try:
                shutil.copy2(str(image_path), str(output_path))
                print(f"  回退: 直接复制 {image_path.name}")
                return True
            except Exception as copy_error:
                print(f"  复制也失败: {copy_error}")
                return False

    def create_improved_splits(self):
        """创建改进的数据分割"""
        valid_data = self.discover_valid_data()

        if len(valid_data) < 100:
            raise ValueError(f"有效数据太少: {len(valid_data)}，至少需要100个样本")

        print(f"\n=== 创建改进的数据分割 ===")
        print(f"总有效数据: {len(valid_data)}")

        # 噪声配置信息
        if self.noise_config:
            print(f"噪声配置: {self.noise_config}")

        # 第一步：从整个数据集中划分固定的验证集
        val_size = int(len(valid_data) * self.val_ratio)
        train_val_data, val_data = train_test_split(
            valid_data,
            test_size=val_size,
            random_state=42,
            shuffle=True
        )

        # 第二步：从剩余数据中划分初始标注集和未标注池
        initial_labeled_size = int(len(valid_data) * self.initial_labeled_ratio)
        if initial_labeled_size > len(train_val_data):
            initial_labeled_size = len(train_val_data)

        initial_labeled_data, unlabeled_pool = train_test_split(
            train_val_data,
            train_size=initial_labeled_size,
            random_state=42,
            shuffle=True
        )

        print(f"数据分割结果:")
        print(f"  验证集: {len(val_data)} 样本 (整个数据集的{self.val_ratio * 100:.1f}%)")
        print(f"  初始标注集: {len(initial_labeled_data)} 样本 (整个数据集的{self.initial_labeled_ratio * 100:.1f}%)")
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
                'unlabeled_samples': len(unlabeled_pool),
                'val_ratio': self.val_ratio,
                'initial_labeled_ratio': self.initial_labeled_ratio
            },
            'noise_config': self.noise_config
        }

        with open('results/improved_data_split_info.json', 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)

    def create_training_structure(self, train_data, val_data, output_dir):
        """创建训练目录结构"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建训练集
        train_dir = output_dir / 'train'
        self._create_split_directory(train_data, train_dir, "训练集")

        # 创建验证集（使用固定的验证集）
        val_dir = output_dir / 'val'
        self._create_split_directory(val_data, val_dir, "验证集")

        # 创建数据集配置文件
        dataset_config = self._create_dataset_yaml(output_dir)

        return dataset_config

    def _create_split_directory(self, data, split_dir, split_name):
        """创建分割目录"""
        split_dir = Path(split_dir)
        split_dir.mkdir(exist_ok=True)

        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)

        print(f"正在处理 {split_name}...")

        # 详细的噪声配置检查
        print(f"完整噪声配置: {self.noise_config}")
        noise_enabled = (
                self.noise_config.get('enable_noise', False) or
                self.noise_config.get('enable_white_noise', False) or
                self.noise_config.get('enable_salt_pepper', False)
        )
        print(f"噪声功能是否启用: {noise_enabled}")

        if noise_enabled:
            print("将应用噪声到图像")
        else:
            print("将直接复制图像（不添加噪声）")

        success_count = 0
        noise_applied_count = 0
        direct_copy_count = 0

        for i, (img_path, label_path) in enumerate(tqdm(data, desc=split_name)):
            try:
                # 处理图像（添加噪声或直接复制）
                dest_img = images_dir / img_path.name

                if noise_enabled:
                    # 应用噪声
                    if self.apply_noise_to_image(img_path, dest_img):
                        success_count += 1
                        noise_applied_count += 1
                        # 显示前几个样本的处理信息
                        if i < 3:  # 只显示前3个
                            print(f"  ✅ 已添加噪声: {img_path.name}")
                    else:
                        # 如果处理失败，直接复制原图
                        if not dest_img.exists():
                            shutil.copy2(str(img_path), str(dest_img))
                            if i < 3:
                                print(f"  ⚠️  噪声处理失败，直接复制: {img_path.name}")
                        success_count += 1
                        direct_copy_count += 1
                else:
                    # 直接复制图像
                    if not dest_img.exists():
                        shutil.copy2(str(img_path), str(dest_img))
                        if i < 3:
                            print(f"  📋 直接复制: {img_path.name}")
                    success_count += 1
                    direct_copy_count += 1

                # 复制标注
                dest_label = labels_dir / label_path.name
                if not dest_label.exists():
                    shutil.copy2(str(label_path), str(dest_label))

            except Exception as e:
                print(f"❌ 处理文件错误 {img_path}: {e}")

        print(f"{split_name}处理完成: {success_count}/{len(data)} 个样本")
        if noise_enabled:
            print(f"  噪声应用: {noise_applied_count} 个样本")
            print(f"  直接复制: {direct_copy_count} 个样本")

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

    def generate_noise_samples(self, image_paths, output_dir, num_samples=5):
        """生成噪声样本用于可视化检查"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"生成噪声样本到: {output_dir}")

        for i, img_path in enumerate(image_paths[:num_samples]):
            try:
                # 读取原图
                original = cv2.imread(str(img_path))

                # 应用噪声
                if self.noise_config.get('enable_white_noise', False):
                    noisy = self.add_white_noise(original)
                elif self.noise_config.get('enable_salt_pepper', False):
                    noisy = self.add_salt_pepper_noise(original)
                else:
                    noisy = original

                # 保存对比图
                comparison = np.hstack([original, noisy])
                output_path = output_dir / f"noise_comparison_{i + 1}.png"
                cv2.imwrite(str(output_path), comparison)

            except Exception as e:
                print(f"生成噪声样本失败 {img_path}: {e}")

    def create_fixed_validation_set(self, val_data, output_dir):
        """创建固定的验证集（只在第一次迭代时调用）"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images_dir = output_dir / 'images'
        labels_dir = output_dir / 'labels'
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)

        print("创建固定验证集...")
        success_count = 0

        for img_path, label_path in tqdm(val_data, desc="固定验证集"):
            try:
                # 处理图像（添加噪声或直接复制）
                dest_img = images_dir / img_path.name

                if self.noise_config.get('enable_noise', False) or self.noise_config.get('enable_white_noise', False):
                    # 应用噪声
                    if self.apply_noise_to_image(img_path, dest_img):
                        success_count += 1
                    else:
                        # 如果处理失败，直接复制原图
                        if not dest_img.exists():
                            shutil.copy2(str(img_path), str(dest_img))
                        success_count += 1
                else:
                    # 直接复制图像
                    if not dest_img.exists():
                        shutil.copy2(str(img_path), str(dest_img))
                    success_count += 1

                # 复制标注
                dest_label = labels_dir / label_path.name
                if not dest_label.exists():
                    shutil.copy2(str(label_path), str(dest_label))

            except Exception as e:
                print(f"处理验证集文件错误 {img_path}: {e}")

        print(f"固定验证集创建完成: {success_count}/{len(val_data)} 个样本")
        return output_dir

    def create_training_structure_with_fixed_val(self, train_data, fixed_val_dir, output_dir):
        """创建训练目录结构（使用固定的验证集）"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建训练集
        train_dir = output_dir / 'train'
        self._create_split_directory(train_data, train_dir, "训练集")

        # 使用固定的验证集，创建符号链接（或直接复制结构）
        val_dir = output_dir / 'val'
        if val_dir.exists():
            shutil.rmtree(val_dir)  # 删除旧目录

        # 复制固定验证集结构
        shutil.copytree(fixed_val_dir, val_dir)
        print(f"使用固定验证集: {fixed_val_dir}")

        # 创建数据集配置文件
        dataset_config = self._create_dataset_yaml(output_dir)

        return dataset_config