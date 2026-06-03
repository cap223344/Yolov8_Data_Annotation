# active_learning_system_fixed.py
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'  # 限制OpenMP线程数

import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
import yaml
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
import json
import gc  # 垃圾回收


class MemoryOptimizedActiveLearning:
    """内存优化的主动学习系统"""

    def __init__(self, config_path='configs/memory_fix_config.yaml'):
        self.config = self.load_config(config_path)
        self.setup_directories()
        self.performance_history = []

    def load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def setup_directories(self):
        """创建必要的目录"""
        dirs = ['models', 'results', 'logs', 'active_learning', 'outputs']
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)

    def prepare_dataset_splits(self):
        """准备数据集分割，过滤无效数据"""
        from data.dataset_loader import DatasetLoader

        dataset_loader = DatasetLoader(
            self.config['data']['dataset_path'],
            self.config['data'].get('img_format', '.png')
        )

        # 发现所有数据，只使用有标注的
        images, labels = dataset_loader.discover_dataset()
        valid_data = [(img, lbl) for img, lbl in zip(images, labels) if lbl is not None and lbl.exists()]

        print(f"有效数据量: {len(valid_data)}")

        if len(valid_data) < 20:
            raise ValueError(f"有效数据太少: {len(valid_data)}，至少需要100个样本")

        # 分割初始标注集和未标注集
        initial_ratio = self.config['active_learning'].get('initial_labeled_ratio', 0.3)
        min_initial = max(20, int(len(valid_data) * 0.05))  # 至少100个初始样本

        if len(valid_data) * initial_ratio < min_initial:
            initial_ratio = min_initial / len(valid_data)

        labeled_data, unlabeled_pool = train_test_split(
            valid_data,
            train_size=initial_ratio,
            random_state=42,
            shuffle=True
        )

        # 从标注集中分割训练集和验证集
        train_data, val_data = train_test_split(
            labeled_data,
            test_size=0.2,
            random_state=42
        )

        print(f"数据集分割完成:")
        print(f"  初始训练集: {len(train_data)} 样本")
        print(f"  验证集: {len(val_data)} 样本")
        print(f"  未标注池: {len(unlabeled_pool)} 样本")

        return train_data, val_data, unlabeled_pool

    def create_training_dataset(self, train_data, val_data, output_dir):
        """创建训练数据集"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建训练集
        self._create_data_split(train_data, output_dir / 'train')
        # 创建验证集
        self._create_data_split(val_data, output_dir / 'val')

        # 创建数据集配置文件
        dataset_config = self._create_dataset_yaml(output_dir)

        return dataset_config

    def _create_data_split(self, data, split_dir):
        """创建数据分割"""
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

    def train_model_memory_safe(self, dataset_config, model_name):
        """内存安全的模型训练"""
        print(f"\n=== 训练模型: {model_name} ===")

        try:
            # 使用更小的nano模型
            model = YOLO('yolov8n.pt')

            # 内存优化的训练参数
            train_args = {
                'data': dataset_config,
                'epochs': self.config['training'].get('epochs_per_iteration', 20),
                'imgsz': self.config['model']['image_size'],
                'batch': self.config['training']['batch_size'],
                'workers': 2,  # 减少workers
                'patience': self.config['training'].get('patience', 8),
                'device': self.config['model']['device'],
                'save': True,
                'verbose': True,
                'project': 'models',
                'name': model_name,
                'exist_ok': True,
                'cache': False,  # 关闭缓存
                'amp': True,  # 自动混合精度
                'close_mosaic': 5,  # 提前关闭mosaic
            }

            # 开始训练
            results = model.train(**train_args)

            # 强制垃圾回收
            gc.collect()

            # 验证模型
            metrics = model.val()
            print(f"训练完成! mAP50: {metrics.box.map50:.4f}")

            return model, metrics

        except Exception as e:
            print(f"训练错误: {e}")
            # 强制清理内存
            gc.collect()
            return None, None

    def predict_batch(self, model, unlabeled_images, batch_size=50):
        """分批预测以避免内存溢出"""
        print(f"分批预测 {len(unlabeled_images)} 个未标注样本...")

        predictions = []
        uncertainty_scores = []

        for i in range(0, len(unlabeled_images), batch_size):
            batch_images = unlabeled_images[i:i + batch_size]
            print(f"预测批次 {i // batch_size + 1}/{(len(unlabeled_images) - 1) // batch_size + 1}")

            for img_path in batch_images:
                try:
                    results = model.predict(img_path, conf=0.1, verbose=False)
                    prediction = results[0]

                    # 计算不确定性分数
                    uncertainty = self.calculate_uncertainty(prediction)

                    predictions.append(prediction)
                    uncertainty_scores.append(uncertainty)

                except Exception as e:
                    print(f"预测错误 {img_path}: {e}")
                    predictions.append(None)
                    uncertainty_scores.append(1.0)

            # 每个批次后清理内存
            gc.collect()

        return predictions, uncertainty_scores

    def calculate_uncertainty(self, prediction):
        """计算预测不确定性"""
        if prediction is None or prediction.boxes is None or len(prediction.boxes) == 0:
            return 1.0  # 没有检测到目标，高不确定性

        try:
            confidences = prediction.boxes.conf.cpu().numpy()

            method = self.config['active_learning'].get('uncertainty_method', 'least_confidence')

            if method == 'least_confidence':
                return 1.0 - np.min(confidences)
            elif method == 'margin':
                if len(confidences) >= 2:
                    sorted_conf = np.sort(confidences)[-2:]
                    return 1.0 - (sorted_conf[1] - sorted_conf[0])
                else:
                    return 1.0
            elif method == 'entropy':
                probs = confidences / np.sum(confidences)
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                return entropy / np.log(len(confidences))
            else:
                return 1.0 - np.mean(confidences)
        except:
            return 1.0

    def run_optimized_active_learning(self):
        """运行优化的主动学习循环"""
        print("=== 内存优化的主动学习系统启动 ===")

        try:
            # 1. 准备数据集分割
            train_data, val_data, unlabeled_pool = self.prepare_dataset_splits()

            # 主动学习配置
            al_iterations = self.config['active_learning'].get('iterations', 3)
            samples_per_iteration = self.config['active_learning'].get('samples_per_iteration', 30)

            # 初始训练
            current_train_data = train_data.copy()
            current_val_data = val_data.copy()

            for iteration in range(al_iterations):
                print(f"\n{'=' * 60}")
                print(f"主动学习迭代 {iteration + 1}/{al_iterations}")
                print(f"{'=' * 60}")

                # 2. 创建当前迭代的训练数据集
                iteration_dir = f"active_learning/iteration_{iteration}"
                dataset_config = self.create_training_dataset(
                    current_train_data, current_val_data, iteration_dir
                )

                # 3. 训练模型
                model_name = f"al_optimized_iteration_{iteration}"
                model, metrics = self.train_model_memory_safe(dataset_config, model_name)

                if model is None:
                    print(f"迭代 {iteration} 训练失败，跳过")
                    continue

                # 记录性能
                performance = {
                    'iteration': iteration,
                    'train_samples': len(current_train_data),
                    'val_samples': len(current_val_data),
                    'unlabeled_samples': len(unlabeled_pool),
                    'mAP50': metrics.box.map50 if metrics else 0,
                    'mAP50_95': metrics.box.map if metrics else 0,
                }
                self.performance_history.append(performance)

                print(f"迭代 {iteration} 性能:")
                print(f"  mAP50: {performance['mAP50']:.4f}")
                print(f"  训练样本: {performance['train_samples']}")
                print(f"  未标注样本: {performance['unlabeled_samples']}")

                # 4. 如果没有未标注数据，停止
                if not unlabeled_pool:
                    print("没有更多未标注数据，停止主动学习")
                    break

                # 5. 预测未标注数据并选择最不确定的样本
                unlabeled_images = [img_path for img_path, _ in unlabeled_pool]
                predictions, uncertainty_scores = self.predict_batch(model, unlabeled_images, batch_size=20)

                # 选择最不确定的样本
                scored_samples = list(zip(range(len(unlabeled_pool)), uncertainty_scores))
                scored_samples.sort(key=lambda x: x[1], reverse=True)
                selected_indices = [idx for idx, _ in scored_samples[:samples_per_iteration]]

                print(f"选择不确定性最高的 {len(selected_indices)} 个样本")

                # 6. 标注选中的样本（使用真实标签）
                newly_labeled = []
                for idx in selected_indices:
                    img_path, label_path = unlabeled_pool[idx]
                    if label_path and label_path.exists():
                        newly_labeled.append((img_path, label_path))

                print(f"成功标注 {len(newly_labeled)} 个样本")

                # 7. 更新数据集
                # 从未标注池中移除已标注的样本
                unlabeled_pool = [sample for i, sample in enumerate(unlabeled_pool)
                                  if i not in selected_indices]

                # 将新标注的样本加入训练集
                current_train_data.extend(newly_labeled)

                print(f"数据集更新:")
                print(f"  + 新标注样本: {len(newly_labeled)}")
                print(f"  = 当前训练集: {len(current_train_data)}")
                print(f"  - 剩余未标注: {len(unlabeled_pool)}")

                # 保存当前模型
                try:
                    model.save(f'models/{model_name}_best.pt')
                except:
                    print("模型保存失败，继续下一轮")

                # 清理内存
                del model
                gc.collect()

            # 保存最终结果
            self.save_final_results()

            print("\n🎉 主动学习完成！")
            return self.performance_history

        except Exception as e:
            print(f"主动学习系统错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_final_results(self):
        """保存最终结果"""
        try:
            # 保存性能历史
            with open('results/performance_history_optimized.json', 'w', encoding='utf-8') as f:
                json.dump(self.performance_history, f, indent=2, ensure_ascii=False)

            # 生成简单报告
            self.generate_simple_report()

        except Exception as e:
            print(f"保存结果错误: {e}")

    def generate_simple_report(self):
        """生成简单报告"""
        if not self.performance_history:
            return

        report = "主动学习优化系统报告\n"
        report += "===================\n\n"

        for perf in self.performance_history:
            report += f"迭代 {perf['iteration']}: "
            report += f"mAP50={perf['mAP50']:.4f}, "
            report += f"训练样本={perf['train_samples']}, "
            report += f"未标注样本={perf['unlabeled_samples']}\n"

        if len(self.performance_history) > 1:
            initial = self.performance_history[0]
            final = self.performance_history[-1]
            improvement = final['mAP50'] - initial['mAP50']
            samples_added = final['train_samples'] - initial['train_samples']

            report += f"\n总结:\n"
            report += f"初始性能: mAP50={initial['mAP50']:.4f}\n"
            report += f"最终性能: mAP50={final['mAP50']:.4f}\n"
            report += f"性能提升: {improvement:.4f}\n"
            report += f"新增样本: {samples_added}\n"

        print(report)

        with open('results/simple_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)


def main():
    """主函数"""
    print("启动内存优化的主动学习系统...")

    # 强制清理内存
    import gc
    gc.collect()

    trainer = MemoryOptimizedActiveLearning()
    performance_history = trainer.run_optimized_active_learning()

    if performance_history:
        print("\n=== 主动学习完成 ===")
        if len(performance_history) > 1:
            initial = performance_history[0]
            final = performance_history[-1]
            improvement = final['mAP50'] - initial['mAP50']

            print(f"初始性能: mAP50={initial['mAP50']:.4f}, 训练样本={initial['train_samples']}")
            print(f"最终性能: mAP50={final['mAP50']:.4f}, 训练样本={final['train_samples']}")
            print(f"性能提升: {improvement:.4f}")
    else:
        print("\n主动学习失败")


if __name__ == "__main__":
    main()