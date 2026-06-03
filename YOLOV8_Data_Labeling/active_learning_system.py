# active_learning_system.py
import os
import shutil
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import defaultdict
import yaml

# active_learning_system.py
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
import yaml
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt


class ActiveLearningSystem:
    """完整的主动学习系统"""

    def __init__(self, config_path='configs/active_learning_system.yaml'):
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
        """准备数据集分割：初始标注集 + 未标注集"""
        from data.dataset_loader import DatasetLoader

        dataset_loader = DatasetLoader(
            self.config['data']['dataset_path'],
            self.config['data'].get('img_format', '.png')
        )

        # 发现所有数据
        images, labels = dataset_loader.discover_dataset()
        valid_data = [(img, lbl) for img, lbl in zip(images, labels) if lbl is not None]

        print(f"总数据量: {len(valid_data)}")

        # 分割初始标注集和未标注集
        initial_ratio = self.config['active_learning'].get('initial_labeled_ratio', 0.3)
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

        return train_data, val_data, unlabeled_pool, dataset_loader

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
            # 复制图像
            dest_img = images_dir / img_path.name
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)

            # 复制标注
            dest_label = labels_dir / label_path.name
            if not dest_label.exists():
                shutil.copy2(label_path, dest_label)

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

    def train_model(self, dataset_config, model_name, use_attention=False):
        """训练模型"""
        print(f"\n=== 训练模型: {model_name} ===")

        # 选择模型
        if use_attention:
            print("使用注意力增强模型")
            model = YOLO('yolov8s.pt')
        else:
            print("使用标准YOLOv8模型")
            model = YOLO('yolov8s.pt')

        # 训练参数
        train_args = {
            'data': dataset_config,
            'epochs': self.config['training'].get('epochs_per_iteration', 30),
            'imgsz': self.config['model']['image_size'],
            'batch': self.config['training']['batch_size'],
            'patience': self.config['training'].get('patience', 10),
            'lr0': self.config['training'].get('learning_rate', 0.01),
            'device': self.config['model']['device'],
            'save': True,
            'verbose': True,
            'project': 'models',
            'name': model_name,
            'exist_ok': True
        }

        # 开始训练
        results = model.train(**train_args)

        # 验证模型
        metrics = model.val()
        print(f"训练完成! mAP50: {metrics.box.map50:.4f}")

        return model, metrics

    def predict_unlabeled_data(self, model, unlabeled_images):
        """预测未标注数据"""
        print(f"预测 {len(unlabeled_images)} 个未标注样本...")

        predictions = []
        uncertainty_scores = []

        for img_path in tqdm(unlabeled_images, desc="预测未标注数据"):
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
                uncertainty_scores.append(1.0)  # 最高不确定性

        return predictions, uncertainty_scores

    def calculate_uncertainty(self, prediction):
        """计算预测不确定性"""
        if prediction.boxes is None or len(prediction.boxes) == 0:
            return 1.0  # 没有检测到目标，高不确定性

        confidences = prediction.boxes.conf.cpu().numpy()

        method = self.config['active_learning'].get('uncertainty_method', 'least_confidence')

        if method == 'least_confidence':
            # 最小置信度
            return 1.0 - np.min(confidences)
        elif method == 'margin':
            # 置信度边际
            if len(confidences) >= 2:
                sorted_conf = np.sort(confidences)[-2:]
                return 1.0 - (sorted_conf[1] - sorted_conf[0])
            else:
                return 1.0
        elif method == 'entropy':
            # 熵
            probs = confidences / np.sum(confidences)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            return entropy / np.log(len(confidences))  # 归一化
        else:
            # 平均不确定性
            return 1.0 - np.mean(confidences)

    def select_most_uncertain_samples(self, unlabeled_pool, uncertainty_scores, n_samples):
        """选择最不确定的样本"""
        # 创建(索引, 不确定性)对
        scored_samples = list(zip(range(len(unlabeled_pool)), uncertainty_scores))

        # 按不确定性降序排序
        scored_samples.sort(key=lambda x: x[1], reverse=True)

        # 选择前n_samples个
        selected_indices = [idx for idx, _ in scored_samples[:n_samples]]

        print(f"选择不确定性最高的 {n_samples} 个样本")
        print(f"不确定性范围: {scored_samples[0][1]:.3f} - {scored_samples[n_samples - 1][1]:.3f}")

        return selected_indices

    def annotate_selected_samples(self, selected_indices, unlabeled_pool, dataset_loader):
        """标注选中的样本（使用真实标签）"""
        newly_labeled = []

        for idx in selected_indices:
            img_path, label_path = unlabeled_pool[idx]

            # 检查标注文件是否存在
            if label_path and label_path.exists():
                newly_labeled.append((img_path, label_path))
            else:
                print(f"警告: 标注文件不存在 {label_path}")

        print(f"成功标注 {len(newly_labeled)} 个样本")
        return newly_labeled

    def run_active_learning_cycle(self):
        """运行完整的主动学习循环"""
        print("=== 主动学习系统启动 ===")

        # 1. 准备数据集分割
        train_data, val_data, unlabeled_pool, dataset_loader = self.prepare_dataset_splits()

        # 主动学习配置
        al_iterations = self.config['active_learning'].get('iterations', 3)
        samples_per_iteration = self.config['active_learning'].get('samples_per_iteration', 50)
        use_attention = self.config['model'].get('use_attention', False)

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
            model_name = f"al_iteration_{iteration}"
            if use_attention:
                model_name = "attention_" + model_name

            model, metrics = self.train_model(dataset_config, model_name, use_attention)

            # 记录性能
            performance = {
                'iteration': iteration,
                'train_samples': len(current_train_data),
                'val_samples': len(current_val_data),
                'unlabeled_samples': len(unlabeled_pool),
                'mAP50': metrics.box.map50,
                'mAP50_95': metrics.box.map,
                'precision': metrics.box.precision,
                'recall': metrics.box.recall
            }
            self.performance_history.append(performance)

            print(f"迭代 {iteration} 性能:")
            print(f"  mAP50: {metrics.box.map50:.4f}")
            print(f"  训练样本: {len(current_train_data)}")
            print(f"  未标注样本: {len(unlabeled_pool)}")

            # 4. 如果没有未标注数据，停止
            if not unlabeled_pool:
                print("没有更多未标注数据，停止主动学习")
                break

            # 5. 预测未标注数据并选择最不确定的样本
            unlabeled_images = [img_path for img_path, _ in unlabeled_pool]
            predictions, uncertainty_scores = self.predict_unlabeled_data(model, unlabeled_images)

            selected_indices = self.select_most_uncertain_samples(
                unlabeled_pool, uncertainty_scores, samples_per_iteration
            )

            # 6. 标注选中的样本（使用真实标签）
            newly_labeled = self.annotate_selected_samples(
                selected_indices, unlabeled_pool, dataset_loader
            )

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
            model.save(f'models/{model_name}_best.pt')

            # 保存选择的不确定性信息
            self.save_uncertainty_info(iteration, selected_indices, uncertainty_scores, unlabeled_pool)

        # 保存最终结果
        self.save_final_results()

        print("\n🎉 主动学习完成！")
        return self.performance_history

    def save_uncertainty_info(self, iteration, selected_indices, uncertainty_scores, unlabeled_pool):
        """保存不确定性信息"""
        uncertainty_dir = Path("results/uncertainty_analysis")
        uncertainty_dir.mkdir(exist_ok=True)

        # 保存选中的样本信息
        selected_info = []
        for idx in selected_indices:
            img_path, label_path = unlabeled_pool[idx]
            selected_info.append({
                'image': str(img_path.name),
                'uncertainty': float(uncertainty_scores[idx]),
                'label_file': str(label_path.name) if label_path else None
            })

        info_path = uncertainty_dir / f"iteration_{iteration}_selected.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(selected_info, f, indent=2, ensure_ascii=False)

    def save_final_results(self):
        """保存最终结果"""
        # 保存性能历史
        with open('results/performance_history.json', 'w', encoding='utf-8') as f:
            json.dump(self.performance_history, f, indent=2, ensure_ascii=False)

        # 绘制学习曲线
        self.plot_learning_curves()

        # 生成最终报告
        self.generate_final_report()

    def plot_learning_curves(self):
        """绘制学习曲线"""
        iterations = [p['iteration'] for p in self.performance_history]
        map50_scores = [p['mAP50'] for p in self.performance_history]
        map_scores = [p['mAP50_95'] for p in self.performance_history]
        train_samples = [p['train_samples'] for p in self.performance_history]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # mAP50曲线
        ax1.plot(iterations, map50_scores, 'o-', linewidth=2, markersize=8, color='blue')
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('mAP50')
        ax1.set_title('主动学习性能提升 (mAP50)')
        ax1.grid(True, alpha=0.3)

        # mAP50-95曲线
        ax2.plot(iterations, map_scores, 's-', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('mAP50-95')
        ax2.set_title('主动学习性能提升 (mAP50-95)')
        ax2.grid(True, alpha=0.3)

        # 训练样本增长
        ax3.plot(iterations, train_samples, '^-', linewidth=2, markersize=8, color='red')
        ax3.set_xlabel('迭代次数')
        ax3.set_ylabel('训练样本数量')
        ax3.set_title('训练数据增长')
        ax3.grid(True, alpha=0.3)

        # 性能vs数据量
        ax4.scatter(train_samples, map50_scores, s=100, alpha=0.7, color='purple')
        for i, (x, y) in enumerate(zip(train_samples, map50_scores)):
            ax4.annotate(f'Iter {i}', (x, y), xytext=(5, 5), textcoords='offset points')
        ax4.set_xlabel('训练样本数量')
        ax4.set_ylabel('mAP50')
        ax4.set_title('性能 vs 数据量')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/active_learning_curves.png', dpi=150, bbox_inches='tight')
        plt.show()

    def generate_final_report(self):
        """生成最终报告"""
        report = {
            'project': '通信信号检测主动学习系统',
            'total_iterations': len(self.performance_history),
            'final_performance': self.performance_history[-1] if self.performance_history else {},
            'performance_summary': {
                'initial_mAP50': self.performance_history[0]['mAP50'] if self.performance_history else 0,
                'final_mAP50': self.performance_history[-1]['mAP50'] if self.performance_history else 0,
                'improvement': self.performance_history[-1]['mAP50'] - self.performance_history[0]['mAP50'] if len(
                    self.performance_history) > 1 else 0,
                'total_samples_added': self.performance_history[-1]['train_samples'] - self.performance_history[0][
                    'train_samples'] if len(self.performance_history) > 1 else 0
            },
            'configuration': self.config
        }

        with open('results/final_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 文本报告
        txt_report = f"""
主动学习系统最终报告
===================

项目: 通信信号检测主动学习系统
总迭代次数: {report['total_iterations']}

性能总结:
--------
初始 mAP50: {report['performance_summary']['initial_mAP50']:.4f}
最终 mAP50: {report['performance_summary']['final_mAP50']:.4f}
性能提升: {report['performance_summary']['improvement']:.4f}
新增样本: {report['performance_summary']['total_samples_added']} 个

详细性能历史:
------------
"""
        for i, perf in enumerate(self.performance_history):
            txt_report += f"迭代 {i}: mAP50={perf['mAP50']:.4f}, 训练样本={perf['train_samples']}\n"

        with open('results/final_report.txt', 'w', encoding='utf-8') as f:
            f.write(txt_report)

        print(txt_report)


def main():
    """主函数"""
    trainer = ActiveLearningSystem()
    performance_history = trainer.run_active_learning_cycle()

    print("\n=== 主动学习完成 ===")
    if performance_history:
        initial_perf = performance_history[0]
        final_perf = performance_history[-1]
        improvement = final_perf['mAP50'] - initial_perf['mAP50']

        print(f"初始性能: mAP50={initial_perf['mAP50']:.4f}, 训练样本={initial_perf['train_samples']}")
        print(f"最终性能: mAP50={final_perf['mAP50']:.4f}, 训练样本={final_perf['train_samples']}")
        print(f"性能提升: {improvement:.4f}")
        print(f"新增样本: {final_perf['train_samples'] - initial_perf['train_samples']}")


if __name__ == "__main__":
    main()