# improved_active_learning.py
import os
import shutil

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
import yaml
from pathlib import Path
import numpy as np
import json
import gc
from improved_data_splitter import ImprovedDataSplitter


class ImprovedActiveLearning:
    """改进的主动学习系统 - 独立验证集"""

    def __init__(self, config_path='configs/improved_al_config.yaml'):
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

    def prepare_improved_dataset(self):
        """准备改进的数据集分割"""
        splitter = ImprovedDataSplitter(
            dataset_path=self.config['data']['dataset_path'],
            val_ratio=self.config['data'].get('val_ratio', 0.2),
            initial_labeled_ratio=self.config['active_learning'].get('initial_labeled_ratio', 0.3),
            noise_config=self.config.get('noise_augmentation', {})  # 传递噪声配置
        )

        initial_labeled_data, val_data, unlabeled_pool = splitter.create_improved_splits()

        # 生成噪声样本用于检查
        if self.config.get('noise_augmentation', {}).get('enable_noise', False):
            sample_images = [img_path for img_path, _ in initial_labeled_data[:3]]
            splitter.generate_noise_samples(sample_images, 'results/noise_samples')

        return initial_labeled_data, val_data, unlabeled_pool, splitter


    def train_model_with_fixed_val(self, train_data, val_data, model_name, iteration=0):
        """使用固定验证集训练模型"""
        print(f"\n=== 训练模型: {model_name} ===")

        try:
            # 使用nano模型节省内存
            model = YOLO('yolov8n.pt')

            # 创建训练结构
            iteration_dir = f"active_learning/improved_iteration_{iteration}"
            dataset_config = self.create_training_structure(train_data, val_data, iteration_dir, iteration)

            # 训练参数
            train_args = {
                'data': dataset_config,
                'epochs': self.config['training'].get('epochs_per_iteration', 20),
                'imgsz': self.config['model']['image_size'],
                'batch': self.config['training']['batch_size'],
                'workers': 2,
                'patience': self.config['training'].get('patience', 8),
                'device': self.config['model']['device'],
                'save': True,
                'verbose': True,
                'project': 'models',
                'name': model_name,
                'exist_ok': True,
                'cache': False,
                'amp': True,
            }

            # 开始训练
            results = model.train(**train_args)

            # 验证模型（使用相同的固定验证集）
            metrics = model.val()
            print(f"训练完成! mAP50: {metrics.box.map50:.4f}")

            return model, metrics, dataset_config

        except Exception as e:
            print(f"训练错误: {e}")
            import traceback
            traceback.print_exc()
            gc.collect()
            return None, None, None

    def create_training_structure(self, train_data, val_data, output_dir, iteration=0):
        """创建训练目录结构"""
        from improved_data_splitter import ImprovedDataSplitter

        # 使用相同的噪声配置创建分割器
        temp_splitter = ImprovedDataSplitter(
            self.config['data']['dataset_path'],
            noise_config=self.config.get('noise_augmentation', {})
        )

        # 固定验证集目录 - 使用绝对路径确保一致性
        fixed_val_dir = Path("active_learning/fixed_validation").absolute()

        # 调试信息
        print(f"迭代 {iteration}: 检查固定验证集路径: {fixed_val_dir}")
        print(f"固定验证集是否存在: {fixed_val_dir.exists()}")

        # 只在第一次迭代创建固定验证集
        if iteration == 0:
            print("第一次迭代 - 创建固定验证集...")
            if fixed_val_dir.exists():
                print("删除已存在的固定验证集...")
                shutil.rmtree(fixed_val_dir)
            temp_splitter.create_fixed_validation_set(val_data, fixed_val_dir)
            print("✅ 固定验证集创建完成")
        else:
            # 后续迭代检查固定验证集是否存在
            if not fixed_val_dir.exists():
                print(f"⚠️ 警告: 迭代 {iteration} 中固定验证集不存在，重新创建...")
                temp_splitter.create_fixed_validation_set(val_data, fixed_val_dir)
            else:
                print(f"✅ 迭代 {iteration}: 复用固定验证集")

        # 创建训练结构（复用固定验证集）
        dataset_config = temp_splitter.create_training_structure_with_fixed_val(
            train_data, fixed_val_dir, output_dir
        )

        return dataset_config

    def check_validation_set_status(self):
        """检查验证集状态"""
        fixed_val_dir = Path("active_learning/fixed_validation").absolute()
        exists = fixed_val_dir.exists()

        if exists:
            image_count = len(list((fixed_val_dir / "images").glob("*.png")))
            label_count = len(list((fixed_val_dir / "labels").glob("*.txt")))
            print(f"固定验证集状态: 存在")
            print(f"  - 图像文件: {image_count} 个")
            print(f"  - 标注文件: {label_count} 个")
            print(f"  - 路径: {fixed_val_dir}")
        else:
            print(f"固定验证集状态: 不存在")

        return exists

    def select_uncertain_samples(self, model, unlabeled_pool, n_samples):
        """选择不确定性样本"""
        print(f"从 {len(unlabeled_pool)} 个未标注样本中选择 {n_samples} 个最不确定的样本")

        uncertainties = []
        valid_samples = []

        # 分批处理避免内存溢出
        batch_size = 20
        for i in range(0, len(unlabeled_pool), batch_size):
            batch = unlabeled_pool[i:i + batch_size]

            for img_path, label_path in batch:
                try:
                    results = model.predict(img_path, conf=0.1, verbose=False)
                    prediction = results[0]

                    if prediction.boxes is None or len(prediction.boxes) == 0:
                        uncertainty = 1.0  # 没有检测到目标，高不确定性
                    else:
                        confidences = prediction.boxes.conf.cpu().numpy()
                        uncertainty = 1.0 - np.min(confidences)  # 最小置信度

                    uncertainties.append(uncertainty)
                    valid_samples.append((img_path, label_path))

                except Exception as e:
                    print(f"预测错误 {img_path}: {e}")
                    uncertainties.append(1.0)
                    valid_samples.append((img_path, label_path))

            # 清理内存
            gc.collect()

        # 选择最不确定的样本
        if len(uncertainties) == 0:
            return []

        # 按不确定性排序
        uncertainty_indices = np.argsort(uncertainties)[::-1]  # 降序
        selected_indices = uncertainty_indices[:min(n_samples, len(uncertainties))]

        selected_samples = [valid_samples[i] for i in selected_indices]

        print(
            f"选择样本的不确定性范围: {uncertainties[selected_indices[0]]:.3f} - {uncertainties[selected_indices[-1]]:.3f}")

        return selected_samples

    def run_improved_active_learning(self):
        """运行改进的主动学习"""
        print("=== 改进的主动学习系统启动 ===")
        print("使用独立验证集确保评估一致性")

        # 检查初始状态
        print("\n初始验证集状态检查:")
        self.check_validation_set_status()

        try:
            # 1. 准备改进的数据集分割
            train_data, val_data, unlabeled_pool, splitter = self.prepare_improved_dataset()

            # 主动学习配置
            al_iterations = self.config['active_learning'].get('iterations', 3)
            samples_per_iteration = self.config['active_learning'].get('samples_per_iteration', 30)

            current_train_data = train_data.copy()

            for iteration in range(al_iterations):
                print(f"\n{'=' * 60}")
                print(f"主动学习迭代 {iteration + 1}/{al_iterations}")
                print(f"{'=' * 60}")

                # 迭代前验证集状态检查
                print(f"迭代 {iteration} 前验证集状态:")
                self.check_validation_set_status()

                # 2. 训练模型（使用固定验证集）
                model_name = f"improved_al_iteration_{iteration}"
                model, metrics, dataset_config = self.train_model_with_fixed_val(
                    current_train_data, val_data, model_name, iteration
                )

                # 迭代后验证集状态检查
                print(f"迭代 {iteration} 后验证集状态:")
                self.check_validation_set_status()



                if model is None:
                    print(f"迭代 {iteration} 训练失败，跳过")
                    continue

                # 记录性能
                performance = {
                    'iteration': iteration,
                    'train_samples': len(current_train_data),
                    'val_samples': len(val_data),  # 固定值
                    'unlabeled_samples': len(unlabeled_pool),
                    'mAP50': metrics.box.map50 if metrics else 0,
                    'mAP50_95': metrics.box.map if metrics else 0,
                }
                self.performance_history.append(performance)

                print(f"迭代 {iteration} 性能:")
                print(f"  mAP50: {performance['mAP50']:.4f}")
                print(f"  训练样本: {performance['train_samples']}")
                print(f"  验证样本: {performance['val_samples']} (固定)")
                print(f"  未标注样本: {performance['unlabeled_samples']}")

                # 3. 如果没有未标注数据，停止
                if not unlabeled_pool:
                    print("没有更多未标注数据，停止主动学习")
                    break

                # 4. 选择最不确定的样本
                selected_samples = self.select_uncertain_samples(
                    model, unlabeled_pool, samples_per_iteration
                )

                print(f"选择了 {len(selected_samples)} 个最不确定的样本")

                # 5. 更新数据集
                # 从未标注池中移除选中的样本
                unlabeled_pool = [sample for sample in unlabeled_pool if sample not in selected_samples]

                # 将选中的样本加入训练集
                current_train_data.extend(selected_samples)

                print(f"数据集更新:")
                print(f"  + 新增样本: {len(selected_samples)}")
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
            self.save_improved_results()

            print("\n🎉 改进的主动学习完成！")
            return self.performance_history

        except Exception as e:
            print(f"主动学习系统错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_improved_results(self):
        """保存改进的结果"""
        try:
            # 保存性能历史
            with open('results/improved_performance_history.json', 'w', encoding='utf-8') as f:
                json.dump(self.performance_history, f, indent=2, ensure_ascii=False)

            # 生成改进的报告
            self.generate_improved_report()

            # 绘制学习曲线
            self.plot_improved_learning_curve()

        except Exception as e:
            print(f"保存结果错误: {e}")

    def generate_improved_report(self):
        """生成改进的报告"""
        if not self.performance_history:
            return

        report = "改进的主动学习系统报告\n"
        report += "=====================\n"
        report += "特点: 使用独立验证集确保评估一致性\n\n"

        # 验证集信息
        if self.performance_history:
            val_samples = self.performance_history[0]['val_samples']
            report += f"固定验证集大小: {val_samples} 样本\n\n"

        for perf in self.performance_history:
            report += f"迭代 {perf['iteration']}: "
            report += f"mAP50={perf['mAP50']:.4f}, "
            report += f"训练样本={perf['train_samples']}, "
            report += f"验证样本={perf['val_samples']}\n"

        if len(self.performance_history) > 1:
            initial = self.performance_history[0]
            final = self.performance_history[-1]
            improvement = final['mAP50'] - initial['mAP50']
            samples_added = final['train_samples'] - initial['train_samples']

            report += f"\n性能总结:\n"
            report += f"  初始性能: mAP50={initial['mAP50']:.4f} (训练样本={initial['train_samples']})\n"
            report += f"  最终性能: mAP50={final['mAP50']:.4f} (训练样本={final['train_samples']})\n"
            report += f"  性能提升: {improvement:.4f}\n"
            report += f"  新增样本: {samples_added}\n"
            report += f"  学习效率: {improvement / samples_added:.6f} mAP50/样本\n"

        print(report)

        with open('results/improved_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)

    def plot_improved_learning_curve(self):
        """绘制改进的学习曲线 - 英文版本"""
        if len(self.performance_history) < 2:
            return

        try:
            import matplotlib.pyplot as plt
            # 设置字体避免中文警告
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            iterations = [p['iteration'] for p in self.performance_history]
            map50_scores = [p['mAP50'] for p in self.performance_history]
            map50_95_scores = [p['mAP50_95'] for p in self.performance_history]
            train_samples = [p['train_samples'] for p in self.performance_history]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # mAP50 曲线 - 使用英文标签
            ax1.plot(iterations, map50_scores, 'o-', linewidth=2, markersize=8, color='blue', label='mAP50')
            ax1.plot(iterations, map50_95_scores, 's-', linewidth=2, markersize=8, color='green', label='mAP50-95')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('mAP Score')
            ax1.set_title('Active Learning Performance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 训练样本数量 - 使用英文标签
            ax2.plot(iterations, train_samples, '^-', linewidth=2, markersize=8, color='red')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Training Samples')
            ax2.set_title('Training Data Growth\n(Fixed Validation Set)')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('results/improved_learning_curve.png', dpi=150, bbox_inches='tight')
            plt.show()  # 关闭图形避免内存泄漏

            print("学习曲线已保存: results/improved_learning_curve.png")

        except Exception as e:
            print(f"绘制学习曲线错误: {e}")


def main():
    """主函数"""
    print("启动改进的主动学习系统...")

    # 强制清理内存
    gc.collect()

    trainer = ImprovedActiveLearning()
    performance_history = trainer.run_improved_active_learning()

    if performance_history:
        print("\n=== 改进的主动学习完成 ===")
        if len(performance_history) > 1:
            initial = performance_history[0]
            final = performance_history[-1]
            improvement = final['mAP50'] - initial['mAP50']
            samples_added = final['train_samples'] - initial['train_samples']

            print(f"初始性能: mAP50={initial['mAP50']:.4f}, 训练样本={initial['train_samples']}")
            print(f"最终性能: mAP50={final['mAP50']:.4f}, 训练样本={final['train_samples']}")
            print(f"性能提升: {improvement:.4f}")
            print(f"新增样本: {samples_added}")
            print(f"学习效率: {improvement / samples_added:.6f} mAP50/样本")
    else:
        print("\n改进的主动学习失败")


if __name__ == "__main__":
    main()