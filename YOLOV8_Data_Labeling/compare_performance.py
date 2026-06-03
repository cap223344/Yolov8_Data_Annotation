# compare_performance.py
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import json
import matplotlib.pyplot as plt
from ultralytics import YOLO


def compare_models():
    """比较标准模型和注意力模型的性能"""

    print("=== 模型性能对比 ===")

    # 模型路径
    standard_model_path = "models/spectrogram_detector/weights/best.pt"
    attention_model_path = "models/attention_al_iteration_0_best.pt"  # 假设已经训练

    models_to_test = []

    if os.path.exists(standard_model_path):
        models_to_test.append(("标准YOLOv8", standard_model_path))

    if os.path.exists(attention_model_path):
        models_to_test.append(("注意力YOLOv8", attention_model_path))

    if not models_to_test:
        print("没有找到可比较的模型")
        return

    # 测试数据集
    test_dataset = "data/training/dataset.yaml"

    results = {}

    for model_name, model_path in models_to_test:
        print(f"\n测试模型: {model_name}")

        model = YOLO(model_path)
        metrics = model.val(data=test_dataset, verbose=False)

        results[model_name] = {
            'mAP50': metrics.box.map50,
            'mAP50_95': metrics.box.map,
            'precision': metrics.box.precision,
            'recall': metrics.box.recall
        }

        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")

    # 绘制对比图
    plot_comparison(results)

    # 保存结果
    with open('results/model_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def plot_comparison(results):
    """绘制性能对比图"""
    model_names = list(results.keys())
    map50_scores = [results[name]['mAP50'] for name in model_names]
    map_scores = [results[name]['mAP50_95'] for name in model_names]

    x = range(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar([i - width / 2 for i in x], map50_scores, width, label='mAP50', alpha=0.8)
    bars2 = ax.bar([i + width / 2 for i in x], map_scores, width, label='mAP50-95', alpha=0.8)

    ax.set_xlabel('模型')
    ax.set_ylabel('得分')
    ax.set_title('模型性能对比')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    compare_models()