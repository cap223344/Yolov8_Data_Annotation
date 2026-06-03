# utils/metrics.py
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


def calculate_signal_detection_metrics(predictions, ground_truth, iou_threshold=0.5):
    """计算信号检测指标"""
    tp, fp, fn = 0, 0, 0

    for pred, gt in zip(predictions, ground_truth):
        # 这里需要实现具体的匹配逻辑
        # 简化实现
        if len(pred.boxes) > 0 and len(gt) > 0:
            tp += min(len(pred.boxes), len(gt))
            fp += max(0, len(pred.boxes) - len(gt))
            fn += max(0, len(gt) - len(pred.boxes))
        elif len(pred.boxes) > 0:
            fp += len(pred.boxes)
        else:
            fn += len(gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def plot_precision_recall_curve(predictions, ground_truth, class_names, save_path=None):
    """绘制精确率-召回率曲线"""
    # 收集所有预测的置信度和真实标签
    all_confidences = []
    all_true_labels = []

    for pred, gt in zip(predictions, ground_truth):
        if len(pred.boxes) > 0:
            confidences = pred.boxes.conf.cpu().numpy()
            all_confidences.extend(confidences)
            # 这里需要根据实际情况设置真实标签
            all_true_labels.extend([1] * len(confidences))  # 简化

    if not all_confidences:
        print("No predictions to plot PR curve")
        return None

    precision, recall, thresholds = precision_recall_curve(all_true_labels, all_confidences)
    ap = average_precision_score(all_true_labels, all_confidences)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, label=f'AP = {ap:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig, ap