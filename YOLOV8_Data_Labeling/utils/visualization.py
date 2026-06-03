# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from pathlib import Path


def plot_spectrogram_comparison(original_iq, processed_spectrogram, title="Spectrogram Comparison"):
    """绘制原始IQ数据和时频图对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 原始IQ数据（实部和虚部）
    time_axis = np.arange(len(original_iq)) / 1e6  # 假设采样率1MHz
    ax1.plot(time_axis, np.real(original_iq), label='I component', alpha=0.7)
    ax1.plot(time_axis, np.imag(original_iq), label='Q component', alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Original IQ Data')
    ax1.legend()
    ax1.grid(True)

    # 时频图
    if len(processed_spectrogram.shape) == 2:
        im = ax2.imshow(processed_spectrogram, aspect='auto', cmap='viridis',
                        extent=[0, time_axis[-1], 0, 0.5])  # 假设频率范围0-0.5MHz
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (MHz)')
        ax2.set_title('Spectrogram')
        plt.colorbar(im, ax=ax2, label='Power (dB)')

    plt.tight_layout()
    return fig


def plot_detection_results(image_path, predictions, class_names, save_path=None):
    """绘制检测结果"""
    # 加载图像
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image)

    # 绘制边界框
    for box in predictions.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().item()
        cls = int(box.cls[0].cpu().item())

        # 绘制矩形框
        rect = plt.Rectangle((xyxy[0], xyxy[1]), xyxy[2] - xyxy[0], xyxy[3] - xyxy[1],
                             fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        # 添加标签
        label = f"{class_names[cls]}: {conf:.2f}"
        ax.text(xyxy[0], xyxy[1] - 5, label, color='red', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    ax.set_title('Detection Results')
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig


def plot_learning_curve(performance_history, save_path=None):
    """绘制学习曲线"""
    iterations = range(1, len(performance_history) + 1)
    map50 = [perf.get('map50', 0) for perf in performance_history]
    map50_95 = [perf.get('map50_95', 0) for perf in performance_history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, map50, 'o-', label='mAP@0.5', linewidth=2)
    ax.plot(iterations, map50_95, 's-', label='mAP@0.5:0.95', linewidth=2)

    ax.set_xlabel('Active Learning Iteration')
    ax.set_ylabel('mAP Score')
    ax.set_title('Active Learning Performance Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig


def plot_attention_maps(model, image_tensor, layer_names, save_path=None):
    """绘制注意力图"""
    # 这里需要实现注意力图的可视化
    # 由于模型结构复杂，这里提供框架
    fig, axes = plt.subplots(1, len(layer_names), figsize=(15, 5))

    if len(layer_names) == 1:
        axes = [axes]

    for ax, layer_name in zip(axes, layer_names):
        # 获取注意力权重并可视化
        ax.set_title(f'Attention: {layer_name}')
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig