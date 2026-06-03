# annotation_expert.py
import os
import cv2
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import ttk
import threading


class ExpertAnnotator:
    """专家标注界面"""

    def __init__(self, unlabeled_dir, output_dir):
        self.unlabeled_dir = Path(unlabeled_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 获取未标注图像
        self.unlabeled_images = list(self.unlabeled_dir.glob("*.png")) + list(self.unlabeled_dir.glob("*.jpg"))
        self.current_index = 0
        self.current_boxes = []

        # 类别映射
        self.classes = {0: "rc", 1: "vt"}
        self.current_class = 0

        print(f"找到 {len(self.unlabeled_images)} 个未标注图像")

    def start_annotation(self):
        """开始标注"""
        if not self.unlabeled_images:
            print("没有找到未标注图像")
            return

        print("启动专家标注界面...")
        print("使用说明:")
        print("  - 左键点击拖动: 绘制边界框")
        print("  - 右键点击: 删除最后一个边界框")
        print("  - 数字键1: 选择rc类别")
        print("  - 数字键2: 选择vt类别")
        print("  - 空格键: 保存并下一张")
        print("  - ESC键: 退出")

        self.annotate_image(self.current_index)

    def annotate_image(self, index):
        """标注单张图像"""
        if index >= len(self.unlabeled_images):
            print("所有图像标注完成！")
            return

        image_path = self.unlabeled_images[index]
        print(f"\n标注图像 {index + 1}/{len(self.unlabeled_images)}: {image_path.name}")

        # 加载图像
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 创建matplotlib图形
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(image)
        self.ax.set_title(f'标注: {image_path.name} (当前类别: {self.classes[self.current_class]})')
        self.ax.axis('off')

        # 初始化变量
        self.current_boxes = []
        self.start_x = None
        self.start_y = None
        self.rect = None

        # 连接事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.tight_layout()
        plt.show()

    def on_press(self, event):
        """鼠标按下事件"""
        if event.button == 1:  # 左键
            if event.inaxes == self.ax:
                self.start_x = event.xdata
                self.start_y = event.ydata
                self.rect = Rectangle((self.start_x, self.start_y), 0, 0,
                                      fill=False, edgecolor='red', linewidth=2)
                self.ax.add_patch(self.rect)
                self.fig.canvas.draw()

        elif event.button == 3:  # 右键
            if self.current_boxes:
                self.current_boxes.pop()
                self.redraw_boxes()

    def on_release(self, event):
        """鼠标释放事件"""
        if event.button == 1 and self.start_x is not None:  # 左键释放
            if event.inaxes == self.ax:
                end_x = event.xdata
                end_y = event.ydata

                # 计算边界框
                x1 = min(self.start_x, end_x)
                y1 = min(self.start_y, end_y)
                x2 = max(self.start_x, end_x)
                y2 = max(self.start_y, end_y)

                width = x2 - x1
                height = y2 - y1

                # 转换为YOLO格式
                img_width = self.ax.get_images()[0].get_size()[0]
                img_height = self.ax.get_images()[0].get_size()[1]

                x_center = (x1 + width / 2) / img_width
                y_center = (y1 + height / 2) / img_height
                width_norm = width / img_width
                height_norm = height / img_height

                # 添加到当前框列表
                self.current_boxes.append({
                    'class_id': self.current_class,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width_norm,
                    'height': height_norm,
                    'bbox': [x1, y1, x2, y2]  # 用于绘制
                })

                self.redraw_boxes()

            # 重置
            self.start_x = None
            self.start_y = None
            self.rect = None

    def on_motion(self, event):
        """鼠标移动事件"""
        if self.start_x is not None and event.inaxes == self.ax:
            if self.rect:
                self.rect.set_width(event.xdata - self.start_x)
                self.rect.set_height(event.ydata - self.start_y)
                self.fig.canvas.draw()

    def on_key(self, event):
        """键盘事件"""
        if event.key == '1':
            self.current_class = 0
            self.ax.set_title(f'标注: {self.unlabeled_images[self.current_index].name} (当前类别: rc)')
            self.fig.canvas.draw()
            print("切换到 rc 类别")

        elif event.key == '2':
            self.current_class = 1
            self.ax.set_title(f'标注: {self.unlabeled_images[self.current_index].name} (当前类别: vt)')
            self.fig.canvas.draw()
            print("切换到 vt 类别")

        elif event.key == ' ':
            self.save_annotation()
            self.current_index += 1
            plt.close(self.fig)
            if self.current_index < len(self.unlabeled_images):
                self.annotate_image(self.current_index)
            else:
                print("所有图像标注完成！")

        elif event.key == 'escape':
            print("标注中断")
            plt.close(self.fig)

    def redraw_boxes(self):
        """重绘所有边界框"""
        # 清除所有矩形
        for patch in self.ax.patches[1:]:  # 保留第一个动态矩形
            patch.remove()

        # 绘制所有边界框
        for i, box in enumerate(self.current_boxes):
            x1, y1, x2, y2 = box['bbox']
            width = x2 - x1
            height = y2 - y1

            color = 'red' if box['class_id'] == 0 else 'blue'
            rect = Rectangle((x1, y1), width, height,
                             fill=False, edgecolor=color, linewidth=2)
            self.ax.add_patch(rect)

            # 添加类别标签
            class_name = self.classes[box['class_id']]
            self.ax.text(x1, y1 - 5, f'{class_name}',
                         color=color, fontsize=10, weight='bold')

        self.fig.canvas.draw()

    def save_annotation(self):
        """保存标注"""
        if not self.current_boxes:
            print("没有标注任何目标，跳过保存")
            return

        image_path = self.unlabeled_images[self.current_index]

        # 复制图像到输出目录
        dest_image = self.output_dir / 'images' / image_path.name
        dest_image.parent.mkdir(parents=True, exist_ok=True)

        import shutil
        shutil.copy2(image_path, dest_image)

        # 保存标注文件
        label_file = self.output_dir / 'labels' / f"{image_path.stem}.txt"
        label_file.parent.mkdir(parents=True, exist_ok=True)

        with open(label_file, 'w') as f:
            for box in self.current_boxes:
                line = f"{box['class_id']} {box['x_center']:.6f} {box['y_center']:.6f} {box['width']:.6f} {box['height']:.6f}\n"
                f.write(line)

        print(f"标注已保存: {label_file}")
        print(f"标注了 {len(self.current_boxes)} 个目标")


def main():
    """主函数"""
    unlabeled_dir = "D:/py/启用主动学习和注意力机制的yolov8/Data/data with no label"
    output_dir = "data/expert_annotated"

    if not os.path.exists(unlabeled_dir):
        print(f"未标注数据目录不存在: {unlabeled_dir}")
        return

    annotator = ExpertAnnotator(unlabeled_dir, output_dir)
    annotator.start_annotation()


if __name__ == "__main__":
    main()