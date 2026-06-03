# find_dataset.py
import os
from pathlib import Path


def find_dataset():
    """查找数据集位置"""
    print("=== 查找数据集 ===")

    # 可能的路径
    possible_paths = [
        "D:/py/启用主动学习和注意力机制的yolov8/Data",
        "D:/py/启用主动学习和注意力机制的yolov8/YOLOV8_Data_Labeling/Data",
        "D:/py/启用主动学习和注意力机制的yolov8/data",
        "D:/py/启用主动学习和注意力机制的yolov8/YOLOV8_Data_Labeling/data",
        "D:/py/yolov8/YOLOV8_Data_Labeling/Data",  # 从测试输出中看到的路径
        "D:/py/yolov8/Data"
    ]

    for path in possible_paths:
        path_obj = Path(path)
        if path_obj.exists():
            print(f"✅ 找到数据集: {path}")
            # 统计文件数量
            png_files = list(path_obj.rglob("*.png"))
            txt_files = list(path_obj.rglob("*.txt"))
            print(f"   图像文件: {len(png_files)}")
            print(f"   标注文件: {len(txt_files)}")
            return path

    print("❌ 未找到数据集")
    return None


if __name__ == "__main__":
    dataset_path = find_dataset()
    if dataset_path:
        print(f"\n数据集位置: {dataset_path}")
    else:
        print("请检查数据集是否已正确复制到新位置")