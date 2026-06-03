# update_paths.py
import os
import fileinput
import sys


def update_file_paths():
    """更新所有脚本中的路径引用"""

    old_base = "D:/py/YOLOV8 Data Labeling Project"
    new_base = "D:/py/启用主动学习和注意力机制的yolov8"

    files_to_update = [
        "train_spectrogram.py",
        "final_test.py",
        "deploy_model.py",
        "configs/base_config.yaml",
        "configs/attention_config.yaml"
    ]

    print("开始更新文件路径...")

    for file_path in files_to_update:
        if os.path.exists(file_path):
            print(f"更新: {file_path}")

            with fileinput.FileInput(file_path, inplace=True, encoding='utf-8') as file:
                for line in file:
                    # 替换路径
                    new_line = line.replace(old_base, new_base)
                    print(new_line, end='')

            print(f"✅ {file_path} - 更新完成")
        else:
            print(f"❌ {file_path} - 文件不存在")

    print("\n路径更新完成！")


if __name__ == "__main__":
    update_file_paths()