# check_project_structure.py
import os
from pathlib import Path


def check_structure():
    """检查项目结构"""
    print("=== 项目结构检查 ===")

    project_root = Path(".").absolute()
    print(f"项目根目录: {project_root}")

    # 检查关键目录
    dirs_to_check = [
        "configs",
        "data",
        "models",
        "utils",
        "active_learning",
        "outputs"
    ]

    print("\n检查目录结构:")
    for dir_name in dirs_to_check:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ - 缺失")

    # 检查关键文件
    files_to_check = [
        "configs/base_config.yaml",
        "configs/attention_config.yaml",
        "train_spectrogram.py",
        "final_test.py",
        "deploy_model.py"
    ]

    print("\n检查关键文件:")
    for file_path in files_to_check:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 缺失")


if __name__ == "__main__":
    check_structure()