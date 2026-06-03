# check_migration.py
import os
import sys
from pathlib import Path


def check_migration():
    """检查项目迁移状态"""
    print("=== 项目迁移检查 ===")

    # 新项目根目录
    new_root = Path("/")

    # 检查关键文件和目录
    critical_paths = [
        new_root / "data",
        new_root / "models",
        new_root / "configs",
        new_root / "utils",
        new_root / "train_spectrogram.py",
        new_root / "final_test.py",
        new_root / "deploy_model.py"
    ]

    print("检查关键文件...")
    for path in critical_paths:
        if path.exists():
            print(f"✅ {path.name} - 存在")
        else:
            print(f"❌ {path.name} - 缺失")

    # 检查数据集
    dataset_path = new_root / "Data"
    if dataset_path.exists():
        image_files = list(dataset_path.rglob("*.png"))
        print(f"✅ 数据集 - 找到 {len(image_files)} 个图像文件")
    else:
        print("❌ 数据集目录不存在")

    # 检查训练好的模型
    model_path = new_root / "models" / "spectrogram_detector" / "weights" / "best.pt"
    if model_path.exists():
        print("✅ 训练好的模型 - 存在")
    else:
        print("❌ 训练好的模型 - 缺失")

    # 检查Python路径
    print(f"\n当前Python路径: {sys.executable}")
    print(f"项目根目录: {new_root}")

    return all(path.exists() for path in critical_paths)


if __name__ == "__main__":
    success = check_migration()
    if success:
        print("\n🎉 迁移检查通过！项目可以正常运行。")
    else:
        print("\n⚠️ 迁移存在问题，请检查上述缺失的文件。")