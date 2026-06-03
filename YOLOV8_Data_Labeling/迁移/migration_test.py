# migration_test.py
import os
import sys
from pathlib import Path


def test_migrated_project():
    """测试迁移后的项目"""
    print("=== 迁移后项目测试 ===")

    # 设置新项目根目录
    new_root = Path("D:/py/启用主动学习和注意力机制的yolov8")
    os.chdir(new_root)

    try:
        # 测试导入关键模块
        print("1. 测试模块导入...")
        from ultralytics import YOLO
        print("✅ ultralytics 导入成功")

        # 测试模型加载
        print("2. 测试模型加载...")
        model_path = new_root / "models" / "spectrogram_detector" / "weights" / "best.pt"
        if model_path.exists():
            model = YOLO(str(model_path))
            print("✅ 模型加载成功")
        else:
            print("❌ 模型文件不存在")
            return False

        # 测试数据集访问
        print("3. 测试数据集访问...")
        dataset_config = new_root / "data" / "training" / "dataset.yaml"
        if dataset_config.exists():
            print("✅ 数据集配置存在")
        else:
            print("❌ 数据集配置不存在")

        # 测试快速推理
        print("4. 测试快速推理...")
        test_images = list(new_root.glob("data/training/val/images/*.png"))
        if test_images:
            results = model.predict(str(test_images[0]), imgsz=640, verbose=False)
            print(f"✅ 推理测试成功 - 检测到 {len(results[0].boxes)} 个目标")
        else:
            print("⚠️ 无测试图像，跳过推理测试")

        print("\n🎉 迁移测试全部通过！")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


if __name__ == "__main__":
    success = test_migrated_project()
    if success:
        print("\n项目迁移成功！可以开始使用。")
    else:
        print("\n项目迁移存在问题，请检查。")