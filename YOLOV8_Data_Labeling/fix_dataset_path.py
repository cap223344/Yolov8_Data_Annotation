# fix_dataset_path.py
import os
import yaml
from pathlib import Path


def fix_dataset_paths():
    """修复所有配置文件中的数据集路径"""

    print("=== 修复数据集路径 ===")

    # 首先找到数据集
    from find_dataset import find_dataset
    dataset_path = find_dataset()

    if not dataset_path:
        print("无法找到数据集，请手动指定路径")
        return False

    print(f"使用数据集路径: {dataset_path}")

    # 需要更新的配置文件
    config_files = [
        "configs/base_config.yaml",
        "configs/attention_config.yaml",
        "configs/low_memory_config.yaml"
    ]

    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"更新: {config_file}")

            # 读取配置
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 更新数据集路径
            if 'data' in config:
                config['data']['dataset_path'] = dataset_path

            # 保存配置
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)

            print(f"✅ {config_file} - 更新完成")
        else:
            print(f"⚠️ {config_file} - 文件不存在")

    # 更新训练脚本中的默认路径
    train_script = "train_spectrogram.py"
    if os.path.exists(train_script):
        print(f"更新: {train_script}")

        with open(train_script, 'r', encoding='utf-8') as f:
            content = f.read()

        # 替换默认路径
        old_path = 'default=\'D:/py/YOLOV8 Data Labeling Project/Data\''
        new_path = f'default=\'{dataset_path}\''
        content = content.replace(old_path, new_path)

        with open(train_script, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✅ {train_script} - 更新完成")

    print("\n路径修复完成！")
    return True


if __name__ == "__main__":
    fix_dataset_paths()