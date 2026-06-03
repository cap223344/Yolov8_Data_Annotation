# test_noise_fix.py
import sys

sys.path.append('.')
from improved_data_splitter import ImprovedDataSplitter
import yaml


def test_noise_application():
    """测试噪声应用"""

    try:
        # 加载配置 - 使用utf-8编码
        with open('configs/improved_al_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        noise_config = config.get('noise_augmentation', {})
        print("噪声配置:", noise_config)

        # 创建分割器
        splitter = ImprovedDataSplitter(
            dataset_path=config['data']['dataset_path'],
            noise_config=noise_config
        )

        # 测试少量数据
        valid_data = splitter.discover_valid_data()[:5]  # 只取前5个样本测试

        if valid_data:
            # 创建测试目录
            test_output_dir = "test_noise_output"
            dataset_config = splitter.create_training_structure(
                valid_data, valid_data, test_output_dir
            )

            print(f"测试完成，检查目录: {test_output_dir}")
            return True
        else:
            print("没有找到有效数据")
            return False

    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_noise_application()