# compare_al_results.py
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def compare_active_learning_results():
    """比较主动学习结果"""

    print("=== 主动学习结果比较 ===")

    # 查找所有性能历史文件
    result_files = list(Path("results").glob("*performance_history.json"))

    if not result_files:
        print("没有找到性能历史文件")
        return

    results = {}

    for result_file in result_files:
        config_name = result_file.parent.name if result_file.parent.name != "results" else "standard"

        with open(result_file, 'r', encoding='utf-8') as f:
            performance_data = json.load(f)

        results[config_name] = performance_data

    # 绘制比较图
    plot_comparison(results)

    # 生成比较报告
    generate_comparison_report(results)

    return results


def plot_comparison(results):
    """绘制结果比较图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for i, (config_name, performance_data) in enumerate(results.items()):
        color = colors[i % len(colors)]

        iterations = [p['iteration'] for p in performance_data]
        map50_scores = [p['mAP50'] for p in performance_data]
        map_scores = [p['mAP50_95'] for p in performance_data]
        train_samples = [p['train_samples'] for p in performance_data]

        # mAP50比较
        ax1.plot(iterations, map50_scores, 'o-', linewidth=2, markersize=6,
                 color=color, label=config_name)

        # mAP50-95比较
        ax2.plot(iterations, map_scores, 's-', linewidth=2, markersize=6,
                 color=color, label=config_name)

        # 训练样本增长
        ax3.plot(iterations, train_samples, '^-', linewidth=2, markersize=6,
                 color=color, label=config_name)

        # 效率曲线 (性能vs数据量)
        ax4.scatter(train_samples, map50_scores, s=80, alpha=0.7,
                    color=color, label=config_name)

    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('mAP50')
    ax1.set_title('主动学习性能比较 (mAP50)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('mAP50-95')
    ax2.set_title('主动学习性能比较 (mAP50-95)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.set_xlabel('迭代次数')
    ax3.set_ylabel('训练样本数量')
    ax3.set_title('训练数据增长比较')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4.set_xlabel('训练样本数量')
    ax4.set_ylabel('mAP50')
    ax4.set_title('学习效率比较')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/al_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def generate_comparison_report(results):
    """生成比较报告"""
    report = {
        'comparison_date': str(np.datetime64('now')),
        'configurations': {},
        'summary': {}
    }

    for config_name, performance_data in results.items():
        if not performance_data:
            continue

        initial = performance_data[0]
        final = performance_data[-1]

        report['configurations'][config_name] = {
            'initial_performance': {
                'mAP50': initial['mAP50'],
                'train_samples': initial['train_samples']
            },
            'final_performance': {
                'mAP50': final['mAP50'],
                'train_samples': final['train_samples']
            },
            'improvement': {
                'mAP50_gain': final['mAP50'] - initial['mAP50'],
                'samples_added': final['train_samples'] - initial['train_samples'],
                'efficiency': (final['mAP50'] - initial['mAP50']) /
                              (final['train_samples'] - initial['train_samples'])
                if final['train_samples'] > initial['train_samples'] else 0
            }
        }

    # 找出最佳配置
    best_config = None
    best_improvement = -1

    for config_name, data in report['configurations'].items():
        improvement = data['improvement']['mAP50_gain']
        if improvement > best_improvement:
            best_improvement = improvement
            best_config = config_name

    report['summary']['best_configuration'] = best_config
    report['summary']['best_improvement'] = best_improvement

    # 保存报告
    with open('results/comparison_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 文本报告
    txt_report = "主动学习系统比较报告\n"
    txt_report += "==================\n\n"

    for config_name, data in report['configurations'].items():
        txt_report += f"配置: {config_name}\n"
        txt_report += f"  初始: mAP50={data['initial_performance']['mAP50']:.4f}, 样本={data['initial_performance']['train_samples']}\n"
        txt_report += f"  最终: mAP50={data['final_performance']['mAP50']:.4f}, 样本={data['final_performance']['train_samples']}\n"
        txt_report += f"  提升: mAP50+{data['improvement']['mAP50_gain']:.4f}, 新增样本={data['improvement']['samples_added']}\n"
        txt_report += f"  效率: {data['improvement']['efficiency']:.6f} mAP50/样本\n\n"

    txt_report += f"最佳配置: {report['summary']['best_configuration']}\n"
    txt_report += f"最佳提升: {report['summary']['best_improvement']:.4f}\n"

    with open('results/comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write(txt_report)

    print(txt_report)


if __name__ == "__main__":
    compare_active_learning_results()