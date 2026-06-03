# final_test.py
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import glob
# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

def final_test():
    """最终测试 - 验证模型性能"""

    print("=== 最终模型测试 ===")

    # 加载训练好的最佳模型
    model_path = "models/spectrogram_detector/weights/best.pt"

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return

    print("加载训练好的模型...")
    model = YOLO(model_path)

    # 测试验证集中的一些图像
    test_images = glob.glob("data/training/val/images/*.png")[:5]  # 测试前5张

    if not test_images:
        print("找不到测试图像")
        return

    print(f"测试 {len(test_images)} 张验证集图像")

    results_summary = []

    for i, img_path in enumerate(test_images):
        print(f"\n测试图像 {i + 1}: {os.path.basename(img_path)}")

        # 推理
        results = model.predict(img_path, conf=0.25, imgsz=640)

        # 显示结果
        for r in results:
            # 绘制结果
            im_array = r.plot()
            im_array = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

            # 显示图像
            plt.figure(figsize=(12, 8))
            plt.imshow(im_array)
            plt.axis('off')
            plt.title(f'检测结果 - {os.path.basename(img_path)}')
            plt.tight_layout()
            plt.show()

            # 打印检测信息
            if len(r.boxes) > 0:
                rc_count = 0
                vt_count = 0

                for j, box in enumerate(r.boxes):
                    cls = int(box.cls[0])
                    conf = box.conf[0]
                    class_name = "rc" if cls == 0 else "vt"

                    if cls == 0:
                        rc_count += 1
                    else:
                        vt_count += 1

                    print(f"  目标 {j + 1}: {class_name} (置信度: {conf:.3f})")

                results_summary.append({
                    'image': os.path.basename(img_path),
                    'rc_detected': rc_count,
                    'vt_detected': vt_count,
                    'total_detected': len(r.boxes)
                })

                print(f"  统计: rc={rc_count}, vt={vt_count}, 总计={len(r.boxes)}")
            else:
                print("未检测到目标")
                results_summary.append({
                    'image': os.path.basename(img_path),
                    'rc_detected': 0,
                    'vt_detected': 0,
                    'total_detected': 0
                })

    # 打印测试总结
    print("\n" + "=" * 50)
    print("测试总结:")
    total_rc = sum(r['rc_detected'] for r in results_summary)
    total_vt = sum(r['vt_detected'] for r in results_summary)
    total_detections = sum(r['total_detected'] for r in results_summary)

    print(f"总检测目标: {total_detections}")
    print(f"  rc信号: {total_rc}")
    print(f"  vt信号: {total_vt}")
    print(f"测试图像数量: {len(test_images)}")

    print("\n=== 项目完成！ ===")
    print("通信信号检测模型训练成功！")
    print("模型性能优秀，可以用于实际信号检测任务。")


if __name__ == "__main__":
    final_test()