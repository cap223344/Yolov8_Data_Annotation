# deploy_model.py
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ultralytics import YOLO
import cv2
import numpy as np


class SignalDetector:
    """信号检测器 - 用于部署"""

    def __init__(self, model_path="models/spectrogram_detector/weights/best.pt"):
        self.model = YOLO(model_path)
        self.class_names = ['rc', 'vt']

    def detect_signals(self, image_path, conf_threshold=0.25):
        """检测图像中的信号"""
        results = self.model.predict(image_path, conf=conf_threshold, imgsz=640)

        detections = []
        for r in results:
            if len(r.boxes) > 0:
                for box in r.boxes:
                    detection = {
                        'class': self.class_names[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                    }
                    detections.append(detection)

        return detections

    def batch_detect(self, image_folder, output_folder=None):
        """批量检测文件夹中的图像"""
        import glob

        image_files = glob.glob(os.path.join(image_folder, "*.png")) + \
                      glob.glob(os.path.join(image_folder, "*.jpg"))

        all_results = {}

        for img_path in image_files:
            detections = self.detect_signals(img_path)
            all_results[os.path.basename(img_path)] = detections

            # 保存带检测结果的图像
            if output_folder:
                self.save_detection_image(img_path, detections, output_folder)

        return all_results

    def save_detection_image(self, image_path, detections, output_folder):
        """保存带检测框的图像"""
        os.makedirs(output_folder, exist_ok=True)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 绘制检测框
        for det in detections:
            bbox = [int(coord) for coord in det['bbox']]
            class_name = det['class']
            confidence = det['confidence']

            # 绘制矩形框
            color = (255, 0, 0) if class_name == 'rc' else (0, 255, 0)  # rc蓝色, vt绿色
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # 添加标签
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 保存图像
        output_path = os.path.join(output_folder, f"detected_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        return output_path


def main():
    """主函数 - 演示如何使用训练好的模型"""
    detector = SignalDetector()

    print("=== 通信信号检测器部署 ===")
    print("模型加载成功！")

    # 测试单张图像
    test_image = "data/training/val/images/004a_20250531_221542.part1.iq_segment1.png"

    if os.path.exists(test_image):
        print(f"\n测试图像: {test_image}")
        detections = detector.detect_signals(test_image)

        print(f"检测到 {len(detections)} 个信号:")
        for i, det in enumerate(detections):
            print(f"  信号 {i + 1}: {det['class']} (置信度: {det['confidence']:.3f})")

        # 保存检测结果图像
        output_path = detector.save_detection_image(test_image, detections, "outputs")
        print(f"检测结果已保存: {output_path}")
    else:
        print("测试图像不存在，请检查路径")

    print("\n=== 部署完成 ===")
    print("模型已准备好用于实际信号检测任务！")


if __name__ == "__main__":
    main()