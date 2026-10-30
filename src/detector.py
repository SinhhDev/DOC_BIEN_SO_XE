import os
import cv2
import numpy as np
from ultralytics import YOLO
from src.config import YOLO_MODEL_PATH, DETECTION_CONF_THRESHOLD, IOU_THRESHOLD


class PlateDetector:
    """
    Module phát hiện vùng biển số xe:
    - Sử dụng YOLOv8 License Plate Detection làm bộ phát hiện chính
    - Tích hợp bộ dò Contour/Morphological của OpenCV làm fallback
    """

    def __init__(self, model_path: str = YOLO_MODEL_PATH, conf_threshold: float = DETECTION_CONF_THRESHOLD):
        self.conf_threshold = conf_threshold
        self.yolo_model = None

        # Tự động tải mô hình nếu chưa tồn tại
        if not os.path.exists(model_path):
            print(f"[INFO] Không tìm thấy file mô hình tại {model_path}. Đang tự động tải từ Hugging Face...")
            try:
                from huggingface_hub import hf_hub_download
                import shutil
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                downloaded = hf_hub_download(repo_id="Koushim/yolov8-license-plate-detection", filename="best.pt")
                shutil.copy(downloaded, model_path)
                print(f"[INFO] Tải mô hình thành công: {model_path}")
            except Exception as e:
                print(f"[Warning] Không thể tự động tải model từ Hugging Face: {e}")

        if os.path.exists(model_path):
            try:
                self.yolo_model = YOLO(model_path)
            except Exception as e:
                print(f"[Warning] Không thể nạp YOLO model: {e}. Sẽ dùng OpenCV fallback.")
        else:
            print(f"[Warning] Không tìm thấy model tại {model_path}. Sẽ dùng OpenCV fallback.")

    def detect(self, image: np.ndarray) -> list[dict]:
        """
        Phát hiện biển số xe trong ảnh.
        Trả về danh sách các detections:
        [
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "source": "yolo" | "opencv_fallback"
            },
            ...
        ]
        """
        if image is None or image.size == 0:
            return []

        results = []

        # 1. Thử dùng YOLO
        if self.yolo_model is not None:
            try:
                yolo_preds = self.yolo_model.predict(
                    source=image,
                    conf=self.conf_threshold,
                    iou=IOU_THRESHOLD,
                    verbose=False
                )

                raw_boxes = []
                for pred in yolo_preds:
                    boxes = pred.boxes
                    for box in boxes:
                        coords = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0].cpu().numpy())
                        raw_boxes.append({
                            "bbox": [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])],
                            "confidence": round(conf, 3),
                            "source": "yolo"
                        })

                # Lọc bỏ các box lồng nhau hoặc bị trùng lặp
                results = self._filter_overlapping_boxes(raw_boxes)
            except Exception as e:
                print(f"[Error] Lỗi khi chạy YOLO detect: {e}")

        # 2. Nếu YOLO không tìm thấy biển nào, kích hoạt OpenCV fallback
        if len(results) == 0:
            fallback_boxes = self._opencv_contour_detect(image)
            for fb in fallback_boxes:
                results.append({
                    "bbox": fb,
                    "confidence": 0.50,
                    "source": "opencv_fallback"
                })

        return results

    def _opencv_contour_detect(self, image: np.ndarray) -> list[list[int]]:
        """
        Thuật toán fallback bằng biến đổi hình thái học và tìm contour hình chữ nhật.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Blackhat filter làm nổi bật chi tiết chữ màu tối trên nền sáng
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

        # Sobel x để phát hiện các cạnh đứng của chữ
        grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        min_val, max_val = np.min(grad_x), np.max(grad_x)
        if max_val - min_val > 0:
            grad_x = 255 * ((grad_x - min_val) / (max_val - min_val))
        grad_x = grad_x.astype("uint8")

        # Gaussian blur + đóng hình thái (morphological closing)
        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
        _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Dãn nở để nối các vùng ký tự lại thành 1 khối
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            aspect_ratio = bw / float(bh)
            area = bw * bh
            img_area = w * h

            # Biển số xe VN:
            # - Biển dài ô tô: aspect ratio khoảng 3.0 -> 5.5
            # - Biển vuông (xe máy, ô tô vuông): aspect ratio khoảng 1.1 -> 1.7
            # Diện tích từ 0.5% đến 25% bức ảnh
            is_valid_ratio = (1.1 <= aspect_ratio <= 5.8)
            is_valid_area = (0.005 * img_area <= area <= 0.35 * img_area)

            if is_valid_ratio and is_valid_area:
                candidates.append([x, y, x + bw, y + bh])

        return candidates[:3]  # Lấy tối đa 3 ứng viên

    def _filter_overlapping_boxes(self, detections: list[dict]) -> list[dict]:
        """
        Lọc các box trùng hoặc lồng nhau.
        Nếu box A nằm trong hoặc giao thoa lớn với box B, giữ box to hơn (bao trọn biển).
        """
        if len(detections) <= 1:
            return detections

        # Sắp xếp theo diện tích giảm dần để ưu tiên box bao quát lớn hơn
        sorted_dets = sorted(
            detections,
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
            reverse=True
        )

        keep = []
        for det in sorted_dets:
            b1 = det["bbox"]
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            is_redundant = False

            for kept in keep:
                b2 = kept["bbox"]
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

                # Tính phần giao nhau (Intersection)
                ix1 = max(b1[0], b2[0])
                iy1 = max(b1[1], b2[1])
                ix2 = min(b1[2], b2[2])
                iy2 = min(b1[3], b2[3])

                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    min_area = min(area1, area2)
                    overlap_ratio = inter_area / float(min_area)

                    # Nếu phần giao nhau chiếm trên 40% của box nhỏ hơn, coi là trùng lặp
                    if overlap_ratio > 0.40:
                        is_redundant = True
                        break

            if not is_redundant:
                keep.append(det)

        return keep
