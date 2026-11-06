import time
import cv2
import numpy as np
from src.config import YOLO_MODEL_PATH, DETECTION_CONF_THRESHOLD
from src.detector import PlateDetector
from src.preprocessor import PlatePreprocessor
from src.ocr_engine import OCREngine
from src.visualizer import PlateVisualizer


class ALPRPipeline:
    """
    Toàn bộ luồng nhận diện biển số xe (Automatic License Plate Recognition):
    Image/Frame -> Plate Detection -> Preprocessing (Deskew/Enhance) -> OCR -> Postprocessing -> Visualization
    """

    def __init__(self, model_path: str = YOLO_MODEL_PATH, conf_threshold: float = DETECTION_CONF_THRESHOLD, use_gpu: bool = False):
        print("[ALPR] Đang khởi tạo mô hình Detection & OCR...")
        self.detector = PlateDetector(model_path=model_path, conf_threshold=conf_threshold)
        self.preprocessor = PlatePreprocessor()
        self.ocr_engine = OCREngine(gpu=use_gpu)
        self.visualizer = PlateVisualizer()
        print("[ALPR] Khởi tạo thành công!")

    def process_image(self, image_input) -> dict:
        """
        Nhận diện biển số xe từ đường dẫn file ảnh hoặc numpy array (OpenCV BGR).
        """
        start_time = time.time()

        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Không thể đọc file ảnh từ: {image_input}")
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
        else:
            raise TypeError("image_input phải là đường dẫn file ảnh (str) hoặc numpy ndarray")

        # 1. Phát hiện vị trí biển số xe
        detections = self.detector.detect(image)

        results = []
        for det in detections:
            bbox = det["bbox"]
            det_conf = det["confidence"]
            det_source = det["source"]

            # 2. Cắt vùng biển số (với padding)
            cropped = self.preprocessor.crop_with_padding(image, bbox, pad_ratio=0.04)
            if cropped is None or cropped.size == 0:
                continue

            # 3. Cân chỉnh xoay thẳng ảnh nếu bị nghiêng
            deskewed = self.preprocessor.deskew_plate(cropped)

            # 4. Tiền xử lý nâng cao (contrast CLAHE + nhị phân hóa)
            enhanced, thresh = self.preprocessor.enhance_for_ocr(deskewed)

            # 5. Nhận diện ký tự bằng OCR
            plate_info = self.ocr_engine.read_plate(deskewed, enhanced, thresh)

            # Cơ chế thông minh: Nếu chỉ nhận được 1 dòng số (4-5 số) mà không có mã tỉnh/chữ cái,
            # khả năng cao là biển số 2 dòng xe máy bị YOLO cắt mất dòng trên.
            # Tự động mở rộng box lên trên và nhận diện lại.
            if not plate_info.get("is_valid", False) and len(plate_info.get("clean_raw", "")) <= 5:
                x1, y1, x2, y2 = bbox
                bh = y2 - y1
                # Mở rộng lên trên 85% chiều cao hiện tại
                expanded_y1 = max(0, y1 - int(bh * 0.85))
                expanded_bbox = [x1, expanded_y1, x2, y2]
                exp_cropped = self.preprocessor.crop_with_padding(image, expanded_bbox, pad_ratio=0.04)
                if exp_cropped is not None and exp_cropped.size > 0:
                    exp_deskewed = self.preprocessor.deskew_plate(exp_cropped)
                    exp_enh, exp_thr = self.preprocessor.enhance_for_ocr(exp_deskewed)
                    exp_info = self.ocr_engine.read_plate(exp_deskewed, exp_enh, exp_thr)
                    # Nếu kết quả mở rộng tìm thấy biển số đầy đủ hơn
                    if len(exp_info.get("clean_raw", "")) > len(plate_info.get("clean_raw", "")):
                        bbox = expanded_bbox
                        cropped = exp_cropped
                        enhanced = exp_enh
                        thresh = exp_thr
                        plate_info = exp_info

            results.append({
                "bbox": bbox,
                "detection_conf": det_conf,
                "detection_source": det_source,
                "cropped_img": cropped,
                "enhanced_img": enhanced,
                "thresh_img": thresh,
                "plate_info": plate_info
            })

        # 6. Vẽ kết quả lên ảnh gốc
        annotated_image = self.visualizer.draw_results(image, results)
        latency = (time.time() - start_time) * 1000.0  # ms

        return {
            "num_plates": len(results),
            "plates": results,
            "original_image": image,
            "annotated_image": annotated_image,
            "latency_ms": round(latency, 1),
            "fps": round(1000.0 / max(1.0, latency), 1)
        }
