import easyocr
import numpy as np
from src.postprocessor import PlatePostprocessor


class OCREngine:
    """
    Engine OCR nhận dạng ký tự biển số xe bằng EasyOCR:
    - Hạn chế ký tự nhận diện chỉ trong [A-Z0-9]
    - Tự động gom nhóm các khối ký tự theo hàng (dòng trên / dòng dưới)
    - Hỗ trợ thử cả ảnh tăng cường và ảnh nhị phân để lấy kết quả tối ưu
    """

    def __init__(self, gpu: bool = False):
        # verbose=False để tránh lỗi Unicode Progress bar trên Windows cp1252
        self.reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)

    def read_plate(self, plate_cropped: np.ndarray, plate_enhanced: np.ndarray, plate_thresh: np.ndarray) -> dict:
        """
        Nhận diện biển số xe từ ảnh biển số đã xử lý:
        Thử nghiệm trên cả 3 biến thể:
        1. Ảnh gốc crop
        2. Ảnh tăng cường tương phản (CLAHE)
        3. Ảnh nhị phân hóa (Otsu Threshold)
        """
        res_crop = self._run_easyocr(plate_cropped)
        res_enh = self._run_easyocr(plate_enhanced)
        res_thr = self._run_easyocr(plate_thresh)

        # Chọn kết quả tối ưu nhất trong 3 kết quả
        candidates = [res_crop, res_enh, res_thr]
        chosen = self._select_best_from_list(candidates)

        # Hậu xử lý văn bản theo quy chuẩn biển số Việt Nam
        formatted_info = PlatePostprocessor.format_plate(chosen["lines"])
        formatted_info["ocr_confidence"] = chosen["confidence"]
        formatted_info["raw_ocr_lines"] = chosen["lines"]

        return formatted_info

    def _select_best_from_list(self, results: list[dict]) -> dict:
        """Chọn kết quả hợp lý nhất: độ dài ký tự chuẩn (7-9 ký tự) và độ tin cậy cao."""
        valid_candidates = []
        for r in results:
            total_chars = sum(len(PlatePostprocessor.clean_raw_text(l)) for l in r["lines"])
            if 7 <= total_chars <= 9:
                valid_candidates.append((r, total_chars, r["confidence"]))

        if valid_candidates:
            # Sắp xếp theo confidence cao nhất
            valid_candidates.sort(key=lambda x: x[2], reverse=True)
            return valid_candidates[0][0]

        # Nếu không có ứng viên nào rơi vào 7-9 ký tự, chọn ứng viên có conf cao nhất mà có ít nhất 5 ký tự
        fallback = [r for r in results if sum(len(l) for l in r["lines"]) >= 5]
        if fallback:
            fallback.sort(key=lambda x: x["confidence"], reverse=True)
            return fallback[0]

        # Cuối cùng trả về kết quả có conf cao nhất
        return max(results, key=lambda x: x["confidence"])

    def _run_easyocr(self, image: np.ndarray) -> dict:
        """
        Chạy EasyOCR trên một ảnh và gom nhóm các dòng theo trục Y.
        """
        if image is None or image.size == 0:
            return {"lines": [], "confidence": 0.0}

        try:
            # allowlist chỉ nhận chữ cái in hoa và số
            results = self.reader.readtext(
                image,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                paragraph=False,
                detail=1
            )
        except Exception as e:
            return {"lines": [], "confidence": 0.0}

        if not results:
            return {"lines": [], "confidence": 0.0}

        # Sắp xếp các box theo vị trí trục Y để phân tách dòng trên / dòng dưới
        # result item: (bbox, text, conf)
        # bbox: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        boxes_info = []
        for bbox, text, conf in results:
            text_clean = text.strip()
            if not text_clean:
                continue
            y_center = (bbox[0][1] + bbox[2][1]) / 2.0
            x_min = bbox[0][0]
            boxes_info.append({
                "bbox": bbox,
                "text": text_clean,
                "conf": conf,
                "y_center": y_center,
                "x_min": x_min
            })

        if not boxes_info:
            return {"lines": [], "confidence": 0.0}

        # Xác định xem có chia làm 2 dòng không dựa trên khoảng cách Y
        h = image.shape[0]
        y_threshold = h * 0.45

        # Nhóm thành các dòng
        top_line = []
        bottom_line = []

        # Kiểm tra sự chênh lệch độ cao giữa các box
        y_values = [b["y_center"] for b in boxes_info]
        y_span = max(y_values) - min(y_values)

        if y_span > h * 0.25:
            # Biển 2 dòng
            for b in boxes_info:
                if b["y_center"] < y_threshold:
                    top_line.append(b)
                else:
                    bottom_line.append(b)

            # Sắp xếp từ trái sang phải trong mỗi dòng
            top_line.sort(key=lambda x: x["x_min"])
            bottom_line.sort(key=lambda x: x["x_min"])

            lines = []
            if top_line:
                lines.append("".join([b["text"] for b in top_line]))
            if bottom_line:
                lines.append("".join([b["text"] for b in bottom_line]))
        else:
            # Biển 1 dòng
            boxes_info.sort(key=lambda x: x["x_min"])
            lines = ["".join([b["text"] for b in boxes_info])]

        avg_conf = float(np.mean([b["conf"] for b in boxes_info]))
        return {
            "lines": lines,
            "confidence": round(avg_conf, 3)
        }

    def _select_best_result(self, res1: dict, res2: dict) -> dict:
        """Chọn kết quả tốt nhất giữa ảnh tăng tương phản và ảnh nhị phân."""
        len1 = sum(len(l) for l in res1["lines"])
        len2 = sum(len(l) for l in res2["lines"])

        # Nếu một bên có độ dài ký tự hợp lý (từ 7 đến 10 ký tự) thì ưu tiên
        is_len1_good = (7 <= len1 <= 10)
        is_len2_good = (7 <= len2 <= 10)

        if is_len1_good and not is_len2_good:
            return res1
        elif is_len2_good and not is_len1_good:
            return res2

        # Nếu cả 2 đều hợp lý hoặc đều không hợp lý, so sánh confidence
        if res1["confidence"] >= res2["confidence"]:
            return res1
        return res2

