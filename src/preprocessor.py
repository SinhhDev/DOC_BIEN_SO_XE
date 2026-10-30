import cv2
import numpy as np


class PlatePreprocessor:
    """
    Module tiền xử lý ảnh vùng biển số xe:
    - Cắt vùng (crop) có padding
    - Xoay thẳng ảnh bị nghiêng (deskew)
    - Tăng độ tương phản (CLAHE)
    - Làm mịn khử nhiễu (Bilateral Filter)
    - Nhị phân hóa (Adaptive / Otsu Threshold)
    """

    @staticmethod
    def crop_with_padding(image: np.ndarray, bbox, pad_ratio: float = 0.03) -> np.ndarray:
        """
        Cắt vùng biển số từ ảnh gốc và thêm padding an toàn.
        bbox dạng (x1, y1, x2, y2).
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        bw = x2 - x1
        bh = y2 - y1

        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)

        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(w, x2 + pad_x)
        y2_pad = min(h, y2 + pad_y)

        cropped = image[y1_pad:y2_pad, x1_pad:x2_pad]
        return cropped

    @staticmethod
    def deskew_plate(plate_img: np.ndarray) -> np.ndarray:
        """
        Tự động cân chỉnh góc nghiêng của biển số xe bằng minAreaRect.
        """
        if plate_img is None or plate_img.size == 0:
            return plate_img

        try:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            # Dùng Canny để tìm viền
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 200)

            # Tìm contours
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return plate_img

            # Chọn contour có diện tích lớn nhất (khung biển)
            c = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            angle = rect[-1]

            # Chuẩn hóa góc trong khoảng [-45, 45]
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle

            # Nếu góc lệch quá lớn (> 25 độ), có thể do nhiễu, hạn chế xoay lệch quá đà
            if abs(angle) > 25 or abs(angle) < 0.8:
                return plate_img

            (h, w) = plate_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(plate_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        except Exception:
            return plate_img

    @staticmethod
    def enhance_for_ocr(plate_img: np.ndarray, target_height: int = 140) -> tuple[np.ndarray, np.ndarray]:
        """
        Tiền xử lý nâng cao cho OCR:
        - Scale kích thước để ký tự rõ ràng
        - Cân bằng ánh sáng (CLAHE)
        - Trả về: (ảnh màu đã xử lý, ảnh nhị phân đã làm rõ chữ)
        """
        if plate_img is None or plate_img.size == 0:
            return plate_img, plate_img

        h, w = plate_img.shape[:2]
        if h <= 0 or w <= 0:
            return plate_img, plate_img

        # Phóng to ảnh nếu quá nhỏ
        if h < target_height:
            scale = target_height / float(h)
            new_w = max(1, int(w * scale))
            plate_img = cv2.resize(plate_img, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # Áp dụng CLAHE để cải thiện độ tương phản cục bộ
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_gray = clahe.apply(gray)

        # Khử nhiễu nhưng vẫn giữ cạnh ký tự sắc nét
        denoised = cv2.bilateralFilter(contrast_gray, 9, 75, 75)

        # Nhị phân hóa Otsu
        _, thresh_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Tự động phát hiện xem chữ là đen nền trắng hay chữ trắng nền đen
        # Hầu hết biển số VN là chữ đen nền trắng (số pixel trắng > pixel đen)
        white_pixel_ratio = np.mean(thresh_otsu == 255)
        if white_pixel_ratio < 0.4:
            # Nghịch đảo màu nếu nền tối chữ sáng
            thresh_otsu = cv2.bitwise_not(thresh_otsu)

        # Chuyển ảnh nhị phân về dạng 3 kênh BGR để tương thích EasyOCR
        thresh_bgr = cv2.cvtColor(thresh_otsu, cv2.COLOR_GRAY2BGR)

        return plate_img, thresh_bgr
