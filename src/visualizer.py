import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class PlateVisualizer:
    """
    Module trực quan hóa kết quả:
    - Vẽ bounding box
    - Vẽ text có dấu tiếng Việt (Hà Nội, TP. Hồ Chí Minh...) bằng PIL
    - Vẽ bảng thông số tóm tắt
    """

    @staticmethod
    def draw_results(image: np.ndarray, detected_plates: list[dict]) -> np.ndarray:
        """
        Vẽ tất cả các biển số được nhận diện lên ảnh gốc.
        detected_plates là danh sách kết quả từ Pipeline.
        """
        if image is None or image.size == 0:
            return image

        annotated = image.copy()

        for item in detected_plates:
            bbox = item.get("bbox", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            plate_info = item.get("plate_info", {})
            formatted_plate = plate_info.get("formatted", "Unknown")
            province = plate_info.get("province", "")
            is_valid = plate_info.get("is_valid", False)

            # Màu viền: Xanh lá nếu hợp lệ, Vàng cam nếu chưa rõ
            color = (0, 200, 0) if is_valid else (0, 165, 255)

            # 1. Vẽ khung bo viền
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # 2. Tạo nhãn text
            label_text = f"{formatted_plate}"
            if province and province != "Không rõ":
                label_text += f" ({province})"

            # 3. Vẽ nhãn bằng PIL để hỗ trợ tiếng Việt có dấu
            annotated = PlateVisualizer._draw_vietnamese_text(
                annotated,
                label_text,
                (x1, max(0, y1 - 32)),
                bg_color=color,
                text_color=(255, 255, 255)
            )

        return annotated

    @staticmethod
    def _draw_vietnamese_text(
        image: np.ndarray,
        text: str,
        pos: tuple[int, int],
        bg_color: tuple[int, int, int] = (0, 200, 0),
        text_color: tuple[int, int, int] = (255, 255, 255),
        font_size: int = 18
    ) -> np.ndarray:
        """
        Vẽ văn bản tiếng Việt Unicode lên ảnh OpenCV bằng PIL ImageDraw.
        """
        # OpenCV BGR -> PIL RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Sử dụng font hệ thống Windows (arial.ttf, segoeui.ttf, tahoma.ttf)
        font = None
        font_names = ["arial.ttf", "segoeui.ttf", "tahoma.ttf"]
        for f_name in font_names:
            try:
                font = ImageFont.truetype(f_name, font_size)
                break
            except Exception:
                continue

        if font is None:
            font = ImageFont.load_default()

        x, y = pos
        # Lấy kích thước hộp chữ
        bbox = draw.textbbox((x, y), text, font=font)
        pad = 4
        rect = [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad]

        # Đổi màu BGR sang RGB cho PIL
        bg_rgb = (bg_color[2], bg_color[1], bg_color[0])
        txt_rgb = (text_color[2], text_color[1], text_color[0])

        draw.rectangle(rect, fill=bg_rgb)
        draw.text((x, y), text, font=font, fill=txt_rgb)

        # PIL RGB -> OpenCV BGR
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
