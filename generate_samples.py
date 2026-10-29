import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_vietnamese_plate(
    plate_text_lines: list[str],
    plate_type: str = "long",  # "long" hoặc "square"
    width: int = 400,
    height: int = 120
) -> np.ndarray:
    """Tạo ảnh biển số xe Việt Nam chân thực với viền bo góc."""
    # Nền trắng phản quang hơi ngả sáng
    img = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    # Viền ngoài màu đen
    border_width = max(3, int(height * 0.04))
    draw.rounded_rectangle([4, 4, width - 4, height - 4], radius=10, outline=(20, 20, 20), width=border_width)

    # Nạp font chữ số đậm nét
    font_names = ["arialbd.ttf", "arial.ttf", "segoeuib.ttf", "tahomabd.ttf"]
    font = None

    if plate_type == "long":
        # 1 dòng chữ: VD "30A - 888.88"
        font_size = int(height * 0.55)
        for fn in font_names:
            try:
                font = ImageFont.truetype(fn, font_size)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()

        text = plate_text_lines[0]
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        x = (width - text_w) // 2
        y = (height - text_h) // 2 - 4
        draw.text((x, y), text, font=font, fill=(15, 15, 15))

    else:
        # 2 dòng chữ (xe máy hoặc ô tô vuông):
        # Dòng 1: "59-F1"
        # Dòng 2: "123.45"
        font_size = int(height * 0.38)
        for fn in font_names:
            try:
                font = ImageFont.truetype(fn, font_size)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()

        # Dòng trên
        t1 = plate_text_lines[0]
        bbox1 = draw.textbbox((0, 0), t1, font=font)
        x1 = (width - (bbox1[2] - bbox1[0])) // 2
        y1 = int(height * 0.12)
        draw.text((x1, y1), t1, font=font, fill=(15, 15, 15))

        # Dòng dưới
        t2 = plate_text_lines[1]
        bbox2 = draw.textbbox((0, 0), t2, font=font)
        x2 = (width - (bbox2[2] - bbox2[0])) // 2
        y2 = int(height * 0.52)
        draw.text((x2, y2), t2, font=font, fill=(15, 15, 15))

    # Chuyển sang OpenCV BGR
    plate_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return plate_cv


def embed_plate_in_vehicle_scene(
    plate_img: np.ndarray,
    scene_width: int = 800,
    scene_height: int = 600,
    plate_pos: tuple[int, int] = (250, 420),
    vehicle_color: tuple[int, int, int] = (45, 45, 50),
    is_car: bool = True
) -> np.ndarray:
    """Tạo một khung cảnh đuôi xe hoặc đầu xe với biển số gắn trên cản xe."""
    scene = np.full((scene_height, scene_width, 3), 180, dtype=np.uint8)

    # Vẽ nền đường nhựa / gara
    scene[int(scene_height * 0.65):, :] = (70, 75, 80)

    # Vẽ thân xe
    car_top = int(scene_height * 0.20)
    car_bottom = int(scene_height * 0.75)
    car_left = int(scene_width * 0.12)
    car_right = int(scene_width * 0.88)

    # Đổ bóng và thân xe bo tròn
    cv2.rectangle(scene, (car_left, car_top), (car_right, car_bottom), vehicle_color, -1)

    if is_car:
        # Đèn hậu 2 bên xe ô tô
        cv2.rectangle(scene, (car_left + 20, car_top + 60), (car_left + 120, car_top + 110), (30, 30, 200), -1)
        cv2.rectangle(scene, (car_right - 120, car_top + 60), (car_right - 20, car_top + 110), (30, 30, 200), -1)
        # Cản xe (Bumper)
        cv2.rectangle(scene, (car_left + 40, car_bottom - 130), (car_right - 40, car_bottom - 20), (30, 30, 35), -1)
        # Kính sau
        cv2.rectangle(scene, (car_left + 60, car_top + 20), (car_right - 60, car_top + 100), (40, 50, 60), -1)
    else:
        # Đèn hậu xe máy
        mid_x = scene_width // 2
        cv2.circle(scene, (mid_x, car_top + 120), 45, (20, 20, 220), -1)
        # Ốp chắn bùn sau
        cv2.rectangle(scene, (mid_x - 120, car_top + 160), (mid_x + 120, car_bottom - 20), (25, 25, 30), -1)

    # Dán biển số vào vị trí
    pw, ph = plate_img.shape[1], plate_img.shape[0]
    px, py = plate_pos

    # Đảm bảo nằm gọn trong ảnh
    px = max(10, min(scene_width - pw - 10, px))
    py = max(10, min(scene_height - ph - 10, py))

    scene[py:py + ph, px:px + pw] = plate_img

    # Thêm một chút nhiễu nhẹ như chụp ảnh thực tế
    noise = np.random.normal(0, 3, scene.shape).astype(np.int16)
    scene = np.clip(scene.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return scene


def generate_all_samples():
    os.makedirs("samples", exist_ok=True)

    print("[1/4] Đang tạo mẫu ô tô Hà Nội (30A-888.88)...")
    plate1 = create_vietnamese_plate(["30A - 888.88"], plate_type="long", width=280, height=80)
    scene1 = embed_plate_in_vehicle_scene(plate1, plate_pos=(260, 430), vehicle_color=(40, 40, 45), is_car=True)
    cv2.imwrite("samples/sample_car_hanoi.jpg", scene1)

    print("[2/4] Đang tạo mẫu ô tô TP.HCM (51K-678.90)...")
    plate2 = create_vietnamese_plate(["51K - 678.90"], plate_type="long", width=290, height=82)
    scene2 = embed_plate_in_vehicle_scene(plate2, plate_pos=(255, 425), vehicle_color=(120, 30, 30), is_car=True)
    cv2.imwrite("samples/sample_car_hcm.jpg", scene2)

    print("[3/4] Đang tạo mẫu xe máy TP.HCM (59-F1 123.45)...")
    plate3 = create_vietnamese_plate(["59-F1", "123.45"], plate_type="square", width=210, height=150)
    scene3 = embed_plate_in_vehicle_scene(plate3, plate_pos=(295, 360), vehicle_color=(25, 25, 25), is_car=False)
    cv2.imwrite("samples/sample_bike_hcm.jpg", scene3)

    print("[4/4] Đang tạo mẫu xe máy Đà Nẵng (43-B1 567.89)...")
    plate4 = create_vietnamese_plate(["43-B1", "567.89"], plate_type="square", width=210, height=150)
    scene4 = embed_plate_in_vehicle_scene(plate4, plate_pos=(295, 360), vehicle_color=(35, 60, 100), is_car=False)
    cv2.imwrite("samples/sample_bike_danang.jpg", scene4)

    print("=> Hoàn tất tạo 4 ảnh mẫu trong thư mục samples/!")


if __name__ == "__main__":
    generate_all_samples()
