import os
import sys
import argparse
import time
import cv2
from pathlib import Path
from src.pipeline import ALPRPipeline
from src.config import OUTPUT_DIR, SAMPLES_DIR


def run_image(pipeline: ALPRPipeline, image_path: str, show: bool = False, save: bool = True):
    print(f"\n[INFO] Đang xử lý ảnh: {image_path}")
    if not os.path.exists(image_path):
        print(f"[ERROR] Không tìm thấy file: {image_path}")
        return

    result = pipeline.process_image(image_path)
    num = result["num_plates"]
    print(f"[KẾT QUẢ] Số biển phát hiện: {num} (Thời gian xử lý: {result['latency_ms']} ms)")

    for idx, p in enumerate(result["plates"], 1):
        info = p["plate_info"]
        print(f"  + Biển #{idx}:")
        print(f"    - Biển số:     {info.get('formatted', 'UNKNOWN')}")
        print(f"    - Loại xe:     {info.get('vehicle_type', 'UNKNOWN')}")
        print(f"    - Tỉnh thành:  {info.get('province', 'UNKNOWN')}")
        print(f"    - Độ tin cậy:  {info.get('ocr_confidence', 0.0) * 100:.1f}%")
        print(f"    - Bounding Box: {p['bbox']}")

    if save:
        filename = Path(image_path).name
        out_path = os.path.join(OUTPUT_DIR, f"result_{filename}")
        cv2.imwrite(out_path, result["annotated_image"])
        print(f"[LƯU ẢNH] Đã lưu kết quả tại: {out_path}")

    if show:
        cv2.imshow("Ket Qua Nhan Dien Bien So", result["annotated_image"])
        print("[INFO] Bấm phím bất kỳ trên cửa sổ ảnh để đóng...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_directory(pipeline: ALPRPipeline, dir_path: str):
    print(f"\n[INFO] Đang quét thư mục: {dir_path}")
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(exts)]

    if not files:
        print(f"[WARNING] Không tìm thấy ảnh hợp lệ trong: {dir_path}")
        return

    for f in files:
        run_image(pipeline, f, show=False, save=True)


def run_webcam(pipeline: ALPRPipeline, cam_id: int = 0):
    print(f"\n[INFO] Đang mở Camera ID {cam_id}... (Nhấn 'q' để thoát)")
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Không thể kết nối tới Webcam {cam_id}")
        return

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Để webcam mượt hơn, có thể giảm kích thước khung hình nếu cần
        h, w = frame.shape[:2]
        if w > 1280:
            frame = cv2.resize(frame, (1280, int(h * 1280 / w)))

        result = pipeline.process_image(frame)
        annotated = result["annotated_image"]

        # Tính FPS thực tế
        curr_time = time.time()
        fps = 1.0 / max(0.001, curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f} | Latency: {result['latency_ms']:.0f}ms",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Nhan Dien Bien So Xe - Webcam (Bam 'q' de thoat)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_video(pipeline: ALPRPipeline, video_path: str, save_output: bool = True):
    print(f"\n[INFO] Đang xử lý video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Không thể đọc video từ: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    if save_output:
        out_name = f"result_{Path(video_path).stem}.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"[INFO] Video kết quả sẽ được lưu tại: {out_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        result = pipeline.process_image(frame)
        annotated = result["annotated_image"]

        if writer:
            writer.write(annotated)

        cv2.imshow("Dang xu ly Video (Bam 'q' de dung)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Đã xử lý xong {frame_count} frames!")


def main():
    parser = argparse.ArgumentParser(description="Hệ thống Nhận Diện Biển Số Xe Việt Nam (ALPR)")
    parser.add_argument("--image", "-i", type=str, help="Đường dẫn tới file ảnh cần nhận diện")
    parser.add_argument("--dir", "-d", type=str, help="Đường dẫn tới thư mục ảnh cần nhận diện hàng loạt")
    parser.add_argument("--webcam", "-w", action="store_true", help="Chạy nhận diện thời gian thực qua Webcam")
    parser.add_argument("--cam-id", type=int, default=0, help="ID camera (mặc định: 0)")
    parser.add_argument("--video", "-v", type=str, help="Đường dẫn tới file video")
    parser.add_argument("--gui", "-g", action="store_true", help="Mở giao diện người dùng đồ họa (GUI)")
    parser.add_argument("--gpu", action="store_true", help="Kích hoạt GPU (CUDA) nếu có")
    parser.add_argument("--show", action="store_true", help="Hiển thị cửa sổ ảnh kết quả")

    args = parser.parse_args()

    # Nếu người dùng muốn mở GUI hoặc không truyền tham số nào
    if args.gui or len(sys.argv) == 1:
        from app_gui import launch_gui
        launch_gui()
        return

    # Khởi tạo pipeline
    pipeline = ALPRPipeline(use_gpu=args.gpu)

    if args.image:
        run_image(pipeline, args.image, show=args.show)
    elif args.dir:
        run_directory(pipeline, args.dir)
    elif args.webcam:
        run_webcam(pipeline, cam_id=args.cam_id)
    elif args.video:
        run_video(pipeline, args.video)
    else:
        print("[INFO] Không có tác vụ được chỉ định. Đang khởi chạy GUI...")
        from app_gui import launch_gui
        launch_gui()


if __name__ == "__main__":
    main()
