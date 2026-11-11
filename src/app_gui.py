import os
import sys
import time
import threading
import datetime
import csv
from pathlib import Path

# Đảm bảo thư mục gốc dự án luôn nằm trong sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

from src.pipeline import ALPRPipeline
from src.config import SAMPLES_DIR, OUTPUT_DIR


class LicensePlateGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hệ Thống Nhận Diện Biển Số Xe Việt Nam (ALPR)")
        self.root.geometry("1260x820")
        self.root.minsize(1050, 700)

        # Trạng thái hệ thống
        self.pipeline = None
        self.is_webcam_running = False
        self.cap = None
        self.current_image = None
        self.current_result = None
        self.history_records = []

        # Cấu hình Theme & Màu sắc hiện đại
        self.colors = {
            "bg_dark": "#1e1e2e",
            "bg_panel": "#252538",
            "bg_card": "#2e2e44",
            "primary": "#4f46e5",
            "primary_hover": "#4338ca",
            "success": "#10b981",
            "warning": "#f59e0b",
            "danger": "#ef4444",
            "text": "#f8fafc",
            "text_muted": "#94a3b8",
            "border": "#3b3b54",
            "accent": "#06b6d4"
        }

        self.root.configure(bg=self.colors["bg_dark"])
        self._setup_styles()
        self._build_ui()

        # Nạp mô hình trong luồng nền để mở app ngay lập tức
        self.status_var.set("Đang khởi tạo các mô hình AI...")
        threading.Thread(target=self._init_pipeline, daemon=True).start()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Cấu hình Treeview (Bảng lịch sử)
        style.configure(
            "Treeview",
            background=self.colors["bg_card"],
            foreground=self.colors["text"],
            fieldbackground=self.colors["bg_card"],
            rowheight=28,
            font=("Segoe UI", 10)
        )
        style.configure(
            "Treeview.Heading",
            background=self.colors["bg_panel"],
            foreground=self.colors["text"],
            font=("Segoe UI", 10, "bold")
        )
        style.map("Treeview", background=[("selected", self.colors["primary"])])

    def _init_pipeline(self):
        try:
            self.pipeline = ALPRPipeline(use_gpu=False)
            self.root.after(0, lambda: self.status_var.set("Sẵn sàng! Hãy chọn ảnh hoặc mở Webcam."))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Lỗi khởi tạo AI: {e}"))

    def _build_ui(self):
        # 1. HEADER
        header_frame = tk.Frame(self.root, bg=self.colors["bg_panel"], height=65, padx=20, pady=10)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header_frame,
            text="HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE THÔNG MINH",
            font=("Segoe UI", 16, "bold"),
            fg=self.colors["text"],
            bg=self.colors["bg_panel"]
        )
        title_lbl.pack(side=tk.LEFT)

        subtitle_lbl = tk.Label(
            header_frame,
            text="Vietnam ALPR • YOLOv8 + EasyOCR",
            font=("Segoe UI", 10),
            fg=self.colors["accent"],
            bg=self.colors["bg_panel"]
        )
        subtitle_lbl.pack(side=tk.LEFT, padx=15, pady=4)

        # 2. TOOLBAR
        toolbar = tk.Frame(self.root, bg=self.colors["bg_card"], padx=15, pady=10)
        toolbar.pack(fill=tk.X, side=tk.TOP, pady=(2, 0))

        btn_font = ("Segoe UI", 10, "bold")

        btn_open = tk.Button(
            toolbar, text="📁 Chọn Ảnh", font=btn_font,
            bg=self.colors["primary"], fg="white", activebackground=self.colors["primary_hover"],
            activeforeground="white", relief=tk.FLAT, padx=15, pady=6, cursor="hand2",
            command=self.open_image_dialog
        )
        btn_open.pack(side=tk.LEFT, padx=5)

        btn_sample = tk.Button(
            toolbar, text="🚗 Ảnh Mẫu (Sample)", font=btn_font,
            bg="#374151", fg="white", activebackground="#4b5563",
            activeforeground="white", relief=tk.FLAT, padx=12, pady=6, cursor="hand2",
            command=self.open_sample_dialog
        )
        btn_sample.pack(side=tk.LEFT, padx=5)

        self.btn_webcam = tk.Button(
            toolbar, text="📷 Bật Webcam", font=btn_font,
            bg=self.colors["success"], fg="white", activebackground="#059669",
            activeforeground="white", relief=tk.FLAT, padx=15, pady=6, cursor="hand2",
            command=self.toggle_webcam
        )
        self.btn_webcam.pack(side=tk.LEFT, padx=5)

        btn_video = tk.Button(
            toolbar, text="🎬 Mở Video", font=btn_font,
            bg="#374151", fg="white", activebackground="#4b5563",
            activeforeground="white", relief=tk.FLAT, padx=12, pady=6, cursor="hand2",
            command=self.open_video_dialog
        )
        btn_video.pack(side=tk.LEFT, padx=5)

        btn_export = tk.Button(
            toolbar, text="📊 Xuất CSV Lịch Sử", font=btn_font,
            bg="#0284c7", fg="white", activebackground="#0369a1",
            activeforeground="white", relief=tk.FLAT, padx=12, pady=6, cursor="hand2",
            command=self.export_history_csv
        )
        btn_export.pack(side=tk.RIGHT, padx=5)

        btn_save_img = tk.Button(
            toolbar, text="💾 Lưu Ảnh Kết Quả", font=btn_font,
            bg="#475569", fg="white", activebackground="#334155",
            activeforeground="white", relief=tk.FLAT, padx=12, pady=6, cursor="hand2",
            command=self.save_current_result
        )
        btn_save_img.pack(side=tk.RIGHT, padx=5)

        # 3. MAIN CONTENT (Split View: Left Image Preview, Right Info Panel)
        content_frame = tk.Frame(self.root, bg=self.colors["bg_dark"], padx=15, pady=10)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left Column: Image / Video Display
        left_frame = tk.Frame(content_frame, bg=self.colors["bg_panel"], padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        lbl_display_title = tk.Label(
            left_frame, text="KHUNG HÌNH NHẬN DIỆN", font=("Segoe UI", 11, "bold"),
            fg=self.colors["text_muted"], bg=self.colors["bg_panel"]
        )
        lbl_display_title.pack(anchor="w", pady=(0, 5))

        self.canvas = tk.Canvas(left_frame, bg="#11111b", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right Column: Result Details & Crop View
        right_frame = tk.Frame(content_frame, bg=self.colors["bg_panel"], width=420, padx=15, pady=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)

        lbl_result_title = tk.Label(
            right_frame, text="KẾT QUẢ ĐỌC BIỂN SỐ", font=("Segoe UI", 11, "bold"),
            fg=self.colors["text_muted"], bg=self.colors["bg_panel"]
        )
        lbl_result_title.pack(anchor="w", pady=(0, 10))

        # Khối hiển thị Biển số lớn
        plate_box = tk.Frame(right_frame, bg=self.colors["bg_card"], padx=15, pady=15, relief=tk.FLAT)
        plate_box.pack(fill=tk.X, pady=(0, 10))

        self.lbl_plate_number = tk.Label(
            plate_box, text="-- CHƯA CÓ DỮ LIỆU --", font=("Segoe UI", 20, "bold"),
            fg=self.colors["accent"], bg=self.colors["bg_card"]
        )
        self.lbl_plate_number.pack(pady=5)

        # Khung thông tin chi tiết
        info_frame = tk.Frame(right_frame, bg=self.colors["bg_card"], padx=15, pady=12)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.lbl_province = self._add_info_row(info_frame, "Tỉnh/Thành phố:", "Chưa rõ")
        self.lbl_veh_type = self._add_info_row(info_frame, "Loại phương tiện:", "Chưa rõ")
        self.lbl_confidence = self._add_info_row(info_frame, "Độ tin cậy OCR:", "0.0%")
        self.lbl_latency = self._add_info_row(info_frame, "Thời gian xử lý:", "0 ms")

        # Vùng xem ảnh crop biển số và ảnh tiền xử lý
        crop_container = tk.Frame(right_frame, bg=self.colors["bg_card"], padx=10, pady=10)
        crop_container.pack(fill=tk.X, pady=(0, 10))

        lbl_crops_title = tk.Label(
            crop_container, text="VÙNG CẮT BIỂN SỐ (CROP & PREPROCESS)",
            font=("Segoe UI", 9, "bold"), fg=self.colors["text_muted"], bg=self.colors["bg_card"]
        )
        lbl_crops_title.pack(anchor="w", pady=(0, 5))

        crops_row = tk.Frame(crop_container, bg=self.colors["bg_card"])
        crops_row.pack(fill=tk.X)

        # Crop Image
        self.lbl_crop_img = tk.Label(crops_row, bg="#11111b", width=18, height=5, text="Vùng gốc", fg="#64748b")
        self.lbl_crop_img.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Thresh Image
        self.lbl_thresh_img = tk.Label(crops_row, bg="#11111b", width=18, height=5, text="Tiền xử lý", fg="#64748b")
        self.lbl_thresh_img.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # 4. BOTTOM: HISTORY TABLE
        bottom_frame = tk.Frame(self.root, bg=self.colors["bg_panel"], height=190, padx=15, pady=8)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=15, pady=(0, 10))

        lbl_hist_title = tk.Label(
            bottom_frame, text="LỊCH SỬ NHẬN DIỆN", font=("Segoe UI", 10, "bold"),
            fg=self.colors["text_muted"], bg=self.colors["bg_panel"]
        )
        lbl_hist_title.pack(anchor="w", pady=(0, 4))

        # Table Treeview
        columns = ("id", "time", "plate", "province", "type", "conf", "latency")
        self.tree = ttk.Treeview(bottom_frame, columns=columns, show="headings", height=5)
        self.tree.heading("id", text="#")
        self.tree.heading("time", text="Thời Gian")
        self.tree.heading("plate", text="Biển Số Xe")
        self.tree.heading("province", text="Tỉnh Thành")
        self.tree.heading("type", text="Loại Xe")
        self.tree.heading("conf", text="Độ Tin Cậy")
        self.tree.heading("latency", text="Độ Trễ")

        self.tree.column("id", width=40, anchor="center")
        self.tree.column("time", width=130, anchor="center")
        self.tree.column("plate", width=160, anchor="center")
        self.tree.column("province", width=180, anchor="w")
        self.tree.column("type", width=170, anchor="w")
        self.tree.column("conf", width=110, anchor="center")
        self.tree.column("latency", width=90, anchor="center")

        tree_scroll = ttk.Scrollbar(bottom_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 5. STATUS BAR
        status_bar = tk.Frame(self.root, bg="#11111b", height=26, padx=15)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_var = tk.StringVar(value="Đang sẵn sàng...")
        lbl_status = tk.Label(
            status_bar, textvariable=self.status_var, font=("Segoe UI", 9),
            fg=self.colors["text_muted"], bg="#11111b"
        )
        lbl_status.pack(side=tk.LEFT, pady=3)

    def _add_info_row(self, parent, label_text: str, default_val: str) -> tk.Label:
        row = tk.Frame(parent, bg=self.colors["bg_card"], pady=3)
        row.pack(fill=tk.X)

        lbl = tk.Label(
            row, text=label_text, font=("Segoe UI", 10),
            fg=self.colors["text_muted"], bg=self.colors["bg_card"], width=16, anchor="w"
        )
        lbl.pack(side=tk.LEFT)

        val_lbl = tk.Label(
            row, text=default_val, font=("Segoe UI", 10, "bold"),
            fg=self.colors["text"], bg=self.colors["bg_card"], anchor="w"
        )
        val_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return val_lbl

    # ================== CÁC CHỨC NĂNG XỬ LÝ ==================

    def open_image_dialog(self):
        if self.is_webcam_running:
            self.stop_webcam()

        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Chọn ảnh biển số xe", filetypes=filetypes)
        if path:
            self.process_single_image(path)

    def open_sample_dialog(self):
        if self.is_webcam_running:
            self.stop_webcam()

        sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg", ".png"))]
        if not sample_files:
            messagebox.showinfo("Thông báo", "Chưa có ảnh mẫu trong thư mục samples/.")
            return

        # Tạo popup chọn ảnh mẫu nhanh
        dlg = tk.Toplevel(self.root)
        dlg.title("Chọn ảnh mẫu")
        dlg.geometry("380x280")
        dlg.configure(bg=self.colors["bg_panel"])
        dlg.transient(self.root)
        dlg.grab_set()

        lbl = tk.Label(dlg, text="Danh sách ảnh mẫu có sẵn:", font=("Segoe UI", 10, "bold"),
                       fg=self.colors["text"], bg=self.colors["bg_panel"])
        lbl.pack(pady=10)

        listbox = tk.Listbox(dlg, bg=self.colors["bg_card"], fg=self.colors["text"], font=("Segoe UI", 10))
        listbox.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 10))

        for s in sample_files:
            listbox.insert(tk.END, s)

        def on_select():
            sel = listbox.curselection()
            if sel:
                chosen = sample_files[sel[0]]
                dlg.destroy()
                self.process_single_image(os.path.join(SAMPLES_DIR, chosen))

        btn = tk.Button(dlg, text="Xác nhận", font=("Segoe UI", 10, "bold"),
                        bg=self.colors["primary"], fg="white", command=on_select)
        btn.pack(pady=(0, 10))

    def process_single_image(self, image_path: str):
        if self.pipeline is None:
            messagebox.showwarning("Đang tải", "Mô hình AI đang khởi tạo, vui lòng chờ trong giây lát...")
            return

        self.status_var.set(f"Đang xử lý ảnh: {Path(image_path).name}...")

        def worker():
            try:
                res = self.pipeline.process_image(image_path)
                self.root.after(0, lambda: self._update_results_ui(res, source=Path(image_path).name))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Không thể xử lý ảnh: {e}"))
                self.root.after(0, lambda: self.status_var.set("Sẵn sàng"))

        threading.Thread(target=worker, daemon=True).start()

    def _update_results_ui(self, result: dict, source: str = "Image"):
        self.current_result = result
        annotated_bgr = result["annotated_image"]

        # 1. Hiển thị ảnh lớn lên Canvas
        self._display_on_canvas(annotated_bgr)

        # 2. Cập nhật bảng kết quả
        num = result["num_plates"]
        latency = result["latency_ms"]

        if num > 0:
            best_plate = result["plates"][0]
            info = best_plate["plate_info"]

            plate_num = info.get("formatted", "UNKNOWN")
            province = info.get("province", "Không rõ")
            veh_type = info.get("vehicle_type", "Không rõ")
            conf = info.get("ocr_confidence", 0.0) * 100

            self.lbl_plate_number.config(text=plate_num, fg="#22c55e" if info.get("is_valid") else self.colors["accent"])
            self.lbl_province.config(text=province)
            self.lbl_veh_type.config(text=veh_type)
            self.lbl_confidence.config(text=f"{conf:.1f}%")
            self.lbl_latency.config(text=f"{latency:.1f} ms ({result['fps']:.1f} FPS)")

            # Cập nhật crop ảnh biển số
            self._display_small_image(best_plate["cropped_img"], self.lbl_crop_img)
            self._display_small_image(best_plate["thresh_img"], self.lbl_thresh_img)

            # Thêm vào bảng lịch sử
            now_str = datetime.datetime.now().strftime("%H:%M:%S")
            rec_id = len(self.history_records) + 1
            item = (rec_id, now_str, plate_num, province, veh_type, f"{conf:.1f}%", f"{latency:.0f}ms")
            self.history_records.append({
                "id": rec_id, "time": now_str, "plate": plate_num, "province": province,
                "type": veh_type, "conf": f"{conf:.1f}%", "source": source
            })
            self.tree.insert("", 0, values=item)

            self.status_var.set(f"Thành công! Phát hiện {num} biển số trong {latency:.1f} ms.")
        else:
            self.lbl_plate_number.config(text="KHÔNG TÌM THẤY BIỂN SỐ", fg=self.colors["danger"])
            self.lbl_province.config(text="--")
            self.lbl_veh_type.config(text="--")
            self.lbl_confidence.config(text="0.0%")
            self.lbl_latency.config(text=f"{latency:.1f} ms")
            self.status_var.set("Không phát hiện được biển số xe nào trong ảnh.")

    def _display_on_canvas(self, bgr_image: np.ndarray):
        """Scale ảnh giữ nguyên tỉ lệ và vẽ vào canvas."""
        canvas_w = self.canvas.winfo_width() or 700
        canvas_h = self.canvas.winfo_height() or 500

        img_h, img_w = bgr_image.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        resized = cv2.resize(bgr_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        self.tk_canvas_img = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.tk_canvas_img)

    def _display_small_image(self, bgr_image: np.ndarray, label_widget: tk.Label):
        """Hiển thị ảnh crop nhỏ ở cột bên phải."""
        if bgr_image is None or bgr_image.size == 0:
            return
        h, w = bgr_image.shape[:2]
        target_w = 170
        scale = target_w / float(w)
        target_h = max(1, int(h * scale))

        resized = cv2.resize(bgr_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        label_widget.config(image=tk_img, text="")
        label_widget.image = tk_img

    # ================== WEBCAM REALTIME ==================

    def toggle_webcam(self):
        if self.is_webcam_running:
            self.stop_webcam()
        else:
            self.start_webcam()

    def start_webcam(self):
        if self.pipeline is None:
            messagebox.showwarning("Đang tải", "Vui lòng chờ AI khởi tạo xong...")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi Webcam", "Không thể kết nối tới Webcam (ID: 0)!")
            return

        self.is_webcam_running = True
        self.btn_webcam.config(text="⏹ Dừng Webcam", bg=self.colors["danger"])
        self.status_var.set("Webcam đang chạy...")

        threading.Thread(target=self._webcam_loop, daemon=True).start()

    def stop_webcam(self):
        self.is_webcam_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_webcam.config(text="📷 Bật Webcam", bg=self.colors["success"])
        self.status_var.set("Đã dừng Webcam.")

    def _webcam_loop(self):
        frame_interval = 2  # Nhận diện mỗi 2 khung hình để đảm bảo FPS cao
        count = 0
        last_result = None

        while self.is_webcam_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            count += 1
            if count % frame_interval == 0:
                try:
                    last_result = self.pipeline.process_image(frame)
                except Exception:
                    pass

            if last_result is not None:
                display_frame = last_result["annotated_image"]
                self.root.after(0, lambda r=last_result: self._update_webcam_ui(r))
            else:
                display_frame = frame

            self.root.after(0, lambda f=display_frame: self._display_on_canvas(f))
            time.sleep(0.02)

    def _update_webcam_ui(self, res: dict):
        if res["num_plates"] > 0:
            best = res["plates"][0]
            info = best["plate_info"]
            plate_num = info.get("formatted", "UNKNOWN")
            conf = info.get("ocr_confidence", 0.0) * 100

            self.lbl_plate_number.config(text=plate_num, fg="#22c55e" if info.get("is_valid") else self.colors["accent"])
            self.lbl_province.config(text=info.get("province", "Không rõ"))
            self.lbl_veh_type.config(text=info.get("vehicle_type", "Không rõ"))
            self.lbl_confidence.config(text=f"{conf:.1f}%")
            self.lbl_latency.config(text=f"{res['latency_ms']:.1f} ms")

            self._display_small_image(best["cropped_img"], self.lbl_crop_img)
            self._display_small_image(best["thresh_img"], self.lbl_thresh_img)

    # ================== VIDEO ==================

    def open_video_dialog(self):
        if self.is_webcam_running:
            self.stop_webcam()

        filetypes = [("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Chọn file Video", filetypes=filetypes)
        if not path:
            return

        messagebox.showinfo("Thông báo", "Đang mở cửa sổ xử lý Video. Bấm phím 'q' trên cửa sổ video để dừng.")
        from main import run_video
        threading.Thread(target=lambda: run_video(self.pipeline, path, save_output=True), daemon=True).start()

    # ================== LƯU & XUẤT DỮ LIỆU ==================

    def save_current_result(self):
        if self.current_result is None or self.current_result.get("annotated_image") is None:
            messagebox.showinfo("Thông báo", "Chưa có ảnh kết quả để lưu.")
            return

        default_name = f"result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        save_path = filedialog.asksaveasfilename(
            title="Lưu ảnh kết quả",
            initialdir=str(OUTPUT_DIR),
            initialfile=default_name,
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
        )
        if save_path:
            cv2.imwrite(save_path, self.current_result["annotated_image"])
            messagebox.showinfo("Thành công", f"Đã lưu ảnh tại:\n{save_path}")

    def export_history_csv(self):
        if not self.history_records:
            messagebox.showinfo("Thông báo", "Chưa có lịch sử nhận diện nào để xuất!")
            return

        default_name = f"license_plate_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        save_path = filedialog.asksaveasfilename(
            title="Xuất file CSV",
            initialdir=str(OUTPUT_DIR),
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv")]
        )
        if save_path:
            with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "time", "plate", "province", "type", "conf", "source"])
                writer.writeheader()
                writer.writerows(self.history_records)
            messagebox.showinfo("Thành công", f"Đã xuất thành công file CSV:\n{save_path}")


def launch_gui():
    root = tk.Tk()
    app = LicensePlateGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
