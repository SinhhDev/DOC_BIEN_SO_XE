@echo off
chcp 65001 >nul
cd /d "%~dp0"
title HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE

echo =======================================================
echo   HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE VIỆT NAM
echo   YOLOv8 + EasyOCR + OpenCV + Tkinter GUI
echo =======================================================
echo.

set PYTHONIOENCODING=utf-8

:: 1. Kiem tra va tu dong khoi tao moi truong neu chua co
if not exist ".venv\Scripts\python.exe" (
    echo [LẦN ĐẦU CHẠY] Chưa tìm thấy môi trường ảo .venv.
    echo Đang tự động thiết lập môi trường...
    echo.
    python -m venv .venv
    if errorlevel 1 (
        echo [LỖI] Không thể tạo môi trường ảo. Vui lòng kiểm tra Python đã được cài đặt và thêm vào PATH!
        pause
        exit /b 1
    )
    echo [1/2] Đã tạo môi trường ảo .venv thành công.
    echo [2/2] Đang cài đặt các thư viện từ requirements.txt...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [LỖI] Cài đặt thư viện thất bại. Vui lòng kiểm tra kết nối Internet!
        pause
        exit /b 1
    )
    echo.
    echo =======================================================
    echo    THIẾT LẬP THÀNH CÔNG! ĐANG KHỞI CHẠY PHẦN MỀM...
    echo =======================================================
    echo.
)

:: 2. Khoi chay ung dung GUI
echo Đang mở giao diện người dùng...
".venv\Scripts\python.exe" "src\main.py" --gui
if errorlevel 1 (
    echo.
    echo [LỖI] Gặp sự cố khi khởi chạy. Vui lòng kiểm tra log lỗi ở trên!
    pause
)
