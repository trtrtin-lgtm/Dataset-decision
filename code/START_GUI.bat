@echo off
title OQS Quantum Cognition - System Control
color 0A

echo ========================================================
echo       HE THONG PHAN TICH NHAN THUC LUONG TU (OQS)
echo                (Quantum Cognition App)
echo ========================================================
echo.
echo Dang khoi dong Giao dien Nguoi dung (GUI)...
echo Vui long doi trong giay lat...

:: Chuyển thư mục hiện tại vào GUI_App (dù file bat được gọi từ đâu)
cd /d "%~dp0\GUI_App"

:: Khởi chạy main.py
python main.py

:: Nếu phần mềm bị tắt do lỗi, cửa sổ sẽ dừng lại để đọc lỗi
pause
