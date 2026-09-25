@echo off
title Auto-Run OQS Quantum Cognition
color 0E

echo ========================================================
echo       HE THONG TU DONG TINH TOAN VA XUAT KET QUA
echo                 (OQS Auto Export)
echo ========================================================
echo.
echo Dang quet du lieu tho va chay hang loat thuat toan...
echo Xin vui long cho doi...
echo.

cd /d "%~dp0\GUI_App"
python auto_export.py

echo.
echo ========================================================
echo Xong! Bam phim bat ky de mo thu muc chua ket qua...
pause >nul
explorer "..\results"
