@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ==========================================================
echo    PHAN MEM DANH GIA SU LUONG LU - OQS ^& CTMC
echo ==========================================================

REM Tim Python trong PATH
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)

if not defined PY (
    echo [LOI] Khong tim thay Python tren may nay.
    pause
    exit /b
)

echo Dang mo giao dien...
%PY% GUI_App\main.py

pause
