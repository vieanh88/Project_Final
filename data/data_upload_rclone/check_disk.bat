@echo off
REM ============================================================================
REM  check_disk.bat — Kiểm tra dung lượng ổ + tìm rác từ các lần thử upload
REM ============================================================================
REM  Chạy: double-click hoặc trong CMD:
REM     check_disk.bat > C:\check_disk_output.txt
REM ============================================================================

echo ============================================================
echo   DUNG LUONG TUNG O
echo ============================================================
wmic logicaldisk get caption,description,freespace,size,volumename /format:table

echo.
echo ============================================================
echo   FILE .TAR LON (co the la rac tu cac lan tar truoc)
echo ============================================================
for %%D in (C D E F G) do (
    if exist %%D:\ (
        echo --- O %%D: ---
        dir %%D:\*.tar /s /b 2>nul
    )
)

echo.
echo ============================================================
echo   FOLDER "upload*" tren cac o (rac tu cac lan thu)
echo ============================================================
for %%D in (C D E F G) do (
    if exist %%D:\ (
        echo --- O %%D: ---
        dir %%D:\*upload* /AD /b 2>nul
    )
)

echo.
echo ============================================================
echo   GOOGLE DRIVE DESKTOP CACHE
echo ============================================================
if exist "%LOCALAPPDATA%\Google\DriveFS\" (
    echo Folder: %LOCALAPPDATA%\Google\DriveFS\
    dir "%LOCALAPPDATA%\Google\DriveFS" /s 2>nul | findstr "Dir(s)\|File(s)" | findstr "free\|bytes"
)

echo.
echo ============================================================
echo   ONEDRIVE FOLDER
echo ============================================================
if exist "%USERPROFILE%\OneDrive\" (
    echo Folder: %USERPROFILE%\OneDrive\
    dir "%USERPROFILE%\OneDrive" /s 2>nul | findstr "Dir(s)\|File(s)" | findstr "free\|bytes" | tail
)

echo.
echo ============================================================
echo   TEMP FOLDER
echo ============================================================
echo Folder: %TEMP%
dir "%TEMP%" 2>nul | findstr "File(s)"

echo.
echo ============================================================
echo   FOLDER LON NHAT TREN O D (top 10)
echo ============================================================
powershell -Command "Get-ChildItem D:\ -Directory -Force -ErrorAction SilentlyContinue | ForEach-Object { $size = (Get-ChildItem $_.FullName -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB; [PSCustomObject]@{ Folder = $_.FullName; SizeGB = [math]::Round($size, 2) } } | Sort-Object SizeGB -Descending | Select-Object -First 10 | Format-Table -AutoSize"

echo.
echo ============================================================
echo   FOLDER LON NHAT TREN O C (top 10)
echo ============================================================
powershell -Command "Get-ChildItem C:\ -Directory -Force -ErrorAction SilentlyContinue | ForEach-Object { $size = (Get-ChildItem $_.FullName -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB; [PSCustomObject]@{ Folder = $_.FullName; SizeGB = [math]::Round($size, 2) } } | Sort-Object SizeGB -Descending | Select-Object -First 10 | Format-Table -AutoSize"

echo.
echo ============================================================
echo   KICH THUOC DATA vivoice_clean_wavs (data goc - KHONG xoa!)
echo ============================================================
if exist "D:\Documents\HUST\HUST_Project\Project_Final\TTS_StyleTTS2\fine-tune\data_pipeline\prepare_vivoice\output\vivoice_clean_wavs" (
    powershell -Command "$f = 'D:\Documents\HUST\HUST_Project\Project_Final\TTS_StyleTTS2\fine-tune\data_pipeline\prepare_vivoice\output\vivoice_clean_wavs'; $size = (Get-ChildItem $f -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB; $count = (Get-ChildItem $f -Recurse -File -Force -ErrorAction SilentlyContinue).Count; Write-Output ('Folder: ' + $f); Write-Output ('Size  : ' + [math]::Round($size, 2) + ' GB'); Write-Output ('Files : ' + $count)"
)

echo.
echo ============================================================
echo   THUNG RAC (Recycle Bin) - co the chua nhieu rac
echo ============================================================
powershell -Command "$shell = New-Object -ComObject Shell.Application; $recycle = $shell.NameSpace(0xA); $size = 0; $count = 0; $recycle.Items() | ForEach-Object { $size += $_.Size; $count++ }; Write-Output ('Items : ' + $count); Write-Output ('Size  : ' + [math]::Round($size / 1GB, 2) + ' GB')"

echo.
echo ============================================================
echo   HOAN TAT - paste output nay cho Claude
echo ============================================================
pause
