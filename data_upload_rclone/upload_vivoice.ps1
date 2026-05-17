# =============================================================================
# upload_vivoice.ps1 -- Stream tar + upload to Google Drive with auto-retry
# =============================================================================
# Purpose: Upload vivoice_clean_wavs (~135GB) to Drive WITHOUT needing
#          135GB free on disk D (uses pipe streaming).
#
# Features:
#   - Auto-retry up to 5 times if upload fails midway
#   - Detailed logging with timestamps
#   - Disables sleep / monitor timeout during upload
#   - Pre-flight check: tar.exe, rclone, gdrive remote, source folder, disk
#
# Usage (open PowerShell, NO admin needed):
#     cd D:\Downloads
#     Set-ExecutionPolicy -Scope Process Bypass -Force
#     .\upload_vivoice.ps1
# =============================================================================

# ============================================================
# CONFIG -- edit these if your paths differ
# ============================================================
$SOURCE_DIR  = "D:\Documents\HUST\HUST_Project\Project_Final\TTS_StyleTTS2\fine-tune\data_pipeline\prepare_vivoice\output\vivoice_clean_wavs"
$DEST_REMOTE = "gdrive:vastai_upload/vivoice_clean_wavs.tar"
$LOG_FILE    = "$PSScriptRoot\upload_vivoice.log"
$MAX_RETRIES = 5
$RETRY_WAIT_MIN = 5

# ============================================================
# HELPER: logging functions
# ============================================================
function Write-Log {
    param([string]$msg, [string]$color = "White")
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line -ForegroundColor $color
    Add-Content -Path $LOG_FILE -Value $line
}

function Write-Section {
    param([string]$title)
    $border = "=" * 70
    Write-Host ""
    Write-Host $border -ForegroundColor Cyan
    Write-Host "  $title" -ForegroundColor Cyan
    Write-Host $border -ForegroundColor Cyan
    Add-Content -Path $LOG_FILE -Value ""
    Add-Content -Path $LOG_FILE -Value $border
    Add-Content -Path $LOG_FILE -Value "  $title"
    Add-Content -Path $LOG_FILE -Value $border
}

# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================
Write-Section "PRE-FLIGHT CHECKS"

# Check 1: tar.exe
$tarPath = Get-Command tar.exe -ErrorAction SilentlyContinue
if (-not $tarPath) {
    Write-Log "FAIL: tar.exe not found! Windows 10+ should have it built-in." "Red"
    exit 1
}
Write-Log "OK: tar.exe at $($tarPath.Source)" "Green"

# Check 2: rclone
$rclonePath = Get-Command rclone -ErrorAction SilentlyContinue
if (-not $rclonePath) {
    Write-Log "FAIL: rclone not found! Restart PowerShell after installing rclone." "Red"
    exit 1
}
Write-Log "OK: rclone at $($rclonePath.Source)" "Green"

# Check 3: rclone remote 'gdrive'
$remotes = rclone listremotes 2>$null
if ($remotes -notmatch "gdrive:") {
    Write-Log "FAIL: rclone remote 'gdrive' not configured! Run: rclone config" "Red"
    exit 1
}
Write-Log "OK: rclone remote 'gdrive' configured" "Green"

# Check 4: Source folder
if (-not (Test-Path $SOURCE_DIR)) {
    Write-Log "FAIL: Source folder not found: $SOURCE_DIR" "Red"
    exit 1
}

$srcInfo = Get-ChildItem $SOURCE_DIR -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum
$srcSize = $srcInfo.Sum / 1GB
$srcCount = $srcInfo.Count
Write-Log "OK: Source folder $SOURCE_DIR" "Green"
Write-Log "    Files: $srcCount" "Green"
Write-Log "    Size : $([math]::Round($srcSize, 2)) GB" "Green"

# Check 5: Disk space
$driveLetter = (Split-Path $SOURCE_DIR -Qualifier).TrimEnd(":")
$drive = Get-PSDrive -Name $driveLetter -ErrorAction SilentlyContinue
if ($drive) {
    $freeGB = $drive.Free / 1GB
    Write-Log "OK: Free disk on ${driveLetter}: $([math]::Round($freeGB, 2)) GB" "Green"
    if ($freeGB -lt 5) {
        Write-Log "WARN: Very low disk free ($([math]::Round($freeGB, 1)) GB). Stream mode is OK." "Yellow"
    }
}

# Check 6: Create remote folder if not exists
Write-Log "Ensuring remote folder exists..." "White"
rclone mkdir "gdrive:vastai_upload" 2>$null
Write-Log "OK: Remote folder ready" "Green"

# ============================================================
# PREVENT SYSTEM SLEEP DURING UPLOAD
# ============================================================
Write-Section "PREVENT SYSTEM SLEEP"
Write-Log "Disabling sleep and monitor timeout..." "White"

powercfg /change standby-timeout-ac 0    | Out-Null
powercfg /change hibernate-timeout-ac 0  | Out-Null
powercfg /change monitor-timeout-ac 0    | Out-Null

Write-Log "OK: Sleep disabled (will restore at end of script)" "Green"

# ============================================================
# UPLOAD LOOP WITH RETRY
# ============================================================
Write-Section "UPLOAD: tar -> rclone rcat"
Write-Log "Source: $SOURCE_DIR" "White"
Write-Log "Dest  : $DEST_REMOTE" "White"
Write-Log "Method: stream pipe (no local tar file needed)" "White"

# Move to parent dir so tar uses relative path
$parentDir = Split-Path $SOURCE_DIR -Parent
$folderName = Split-Path $SOURCE_DIR -Leaf
Push-Location $parentDir
Write-Log "Working dir: $parentDir" "White"
Write-Log "Tar target : $folderName" "White"

$attempt = 0
$success = $false
$startTime = Get-Date

while ($attempt -lt $MAX_RETRIES -and -not $success) {
    $attempt++
    Write-Log "" "White"
    Write-Log "ATTEMPT $attempt / $MAX_RETRIES" "Yellow"
    Write-Log "----------------------------------------" "Yellow"
    $attemptStart = Get-Date
    Write-Log "Start: $($attemptStart.ToString('HH:mm:ss'))" "White"

    # Use cmd.exe to handle binary pipe properly
    # PowerShell native pipe corrupts binary data, cmd /c handles it correctly
    $exitCode = 0
    try {
        $cmdLine = "tar.exe -cf - `"$folderName`" | rclone rcat `"$DEST_REMOTE`" --progress --drive-chunk-size 256M --buffer-size 64M --retries 10 --low-level-retries 20 --stats 30s"
        Write-Log "Command: $cmdLine" "Gray"

        & cmd /c $cmdLine
        $exitCode = $LASTEXITCODE
    }
    catch {
        Write-Log "Exception: $($_.Exception.Message)" "Red"
        $exitCode = 1
    }

    $attemptEnd = Get-Date
    $elapsed = $attemptEnd - $attemptStart
    Write-Log "End: $($attemptEnd.ToString('HH:mm:ss'))" "White"
    Write-Log "Elapsed: $($elapsed.ToString('hh\:mm\:ss'))" "White"
    Write-Log "Exit code: $exitCode" "White"

    if ($exitCode -eq 0) {
        Write-Log "SUCCESS: Upload completed" "Green"

        # Verify size
        $remoteInfo = rclone size $DEST_REMOTE 2>$null
        Write-Log "Remote info: $remoteInfo" "Green"
        $success = $true
    }
    else {
        Write-Log "FAILED: Upload returned exit code $exitCode" "Red"
        if ($attempt -lt $MAX_RETRIES) {
            Write-Log "Waiting $RETRY_WAIT_MIN minutes before retry..." "Yellow"
            Start-Sleep -Seconds ($RETRY_WAIT_MIN * 60)

            # Delete partial file on Drive before retry
            Write-Log "Deleting partial file on Drive before retry..." "Yellow"
            rclone delete $DEST_REMOTE 2>$null
        }
    }
}

Pop-Location

# ============================================================
# RESTORE POWER SETTINGS
# ============================================================
Write-Section "CLEANUP"
Write-Log "Restoring power settings..." "White"
powercfg /change standby-timeout-ac 30   | Out-Null
powercfg /change monitor-timeout-ac 10   | Out-Null
Write-Log "OK: Power settings restored" "Green"

# ============================================================
# SUMMARY
# ============================================================
Write-Section "SUMMARY"
$totalElapsed = (Get-Date) - $startTime
Write-Log "Total time: $($totalElapsed.ToString('hh\:mm\:ss'))" "White"
Write-Log "Attempts  : $attempt / $MAX_RETRIES" "White"

if ($success) {
    Write-Log "" "White"
    Write-Log "============================================================" "Green"
    Write-Log "  UPLOAD COMPLETED SUCCESSFULLY" "Green"
    Write-Log "============================================================" "Green"
    Write-Log "" "White"
    Write-Log "Next steps:" "Cyan"
    Write-Log "  1. Verify: rclone size gdrive:vastai_upload" "White"
    Write-Log "  2. Upload metadata files (filelists, vocab, plbert_v2)" "White"
    Write-Log "  3. On Vast.ai: bash pull_data.sh" "White"
}
else {
    Write-Log "" "White"
    Write-Log "============================================================" "Red"
    Write-Log "  UPLOAD FAILED after $MAX_RETRIES attempts" "Red"
    Write-Log "============================================================" "Red"
    Write-Log "" "White"
    Write-Log "Troubleshooting:" "Cyan"
    Write-Log "  - Internet stable? Test: speedtest.net" "White"
    Write-Log "  - Drive quota? Check: rclone about gdrive:" "White"
    Write-Log "  - Re-auth token: rclone config reconnect gdrive:" "White"
}

Write-Log "Log file: $LOG_FILE" "White"
