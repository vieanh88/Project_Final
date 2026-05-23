#!/usr/bin/env bash
# =============================================================================
# pull_data.sh — Stream extract vivoice_clean_wavs.tar từ Google Drive về Vast.ai
# =============================================================================
# Mục tiêu:
#   - CHỈ tải/extract dataset vivoice_clean_wavs
#   - KHÔNG lưu file .tar 126+ GiB xuống disk
#   - Phù hợp khi Vast.ai còn ~210GB, data sau giải nén ~126-135GB
#
# Cơ chế:
#   rclone cat gdrive:vastai_upload/vivoice_clean_wavs.tar | tar -xf -
#
# Mặc định theo cấu trúc bạn đang dùng trên Vast.ai:
#   /workspace/Project_Final/TTS_StyleTTS2/fine-tune/data_pipeline/prepare_vivoice/output
#
# Có thể override bằng biến môi trường, ví dụ:
#   LOCAL_OUTPUT_PARENT=/workspace/.../output bash pull_data.sh
#   GDRIVE_TAR_PATH=vastai_upload/vivoice_clean_wavs.tar bash pull_data.sh
# =============================================================================

set -Eeuo pipefail

# ----------------------------- CONFIG ----------------------------------------
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
GDRIVE_TAR_PATH="${GDRIVE_TAR_PATH:-vastai_upload/vivoice_clean_wavs.tar}"
LOCAL_OUTPUT_PARENT="${LOCAL_OUTPUT_PARENT:-/workspace/Project_Final/TTS_StyleTTS2/fine-tune/data_pipeline/prepare_vivoice/output}"
TARGET_DIR="${LOCAL_OUTPUT_PARENT}/vivoice_clean_wavs"

EXPECTED_WAV_COUNT="${EXPECTED_WAV_COUNT:-695111}"
MIN_WAV_COUNT="${MIN_WAV_COUNT:-690000}"
MIN_FREE_GB="${MIN_FREE_GB:-155}"
MIN_FREE_INODES="${MIN_FREE_INODES:-750000}"

# Nếu thư mục đích đã có dữ liệu, script sẽ dừng để tránh ghi đè nhầm.
# Muốn chạy lại/extract đè lên thư mục đang có, dùng:
#   FORCE_OVERWRITE=1 bash pull_data.sh
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"

# Rclone tuning. Giảm multi-thread nếu bị Google Drive rate-limit khi download.
RCLONE_BUFFER_SIZE="${RCLONE_BUFFER_SIZE:-256M}"
RCLONE_MULTI_THREAD_STREAMS="${RCLONE_MULTI_THREAD_STREAMS:-4}"
RCLONE_MULTI_THREAD_CUTOFF="${RCLONE_MULTI_THREAD_CUTOFF:-250M}"

# ----------------------------- COLORS/LOG ------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${BLUE}[$(date '+%F %T')]${NC} $*"; }
ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $*"; }
err()  { echo -e "  ${RED}✗${NC} $*"; }

on_error() {
    local exit_code=$?
    echo ""
    err "Script failed with exit code ${exit_code}."
    warn "Nếu lỗi xảy ra giữa chừng, thư mục ${TARGET_DIR} có thể đang ở trạng thái extract dở."
    warn "Không train bằng dữ liệu này cho tới khi chạy verify thành công."
    exit "$exit_code"
}
trap on_error ERR

# ----------------------------- PREFLIGHT -------------------------------------
echo "============================================================================="
echo "  PULL VIVOICE DATA: Google Drive tar -> stream extract -> Vast.ai disk"
echo "============================================================================="
log "Config đang dùng:"
echo "  RCLONE_REMOTE       = ${RCLONE_REMOTE}:"
echo "  GDRIVE_TAR_PATH     = ${GDRIVE_TAR_PATH}"
echo "  Remote tar          = ${RCLONE_REMOTE}:${GDRIVE_TAR_PATH}"
echo "  LOCAL_OUTPUT_PARENT = ${LOCAL_OUTPUT_PARENT}"
echo "  TARGET_DIR          = ${TARGET_DIR}"
echo ""

log "STEP 1/6: Kiểm tra command bắt buộc..."
for cmd in rclone tar df awk find du tee; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        err "Thiếu command: $cmd"
        exit 1
    fi
    ok "$cmd: $(command -v "$cmd")"
done

log "STEP 2/6: Kiểm tra rclone remote..."
if ! rclone listremotes | grep -Fxq "${RCLONE_REMOTE}:"; then
    err "Không thấy rclone remote '${RCLONE_REMOTE}:' trong server."
    echo ""
    echo "Cách xử lý:"
    echo "  1) Chạy: rclone config"
    echo "  2) Tạo remote tên: ${RCLONE_REMOTE}"
    echo "  3) Type: drive / Google Drive"
    echo "  4) Nếu server không mở được browser, dùng chế độ headless/remote config của rclone."
    exit 1
fi
ok "Remote ${RCLONE_REMOTE}: tồn tại"

log "STEP 3/6: Kiểm tra file tar trên Google Drive..."
REMOTE_TAR="${RCLONE_REMOTE}:${GDRIVE_TAR_PATH}"
REMOTE_DIR="${RCLONE_REMOTE}:$(dirname "${GDRIVE_TAR_PATH}")"
REMOTE_FILE="$(basename "${GDRIVE_TAR_PATH}")"

if ! rclone lsf "$REMOTE_DIR" --files-only 2>/dev/null | grep -Fxq "$REMOTE_FILE"; then
    err "Không tìm thấy ${REMOTE_TAR}"
    echo ""
    echo "Hãy kiểm tra bằng lệnh:"
    echo "  rclone lsf ${REMOTE_DIR} --files-only"
    echo ""
    echo "Theo log upload hiện tại, path đúng nhiều khả năng là:"
    echo "  gdrive:vastai_upload/vivoice_clean_wavs.tar"
    exit 1
fi
ok "Tìm thấy ${REMOTE_TAR}"

log "Thông tin size remote:"
rclone size "$REMOTE_TAR"

log "STEP 4/6: Kiểm tra disk space và inode..."
mkdir -p "$LOCAL_OUTPUT_PARENT"
AVAIL_GB=$(df --output=avail -BG "$LOCAL_OUTPUT_PARENT" | tail -1 | tr -dc '0-9')
AVAIL_INODES=$(df -i --output=iavail "$LOCAL_OUTPUT_PARENT" | tail -1 | tr -dc '0-9')

echo "  Available disk  : ${AVAIL_GB} GB"
echo "  Available inode : ${AVAIL_INODES}"

if [ "$AVAIL_GB" -lt "$MIN_FREE_GB" ]; then
    err "Không đủ disk. Cần >= ${MIN_FREE_GB}GB free, hiện có ${AVAIL_GB}GB."
    exit 1
fi
ok "Disk đủ để stream extract mà không lưu tar"

if [ "$AVAIL_INODES" -lt "$MIN_FREE_INODES" ]; then
    err "Không đủ inode. Dataset có khoảng ${EXPECTED_WAV_COUNT} file .wav, hiện chỉ còn ${AVAIL_INODES} inode."
    exit 1
fi
ok "Inode đủ cho ~${EXPECTED_WAV_COUNT} file wav"

if [ -d "$TARGET_DIR" ] && find "$TARGET_DIR" -mindepth 1 -print -quit | grep -q .; then
    EXISTING_WAVS=$(find "$TARGET_DIR" -type f -name '*.wav' 2>/dev/null | wc -l | tr -dc '0-9')
    warn "TARGET_DIR đã tồn tại và không rỗng: ${TARGET_DIR}"
    warn "Số wav hiện có trong thư mục: ${EXISTING_WAVS}"
    if [ "$FORCE_OVERWRITE" != "1" ]; then
        echo ""
        echo "Để tránh ghi đè nhầm, script dừng tại đây."
        echo "Nếu đây là extract dở và bạn muốn chạy lại/ghi đè, chạy:"
        echo "  FORCE_OVERWRITE=1 bash pull_data.sh"
        exit 1
    fi
    warn "FORCE_OVERWRITE=1 nên sẽ tiếp tục extract đè lên dữ liệu hiện có."
fi

# ----------------------------- STREAM EXTRACT --------------------------------
log "STEP 5/6: Stream extract vivoice_clean_wavs.tar..."
echo "  Method: rclone cat \"${REMOTE_TAR}\" | tar -xf - -C \"${LOCAL_OUTPUT_PARENT}\""
echo "  Chú ý: không tạo file .tar local, chỉ ghi thư mục vivoice_clean_wavs sau giải nén."
echo ""

LOG_FILE="${LOCAL_OUTPUT_PARENT}/pull_vivoice_$(date '+%Y%m%d_%H%M%S').log"
log "Log sẽ lưu tại: ${LOG_FILE}"

# Dùng subshell để vừa hiển thị console vừa ghi log.
(
    echo "Started at: $(date '+%F %T')"
    echo "Remote tar: ${REMOTE_TAR}"
    echo "Output    : ${LOCAL_OUTPUT_PARENT}"
    echo ""

    rclone cat "$REMOTE_TAR" \
        --buffer-size "$RCLONE_BUFFER_SIZE" \
        --multi-thread-streams "$RCLONE_MULTI_THREAD_STREAMS" \
        --multi-thread-cutoff "$RCLONE_MULTI_THREAD_CUTOFF" \
        --retries 10 \
        --low-level-retries 20 \
        --stats 30s \
        --stats-one-line \
        | tar -xf - -C "$LOCAL_OUTPUT_PARENT" --no-same-owner --checkpoint=10000 --checkpoint-action=dot

    echo ""
    echo "Finished extract at: $(date '+%F %T')"
) 2>&1 | tee "$LOG_FILE"

ok "Stream extract command finished"

# ----------------------------- VERIFY ----------------------------------------
log "STEP 6/6: Verify dữ liệu đã giải nén..."

if [ ! -d "$TARGET_DIR" ]; then
    err "Không thấy thư mục sau giải nén: ${TARGET_DIR}"
    exit 1
fi

WAV_COUNT=$(find "$TARGET_DIR" -type f -name '*.wav' | wc -l | tr -dc '0-9')
SIZE_HUMAN=$(du -sh "$TARGET_DIR" | awk '{print $1}')
SIZE_GB=$(du -s "$TARGET_DIR" | awk '{printf "%.2f", $1/1024/1024}')

echo "  Target dir : ${TARGET_DIR}"
echo "  WAV count  : ${WAV_COUNT}"
echo "  Size       : ${SIZE_HUMAN} (${SIZE_GB} GiB approx)"
echo ""

if [ "$WAV_COUNT" -lt "$MIN_WAV_COUNT" ]; then
    err "WAV count quá thấp: ${WAV_COUNT}. Kỳ vọng khoảng ${EXPECTED_WAV_COUNT}. Extract có thể bị lỗi/dở."
    exit 1
fi

if [ "$WAV_COUNT" -ne "$EXPECTED_WAV_COUNT" ]; then
    warn "WAV count khác con số upload log: expected=${EXPECTED_WAV_COUNT}, actual=${WAV_COUNT}."
    warn "Nếu chênh lệch nhỏ, hãy kiểm tra thêm; nếu chênh lớn, không train vội."
else
    ok "WAV count khớp upload log: ${EXPECTED_WAV_COUNT}"
fi

log "Sample vài file đầu tiên:"
find "$TARGET_DIR" -type f -name '*.wav' | head -5

echo ""
echo "Disk sau khi extract:"
df -h "$LOCAL_OUTPUT_PARENT"

echo ""
echo "============================================================================="
echo -e "${GREEN}  ✅ DONE: vivoice_clean_wavs đã được stream extract thành công${NC}"
echo "============================================================================="
echo "Target: ${TARGET_DIR}"
echo "Log   : ${LOG_FILE}"
echo ""
