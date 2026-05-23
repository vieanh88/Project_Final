#!/usr/bin/env bash
# =============================================================================
# pull_data_vastai_fixed.sh
# Stream-extract vivoice_clean_wavs.tar from Google Drive to Vast.ai WITHOUT
# storing the .tar on the Vast disk.
# =============================================================================

set -Eeuo pipefail

# -------- EDIT ONLY THESE VARIABLES IF YOUR PATHS DIFFER --------
GDRIVE_ROOT="gdrive:vastai_upload"
TAR_NAME="vivoice_clean_wavs.tar"
LOCAL_DATA="/workspace/Project_Final/TTS_StyleTTS2/fine-tune/data_pipeline/prepare_vivoice/output"
MIN_FREE_GB=150
EXPECTED_MIN_WAV=690000
# ---------------------------------------------------------------

TAR_REMOTE="${GDRIVE_ROOT}/${TAR_NAME}"
TARGET_DIR="${LOCAL_DATA}/vivoice_clean_wavs"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${BLUE}[$(date '+%F %T')]${NC} $*"; }
ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $*"; }
err()  { echo -e "  ${RED}✗${NC} $*"; }

on_error() {
  err "Command failed at line ${BASH_LINENO[0]}: ${BASH_COMMAND}"
  warn "Nếu lỗi xảy ra giữa lúc giải nén, có thể chạy lại script. tar sẽ ghi đè lại các file đã có."
}
trap on_error ERR

log "Pre-flight checks"

if ! command -v rclone >/dev/null 2>&1; then
  err "rclone chưa được cài trong instance. Cài bằng: curl https://rclone.org/install.sh | sudo bash"
  exit 1
fi
ok "rclone found: $(rclone version | head -n 1)"

if ! rclone listremotes | grep -q '^gdrive:'; then
  err "Không thấy remote 'gdrive:' trong instance. Cần cấu hình rclone trên Vast trước."
  echo "Gợi ý: chạy rclone config, hoặc copy rclone.conf từ máy Windows sang instance."
  exit 1
fi
ok "rclone remote gdrive exists"

if ! rclone ls "${TAR_REMOTE}" >/dev/null 2>&1; then
  err "Không tìm thấy tar trên Drive: ${TAR_REMOTE}"
  echo "Kiểm tra bằng: rclone lsf gdrive:vastai_upload"
  echo "Nếu file nằm chỗ khác, sửa GDRIVE_ROOT/TAR_NAME ở đầu script."
  exit 1
fi
ok "Found remote tar: ${TAR_REMOTE}"

mkdir -p "${LOCAL_DATA}"
AVAIL_GB=$(df --output=avail -BG "${LOCAL_DATA}" | tail -1 | tr -dc '0-9')
log "Available disk at ${LOCAL_DATA}: ${AVAIL_GB} GB"
if [ "${AVAIL_GB}" -lt "${MIN_FREE_GB}" ]; then
  err "Không đủ disk. Cần tối thiểu ${MIN_FREE_GB}GB trống vì data giải nén khoảng 126-135GB."
  exit 1
fi
ok "Disk is enough for stream extract"

if [ -d "${TARGET_DIR}" ]; then
  EXISTING_WAV=$(find "${TARGET_DIR}" -type f -name '*.wav' 2>/dev/null | wc -l || true)
  if [ "${EXISTING_WAV}" -gt 1000 ]; then
    warn "${TARGET_DIR} đã có ${EXISTING_WAV} wav files. Script sẽ ghi đè khi extract."
    read -r -p "Tiếp tục? [y/N] " ans
    if [ "${ans}" != "y" ] && [ "${ans}" != "Y" ]; then
      warn "Stopped by user."
      exit 0
    fi
  fi
fi

log "Stream extracting: ${TAR_REMOTE} -> ${LOCAL_DATA}"
log "Không lưu ${TAR_NAME} xuống disk. Dữ liệu đi qua pipe: rclone cat | tar -xf -"
mkdir -p "${TARGET_DIR}"
cd "${LOCAL_DATA}"

# The uploaded tar was created from the parent folder and contains top-level folder vivoice_clean_wavs/.
rclone cat "${TAR_REMOTE}" \
  --buffer-size 256M \
  --multi-thread-streams 4 \
  --multi-thread-cutoff 250M \
  --stats 30s \
  | tar -xf - --checkpoint=10000 --checkpoint-action=dot

echo ""
ok "Stream extract completed"

log "Verifying extracted dataset"
WAV_COUNT=$(find "${TARGET_DIR}" -type f -name '*.wav' | wc -l)
SIZE_HUMAN=$(du -sh "${TARGET_DIR}" | awk '{print $1}')
log "WAV count: ${WAV_COUNT}"
log "Size     : ${SIZE_HUMAN}"

if [ "${WAV_COUNT}" -lt "${EXPECTED_MIN_WAV}" ]; then
  err "Số wav thấp hơn kỳ vọng (${EXPECTED_MIN_WAV}). Không nên train trước khi kiểm tra lại."
  exit 1
fi
ok "Verified ${WAV_COUNT} wav files"

cat <<MSG

=============================================================================
ViVoice wav data is ready
=============================================================================
Local wav folder:
  ${TARGET_DIR}

Next checks:
  df -h "${LOCAL_DATA}"
  du -sh "${TARGET_DIR}"
  find "${TARGET_DIR}" -type f -name '*.wav' | wc -l

Important:
  Filelist train/val phải trỏ tới đúng path trên server:
  ${TARGET_DIR}/<filename>.wav
MSG
