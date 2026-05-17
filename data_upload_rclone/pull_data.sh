#!/bin/bash
# =============================================================================
#  pull_data.sh — Tải data từ Drive về VPS (STREAM EXTRACT, không cần lưu tar)
# =============================================================================
#  Giải quyết vấn đề: VPS chỉ 210GB disk nhưng data 130GB + tar 130GB = 260GB.
#  Solution: rclone cat <tar> | tar -xf -   (KHÔNG lưu tar xuống disk)
#
#  Yêu cầu trước:
#    - rclone đã setup với remote "gdrive"
#    - Trên Google Drive đã có:
#        gdrive:ghost_story_narrator/data/vivoice_clean_wavs.tar
#        gdrive:ghost_story_narrator/data/<other files>
#        gdrive:ghost_story_narrator/plbert_v2/checkpoints/
#
#  Chạy:
#    bash /workspace/scripts/pull_data.sh
# =============================================================================

set -e
set -u

GDRIVE_ROOT="gdrive:vastai_upload"
TAR_NAME="vivoice_clean_wavs.tar"
LOCAL_DATA="/workspace/Project_Final/TTS_StyleTTS2/fine-tune/data_pipeline/prepare_vivoice/output"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log()    { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()     { echo -e "  ${GREEN}✓${NC} $*"; }
warn()   { echo -e "  ${YELLOW}⚠${NC} $*"; }
err()    { echo -e "  ${RED}✗${NC} $*"; }

# ============================================================
# CHECK PREREQUISITES
# ============================================================
if ! rclone listremotes | grep -q "^gdrive:"; then
    err "rclone remote 'gdrive' chưa setup. Chạy: rclone config"
    exit 1
fi

if ! rclone lsd "$GDRIVE_REMOTE" > /dev/null 2>&1; then
    err "Không thấy folder '$GDRIVE_REMOTE'. Kiểm tra tên folder trên Drive."
    exit 1
fi

mkdir -p "$LOCAL_DATA"

# ============================================================
# DISK SPACE CHECK
# ============================================================
log "Kiểm tra disk space..."
AVAIL_GB=$(df --output=avail -BG "$LOCAL_DATA" | tail -1 | tr -dc '0-9')
log "  Available: ${AVAIL_GB} GB"
if [ "$AVAIL_GB" -lt 150 ]; then
    err "Cần ít nhất 150GB free, hiện chỉ ${AVAIL_GB}GB. Stop."
    exit 1
fi
ok "Disk đủ"

# ============================================================
# STEP 1: Tải METADATA nhỏ trước (vài MB, vài giây)
# ============================================================
log "STEP 1/4: Tải metadata files..."

declare -a META_FILES=(
    "data/phoneme_vocab.json"
    "data/vivoice_train_list.txt"
    "data/vivoice_val_list.txt"
    "data/speaker_id_map.json"
    "data/OOD_texts_phoneme.txt"
)

for f in "${META_FILES[@]}"; do
    if rclone copy "$GDRIVE_REMOTE/$f" "$LOCAL_DATA/$(dirname "$f")/" \
        --transfers 4 \
        --quiet 2>/dev/null; then
        ok "$(basename $f)"
    else
        warn "$f không có hoặc tải thất bại"
    fi
done

# ============================================================
# STEP 2: Tải Ngạn data (nhỏ, ~1-5 GB) — KHÔNG cần stream
# ============================================================
log ""
log "STEP 2/4: Tải Ngạn dataset..."
mkdir -p "$LOCAL_DATA/ngan"
rclone copy "$GDRIVE_REMOTE/data/ngan/" "$LOCAL_DATA/ngan/" \
    --progress \
    --transfers 8 \
    --buffer-size 32M
ok "Ngạn dataset downloaded"

# ============================================================
# STEP 3: Tải PL-BERT v2 checkpoints (vài MB)
# ============================================================
log ""
log "STEP 3/4: Tải PL-BERT v2 checkpoints..."
mkdir -p "$LOCAL_DATA/plbert_v2/checkpoints"
rclone copy "$GDRIVE_REMOTE/plbert_v2/checkpoints/" "$LOCAL_DATA/plbert_v2/checkpoints/" \
    --progress \
    --transfers 4
ok "PL-BERT checkpoints downloaded"
ls -la "$LOCAL_DATA/plbert_v2/checkpoints/"

# ============================================================
# STEP 4: STREAM EXTRACT  ⭐ ĐOẠN QUAN TRỌNG NHẤT ⭐
# ============================================================
log ""
log "STEP 4/4: STREAM EXTRACT vivoice_clean_wavs (~130GB)..."
log "  Method: rclone cat | tar -xf -  (KHÔNG lưu .tar xuống disk)"
log "  Estimated time: 1-3 giờ tùy bandwidth Drive → VPS"
echo ""

TARGET_DIR="$LOCAL_DATA/vivoice_clean_wavs"

if [ -d "$TARGET_DIR" ] && [ "$(ls -A "$TARGET_DIR" 2>/dev/null | wc -l)" -gt 1000 ]; then
    warn "Folder $TARGET_DIR đã có nhiều file"
    read -p "  Continue extract (sẽ overwrite)? [y/N] " ans
    if [ "$ans" != "y" ]; then
        warn "Skip extract"
        exit 0
    fi
fi

mkdir -p "$TARGET_DIR"
cd "$LOCAL_DATA"

# Phát hiện chunked hay single file
if rclone ls "$GDRIVE_REMOTE/data/" --include "vivoice_clean_wavs.tar.part_*" 2>/dev/null | head -1 | grep -q "part"; then
    log "  Detected CHUNKED tar (.part_*)"
    CHUNKS=$(rclone lsf "$GDRIVE_REMOTE/data/" --include "vivoice_clean_wavs.tar.part_*" | sort)
    echo "  Chunks:"
    for c in $CHUNKS; do
        echo "    - $c"
    done

    # Cat các chunk → pipe vào tar extract
    (
        for c in $CHUNKS; do
            rclone cat "$GDRIVE_REMOTE/data/$c" \
                --buffer-size 256M \
                --multi-thread-streams 8 \
                --multi-thread-cutoff 250M
        done
    ) | tar -xf - --checkpoint=10000 --checkpoint-action=dot

else
    log "  Detected SINGLE tar file"
    rclone cat "$GDRIVE_REMOTE/data/vivoice_clean_wavs.tar" \
        --buffer-size 256M \
        --multi-thread-streams 8 \
        --multi-thread-cutoff 250M \
        | tar -xf - --checkpoint=10000 --checkpoint-action=dot
fi

ok "Stream extract completed"

# ============================================================
# VERIFY
# ============================================================
log ""
log "Verifying extracted data..."
WAV_COUNT=$(find "$TARGET_DIR" -name "*.wav" -type f 2>/dev/null | wc -l)
EXTRACTED_GB=$(du -s "$TARGET_DIR" | awk '{print $1/1024/1024}')

log "  Total .wav files: $WAV_COUNT"
log "  Total size      : ${EXTRACTED_GB} GB (approx)"

if [ "$WAV_COUNT" -lt 100000 ]; then
    err "Wav count quá thấp ($WAV_COUNT). Extract có vấn đề?"
    exit 1
fi
ok "Verified: $WAV_COUNT wav files"

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "============================================================================="
echo -e "${GREEN}  ✅ DATA PULL COMPLETED${NC}"
echo "============================================================================="
echo ""
df -h "$LOCAL_DATA"
echo ""
du -sh "$LOCAL_DATA"/*
echo ""
echo "Records:"
for f in vivoice_train_list.txt vivoice_val_list.txt; do
    if [ -f "$LOCAL_DATA/$f" ]; then
        count=$(wc -l < "$LOCAL_DATA/$f")
        printf "  %-30s %s records\n" "$f" "$count"
    fi
done
echo ""
echo "NEXT:"
echo "  export PHONEME_VOCAB_PATH=$LOCAL_DATA/phoneme_vocab.json"
echo "  cd /workspace/code/StyleTTS2/Utils/PLBERT"
echo "  python verify_plbert.py --plbert_dir $LOCAL_DATA/plbert_v2/checkpoints"