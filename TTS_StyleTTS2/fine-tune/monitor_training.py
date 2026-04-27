"""
=============================================================================
  MONITOR TRAINING — Giám sát training StyleTTS2 qua TensorBoard + Discord
=============================================================================
Mục tiêu: Chạy SONG SONG với train_wrapper.py để:
  1. Đọc TensorBoard events file realtime
  2. Phát hiện plateau của val loss → gợi ý early-stop thủ công
  3. Cảnh báo NaN/Inf loss (training sẽ hỏng)
  4. Cảnh báo overfitting (val loss tăng liên tiếp)
  5. Báo cáo tiến độ mỗi N epochs
  6. Gửi notification qua Discord Webhook tới 2 thiết bị
  7. Print ra console

CHÚ Ý: Script CHỈ cảnh báo, KHÔNG tự dừng training.
        Bạn tự quyết định Ctrl+C khi nhận notification.

Setup Discord Webhook:
  1. Discord → Server Settings → Integrations → Webhooks → New Webhook
  2. Copy URL dạng: https://discord.com/api/webhooks/XXXXX/YYYYY
  3. Paste vào .env hoặc CLI flag

Chạy lệnh (Terminal 2 — song song với train):
    python monitor_training.py --log-dir "Models/VietnameseBase"
    python monitor_training.py --log-dir "..." --patience 15 --min-delta 0.001
    python monitor_training.py --log-dir "..." --dry-run  # không gửi Discord
=============================================================================
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from collections import deque
from datetime import datetime

from dotenv import load_dotenv

# KHẮC PHỤC LỖI ENCODING TRÊN WINDOWS
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"

# CONFIGURATION
@dataclass
class MonitorConfig:
    """Cấu hình cho script monitoring."""

    # --- Đường dẫn log_dir của stage đang train ---
    # TensorBoard events file nằm ở: {log_dir}/tensorboard/
    log_dir: str = ""

    # --- Stage (chỉ dùng để hiển thị trong notification) ---
    stage: int = 1
    stage_name: str = "Stage 1 - Acoustic & Alignment"

    # --- Metrics cần track ---
    # Theo code gốc train_first.py, các val metrics được ghi bởi:
    #   writer.add_scalar('eval/mel_loss', ...)
    #   writer.add_scalar('eval/mono_align_loss', ...) / 'eval/align_loss'
    # Nếu tên tag khác, chỉnh ở đây.
    primary_metric: str = "eval/mel_loss"
    secondary_metric: str = "eval/mono_align_loss"

    # --- Plateau detection ---
    patience: int = 5                # Số epochs liên tiếp không giảm đáng kể
    min_delta: float = 0.001         # Mức giảm tối thiểu được coi là "có cải thiện"

    # --- Overfitting detection ---
    overfitting_patience: int = 5     # Val loss tăng liên tiếp N epochs → cảnh báo
    overfitting_min_increase: float = 0.01  # Mức tăng tối thiểu coi là "đang overfit"

    # --- Progress report ---
    progress_report_interval: int = 10  # Báo cáo mỗi N epochs

    # --- Polling interval ---
    poll_interval_s: int = 60          # Scan TensorBoard mỗi N giây

    # --- Discord Webhooks ---
    # 2 URL để gửi tới 2 thiết bị
    discord_webhook_1: str = ""
    discord_webhook_2: str = ""

    # --- Tùy chọn ---
    dry_run: bool = False              # Không thực sự gửi Discord (chỉ print)

    # --- Log file cho monitor ---
    monitor_log_file: str = "monitor_training.log"

    @classmethod
    def from_env(cls) -> "MonitorConfig":
        """Load Discord webhooks từ .env."""
        config = cls()
        config.discord_webhook_1 = os.environ.get("DISCORD_WEBHOOK_1", "")
        config.discord_webhook_2 = os.environ.get("DISCORD_WEBHOOK_2", "")
        return config

# LOGGING SETUP
def setup_logging(log_path: Path) -> logging.Logger:
    """Thiết lập logging ra console + file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_path), encoding="utf-8", mode="w"),
        ],
    )
    return logging.getLogger("monitor")

# DISCORD NOTIFICATION
# Màu sắc embed theo mức độ (Discord dùng decimal int)
DISCORD_COLORS = {
    "info":      0x3498DB,  # Xanh dương
    "progress":  0x2ECC71,  # Xanh lá
    "warning":   0xF39C12,  # Cam
    "critical":  0xE74C3C,  # Đỏ
}

def send_discord(
    webhook_url: str,
    title: str,
    description: str,
    level: str,
    logger: logging.Logger,
    fields: Optional[List[Dict]] = None,
) -> bool:
    """
    Gửi message tới Discord qua webhook.

    Args:
        webhook_url: URL webhook
        title: Tiêu đề embed
        description: Nội dung chính
        level: "info" | "progress" | "warning" | "critical"
        fields: List of {"name": str, "value": str, "inline": bool}

    Returns:
        True nếu gửi thành công.
    """
    import requests

    if not webhook_url:
        return False

    color = DISCORD_COLORS.get(level, DISCORD_COLORS["info"])

    embed = {
        "title": title,
        "description": description,
        "color": color,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "footer": {"text": "StyleTTS2 Training Monitor"},
    }

    if fields:
        embed["fields"] = fields

    payload = {
        "username": "StyleTTS2 Monitor",
        "embeds": [embed],
    }

    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        if resp.status_code in (200, 204):
            return True
        logger.warning(f"Discord webhook trả về status {resp.status_code}: {resp.text[:200]}")
        return False
    except Exception as e:
        logger.warning(f"Không gửi được Discord: {e}")
        return False


def broadcast(
    config: MonitorConfig,
    title: str,
    description: str,
    level: str,
    logger: logging.Logger,
    fields: Optional[List[Dict]] = None,
):
    """Gửi notification tới CẢ 2 webhook + print ra console."""
    # Icon theo level
    icons = {
        "info":      "ℹ️",
        "progress":  "📊",
        "warning":   "⚠️",
        "critical":  "🚨",
    }
    icon = icons.get(level, "ℹ️")

    # Print ra console (luôn luôn, kể cả dry_run)
    console_msg = f"\n{'=' * 60}\n{icon}  [{level.upper()}] {title}\n{'=' * 60}\n{description}"
    if fields:
        console_msg += "\n\n"
        for f in fields:
            console_msg += f"  • {f['name']}: {f['value']}\n"
    console_msg += f"{'=' * 60}\n"

    if level == "critical":
        logger.error(console_msg)
    elif level == "warning":
        logger.warning(console_msg)
    else:
        logger.info(console_msg)

    # Gửi Discord (trừ khi dry_run)
    if config.dry_run:
        logger.info("  [DRY RUN] Không gửi Discord thực tế.")
        return

    # Thêm icon vào title cho Discord
    discord_title = f"{icon} {title}"

    results = []
    for idx, url in enumerate([config.discord_webhook_1, config.discord_webhook_2], 1):
        if url:
            ok = send_discord(url, discord_title, description, level, logger, fields)
            results.append(f"Webhook #{idx}: {'OK' if ok else 'FAIL'}")

    if results:
        logger.info(f"  Discord: {' | '.join(results)}")
    else:
        logger.warning("  Không có Discord webhook nào được cấu hình!")


# =============================================================================
# TENSORBOARD READER
# =============================================================================

class TensorBoardReader:
    """
    Đọc TensorBoard events file từ một thư mục.
    Track các scalar metrics đã xem để chỉ trả về dữ liệu MỚI sau mỗi lần gọi.
    """

    def __init__(self, tensorboard_dir: Path, logger: logging.Logger):
        self.tensorboard_dir = tensorboard_dir
        self.logger = logger

        # Track (tag, step) đã thấy → tránh emit lại
        self._seen_entries = set()

        # Lazy import — chỉ import khi thực sự cần
        self._ea = None

    def _get_event_accumulator(self):
        """Khởi tạo EventAccumulator (reload mỗi lần để đọc file mới)."""
        try:
            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )
        except ImportError:
            self.logger.error(
                "Thiếu package 'tensorboard'. Cài bằng: pip install tensorboard"
            )
            raise

        # size_guidance=0 → load tất cả scalar events
        ea = EventAccumulator(
            str(self.tensorboard_dir),
            size_guidance={"scalars": 0},
        )
        ea.Reload()
        return ea

    def wait_for_events_file(self, max_wait_s: int = 300):
        """Chờ tới khi TensorBoard events file xuất hiện."""
        start = time.time()
        while True:
            event_files = list(self.tensorboard_dir.glob("events.out.tfevents.*"))
            if event_files:
                self.logger.info(f"Phát hiện events file: {event_files[0].name}")
                return True

            if time.time() - start > max_wait_s:
                self.logger.error(
                    f"Không tìm thấy events file sau {max_wait_s}s tại {self.tensorboard_dir}"
                )
                return False

            self.logger.info(
                f"Chờ TensorBoard events file... ({int(time.time() - start)}s)"
            )
            time.sleep(5)

    def scan_new_metrics(self) -> Dict[str, List]:
        """
        Quét events file và trả về các scalar entries MỚI (chưa từng thấy).

        Returns:
            Dict[tag_name, List[(step, value, wall_time)]]
            Ví dụ: {"eval/mel_loss": [(10, 0.45, 1730...), (20, 0.42, 1730...)]}
        """
        try:
            ea = self._get_event_accumulator()
        except Exception as e:
            self.logger.warning(f"Lỗi đọc events: {e}")
            return {}

        available_tags = ea.Tags().get("scalars", [])
        if not available_tags:
            return {}

        new_data = {}

        for tag in available_tags:
            try:
                events = ea.Scalars(tag)
            except KeyError:
                continue

            new_entries = []
            for ev in events:
                key = (tag, ev.step)
                if key not in self._seen_entries:
                    self._seen_entries.add(key)
                    new_entries.append((ev.step, ev.value, ev.wall_time))

            if new_entries:
                new_data[tag] = new_entries

        return new_data

    def list_available_tags(self) -> List[str]:
        """Liệt kê tất cả scalar tags hiện có (để debug)."""
        try:
            ea = self._get_event_accumulator()
            return ea.Tags().get("scalars", [])
        except Exception:
            return []

# PLATEAU / OVERFITTING DETECTOR
class MetricTracker:
    """
    Theo dõi lịch sử 1 metric qua các epoch, phát hiện plateau/overfitting.
    """

    def __init__(self, name: str, min_delta: float, patience: int):
        self.name = name
        self.min_delta = min_delta
        self.patience = patience

        # History: List[(step/epoch, value)]
        self.history: List[tuple] = []

        # Best value seen so far
        self.best_value: Optional[float] = None
        self.best_step: Optional[int] = None

        # Số epochs liên tiếp không cải thiện
        self.stale_count: int = 0

        # Số epochs liên tiếp VAL TĂNG (overfitting)
        self.increasing_count: int = 0

        # Flag đã cảnh báo plateau chưa (tránh spam)
        self.plateau_alerted: bool = False
        self.overfit_alerted: bool = False

    def update(self, step: int, value: float) -> Dict[str, bool]:
        """
        Cập nhật metric mới. Trả về flags.

        Returns:
            {
                "is_nan": bool,
                "is_plateau": bool,           # Mới phát hiện plateau (chưa alert)
                "is_overfitting": bool,       # Mới phát hiện overfitting
                "improved": bool,             # Đạt best mới
            }
        """
        flags = {
            "is_nan": False,
            "is_plateau": False,
            "is_overfitting": False,
            "improved": False,
        }

        # Skip duplicate step
        if self.history and self.history[-1][0] == step:
            return flags

        # Check NaN/Inf
        if value != value or value in (float("inf"), float("-inf")):
            flags["is_nan"] = True
            self.history.append((step, value))
            return flags

        # Bắt đầu — set best
        if self.best_value is None:
            self.best_value = value
            self.best_step = step
            self.history.append((step, value))
            flags["improved"] = True
            return flags

        # So sánh với best
        if value < self.best_value - self.min_delta:
            # Có cải thiện đáng kể
            self.best_value = value
            self.best_step = step
            self.stale_count = 0
            self.increasing_count = 0
            self.plateau_alerted = False  # Reset alert flag
            self.overfit_alerted = False
            flags["improved"] = True
        else:
            # Không cải thiện đáng kể
            self.stale_count += 1

        # Check overfitting: val tăng liên tiếp so với epoch trước
        if len(self.history) >= 1:
            prev_value = self.history[-1][1]
            if prev_value == prev_value:  # không NaN
                if value > prev_value + self.min_delta:
                    self.increasing_count += 1
                else:
                    self.increasing_count = 0

        self.history.append((step, value))

        # Phát hiện plateau (chỉ alert 1 lần)
        if self.stale_count >= self.patience and not self.plateau_alerted:
            flags["is_plateau"] = True
            self.plateau_alerted = True

        return flags

    def get_summary(self) -> str:
        """Tóm tắt trạng thái metric hiện tại."""
        if not self.history:
            return f"{self.name}: chưa có data"

        latest_step, latest_value = self.history[-1]
        return (
            f"{self.name}: {latest_value:.4f} @ step {latest_step} | "
            f"Best: {self.best_value:.4f} @ {self.best_step} | "
            f"Stale: {self.stale_count}/{self.patience}"
        )


# =============================================================================
# MAIN MONITORING LOOP
# =============================================================================

def run_monitor(config: MonitorConfig, logger: logging.Logger):
    """Vòng lặp giám sát chính."""

    log_dir = Path(config.log_dir)
    tensorboard_dir = log_dir / "tensorboard"

    if not log_dir.exists():
        logger.warning(f"log_dir chưa tồn tại: {log_dir}")
        logger.warning("Sẽ chờ training tạo thư mục...")
        # Tạo thư mục để tránh race
        log_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    # --- Reader ---
    reader = TensorBoardReader(tensorboard_dir, logger)

    logger.info(f"Chờ TensorBoard events file tại: {tensorboard_dir}")
    if not reader.wait_for_events_file(max_wait_s=1800):
        logger.error("Timeout — không tìm thấy events file. Thoát.")
        return

    # --- Trackers ---
    primary_tracker = MetricTracker(
        config.primary_metric,
        config.min_delta,
        config.patience,
    )
    secondary_tracker = MetricTracker(
        config.secondary_metric,
        config.min_delta,
        config.patience,
    )

    # --- Trạng thái global ---
    combined_plateau_alerted = False  # Chỉ cảnh báo khi CẢ HAI metrics plateau
    last_progress_epoch = 0           # Epoch cuối cùng đã báo cáo progress

    # --- Notification khởi động ---
    broadcast(
        config,
        title=f"Monitor Started — {config.stage_name}",
        description=(
            f"Bắt đầu giám sát training.\n"
            f"**log_dir**: `{config.log_dir}`\n"
            f"**Patience**: {config.patience} epochs\n"
            f"**Min delta**: {config.min_delta}\n"
            f"**Progress report**: mỗi {config.progress_report_interval} epochs"
        ),
        level="info",
        logger=logger,
    )

    # --- In danh sách tags available (debug) ---
    available_tags = reader.list_available_tags()
    if available_tags:
        logger.info(f"TensorBoard tags có sẵn ({len(available_tags)}):")
        for t in available_tags[:20]:
            logger.info(f"  - {t}")
        if len(available_tags) > 20:
            logger.info(f"  ... và {len(available_tags) - 20} tags khác")

        # Cảnh báo nếu primary/secondary không có
        if config.primary_metric not in available_tags:
            logger.warning(
                f"Primary metric '{config.primary_metric}' CHƯA có trong events file. "
                f"Sẽ chờ..."
            )
        if config.secondary_metric not in available_tags:
            logger.warning(
                f"Secondary metric '{config.secondary_metric}' CHƯA có trong events file. "
                f"Sẽ chờ..."
            )

    # =================================================================
    # MAIN LOOP
    # =================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("  BẮT ĐẦU VÒNG LẶP GIÁM SÁT")
    logger.info(f"  Poll interval: {config.poll_interval_s}s")
    logger.info("  Nhấn Ctrl+C để dừng monitor.")
    logger.info("=" * 60)

    iteration = 0
    try:
        while True:
            iteration += 1
            time.sleep(config.poll_interval_s)

            # Scan metrics mới
            new_data = reader.scan_new_metrics()

            if not new_data:
                logger.info(f"[Poll #{iteration}] Không có data mới.")
                continue

            logger.info(f"[Poll #{iteration}] Nhận {sum(len(v) for v in new_data.values())} entries mới")

            # --- Xử lý primary metric ---
            if config.primary_metric in new_data:
                for step, value, _ in new_data[config.primary_metric]:
                    flags = primary_tracker.update(step, value)
                    _handle_flags(
                        config, logger,
                        tracker=primary_tracker,
                        flags=flags,
                        step=step,
                        value=value,
                    )

            # --- Xử lý secondary metric ---
            if config.secondary_metric in new_data:
                for step, value, _ in new_data[config.secondary_metric]:
                    flags = secondary_tracker.update(step, value)
                    _handle_flags(
                        config, logger,
                        tracker=secondary_tracker,
                        flags=flags,
                        step=step,
                        value=value,
                    )

            # --- Kiểm tra COMBINED PLATEAU (cả 2 metrics đều plateau) ---
            both_plateau = (
                primary_tracker.plateau_alerted and
                secondary_tracker.plateau_alerted
            )

            if both_plateau and not combined_plateau_alerted:
                combined_plateau_alerted = True
                fields = [
                    {
                        "name": f"📉 {config.primary_metric}",
                        "value": f"`{primary_tracker.best_value:.4f}` @ epoch {primary_tracker.best_step}",
                        "inline": True,
                    },
                    {
                        "name": f"📉 {config.secondary_metric}",
                        "value": f"`{secondary_tracker.best_value:.4f}` @ epoch {secondary_tracker.best_step}",
                        "inline": True,
                    },
                    {
                        "name": "🕒 Stale epochs",
                        "value": f"{primary_tracker.stale_count}/{config.patience}",
                        "inline": False,
                    },
                ]
                broadcast(
                    config,
                    title=f"🛑 CẢ 2 METRICS PLATEAU — NÊN DỪNG TRAINING",
                    description=(
                        f"**{config.stage_name}**\n\n"
                        f"CẢ HAI metrics `{config.primary_metric}` và `{config.secondary_metric}` "
                        f"đều đã PLATEAU (không giảm > {config.min_delta} trong "
                        f"{config.patience} epochs liên tiếp).\n\n"
                        f"🔔 **Đề xuất**: Nhấn Ctrl+C để dừng training, "
                        f"rồi chạy stage tiếp theo.\n\n"
                        f"Best checkpoint sẽ tự động load ở stage sau."
                    ),
                    level="critical",
                    logger=logger,
                    fields=fields,
                )

            # --- Progress report định kỳ ---
            latest_epoch = max(
                primary_tracker.history[-1][0] if primary_tracker.history else 0,
                secondary_tracker.history[-1][0] if secondary_tracker.history else 0,
            )

            # Trigger khi latest_epoch vượt qua mốc progress_report_interval tiếp theo
            next_milestone = (last_progress_epoch // config.progress_report_interval + 1) * config.progress_report_interval
            if latest_epoch >= next_milestone and latest_epoch > last_progress_epoch:
                _send_progress_report(
                    config, logger,
                    primary_tracker, secondary_tracker,
                    current_epoch=latest_epoch,
                )
                last_progress_epoch = latest_epoch

    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  MONITOR STOPPED BY USER (Ctrl+C)")
        logger.info("=" * 60)
        broadcast(
            config,
            title="Monitor Stopped",
            description="Monitor đã dừng bởi người dùng (Ctrl+C).",
            level="info",
            logger=logger,
        )


def _handle_flags(
    config: MonitorConfig,
    logger: logging.Logger,
    tracker: MetricTracker,
    flags: Dict[str, bool],
    step: int,
    value: float,
):
    """Xử lý các flag từ tracker.update() — gửi cảnh báo khi cần."""

    # --- NaN/Inf detection (CRITICAL) ---
    if flags["is_nan"]:
        broadcast(
            config,
            title=f"🚨 NaN/Inf DETECTED in {tracker.name}",
            description=(
                f"**{config.stage_name}**\n\n"
                f"Metric `{tracker.name}` = `{value}` tại epoch {step}.\n\n"
                f"Training sẽ HỎNG HOÀN TOÀN nếu tiếp tục.\n\n"
                f"🔔 **Đề xuất**: Ctrl+C NGAY LẬP TỨC.\n"
                f"Kiểm tra: gradient explosion, LR quá cao, data corruption."
            ),
            level="critical",
            logger=logger,
        )
        return

    # --- Log info khi improved ---
    if flags["improved"]:
        logger.info(
            f"  ✓ {tracker.name} improved: {value:.4f} @ epoch {step} "
            f"(prev best: {tracker.best_value:.4f})"
        )
    else:
        logger.info(
            f"  · {tracker.name}: {value:.4f} @ epoch {step} "
            f"(stale: {tracker.stale_count}/{tracker.patience})"
        )

    # --- Overfitting detection ---
    if (
        tracker.increasing_count >= config.overfitting_patience
        and not tracker.overfit_alerted
    ):
        tracker.overfit_alerted = True
        broadcast(
            config,
            title=f"⚠️ OVERFITTING — {tracker.name}",
            description=(
                f"**{config.stage_name}**\n\n"
                f"Metric `{tracker.name}` TĂNG LIÊN TIẾP "
                f"{tracker.increasing_count} epochs.\n\n"
                f"Giá trị hiện tại: `{value:.4f}` @ epoch {step}\n"
                f"Best value: `{tracker.best_value:.4f}` @ epoch {tracker.best_step}\n\n"
                f"🔔 **Đề xuất**: Cân nhắc dừng training, "
                f"rollback về checkpoint best."
            ),
            level="warning",
            logger=logger,
        )


def _send_progress_report(
    config: MonitorConfig,
    logger: logging.Logger,
    primary_tracker: MetricTracker,
    secondary_tracker: MetricTracker,
    current_epoch: int,
):
    """Gửi báo cáo tiến độ định kỳ."""

    def _get_trend(tracker: MetricTracker) -> str:
        """So sánh giá trị hiện tại với N epochs trước → trend."""
        if len(tracker.history) < 2:
            return "—"
        current = tracker.history[-1][1]
        # Lấy giá trị ~N epochs trước
        lookback = min(len(tracker.history) - 1, config.progress_report_interval)
        past = tracker.history[-lookback - 1][1] if lookback > 0 else current
        diff = current - past
        if diff < -config.min_delta:
            return f"📉 giảm ({diff:+.4f})"
        elif diff > config.min_delta:
            return f"📈 tăng ({diff:+.4f})"
        else:
            return f"➡️ ổn định ({diff:+.4f})"

    fields = []

    if primary_tracker.history:
        latest = primary_tracker.history[-1][1]
        fields.append({
            "name": f"{config.primary_metric}",
            "value": (
                f"Current: `{latest:.4f}` | "
                f"Best: `{primary_tracker.best_value:.4f}` @ {primary_tracker.best_step}\n"
                f"Trend: {_get_trend(primary_tracker)} | "
                f"Stale: {primary_tracker.stale_count}/{config.patience}"
            ),
            "inline": False,
        })

    if secondary_tracker.history:
        latest = secondary_tracker.history[-1][1]
        fields.append({
            "name": f"{config.secondary_metric}",
            "value": (
                f"Current: `{latest:.4f}` | "
                f"Best: `{secondary_tracker.best_value:.4f}` @ {secondary_tracker.best_step}\n"
                f"Trend: {_get_trend(secondary_tracker)} | "
                f"Stale: {secondary_tracker.stale_count}/{config.patience}"
            ),
            "inline": False,
        })

    broadcast(
        config,
        title=f"📊 Progress Report — Epoch {current_epoch}",
        description=f"**{config.stage_name}**\n\nTraining đang chạy bình thường.",
        level="progress",
        logger=logger,
        fields=fields,
    )

# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Monitor StyleTTS2 training qua TensorBoard + Discord notifications"
    )
    parser.add_argument(
        "--log-dir", "-l",
        type=str,
        required=True,
        help="Đường dẫn log_dir của stage (chứa thư mục tensorboard/)",
    )
    parser.add_argument(
        "--stage", "-s",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Stage đang chạy (1/2/3)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience cho plateau detection (mặc định: 5)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.001,
        help="Min delta cho plateau detection (mặc định: 0.001)",
    )
    parser.add_argument(
        "--primary-metric",
        type=str,
        default="eval/mel_loss",
        help="Metric chính (mặc định: eval/mel_loss)",
    )
    parser.add_argument(
        "--secondary-metric",
        type=str,
        default="eval/mono_align_loss",
        help="Metric phụ (mặc định: eval/mono_align_loss)",
    )
    parser.add_argument(
        "--progress-report-interval",
        type=int,
        default=10,
        help="Báo cáo tiến độ mỗi N epochs (mặc định: 10)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Khoảng thời gian scan TensorBoard (giây, mặc định: 60)",
    )
    parser.add_argument(
        "--webhook-1",
        type=str,
        default=None,
        help="Override Discord Webhook URL #1 (nếu không dùng .env)",
    )
    parser.add_argument(
        "--webhook-2",
        type=str,
        default=None,
        help="Override Discord Webhook URL #2",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Không gửi Discord thực tế (chỉ print)",
    )
    args = parser.parse_args()

    # --- Load .env ---
    env_candidates = [Path(".env"), Path("../.env"), Path("../../.env"), Path("../../../.env")]
    for ep in env_candidates:
        if ep.exists():
            load_dotenv(str(ep))
            break

    # --- Build config ---
    config = MonitorConfig.from_env()

    config.log_dir = args.log_dir
    config.stage = args.stage
    config.stage_name = {
        1: "Stage 1 - Acoustic & Alignment",
        2: "Stage 2 - Expressive Training",
        3: "Stage 3 - Fine-tune Giọng Bác Ngạn",
    }.get(args.stage, f"Stage {args.stage}")

    config.patience = args.patience
    config.min_delta = args.min_delta
    config.primary_metric = args.primary_metric
    config.secondary_metric = args.secondary_metric
    config.progress_report_interval = args.progress_report_interval
    config.poll_interval_s = args.poll_interval
    config.dry_run = args.dry_run

    if args.webhook_1:
        config.discord_webhook_1 = args.webhook_1
    if args.webhook_2:
        config.discord_webhook_2 = args.webhook_2

    # --- Validate ---
    if not config.dry_run:
        if not config.discord_webhook_1 and not config.discord_webhook_2:
            print("[LỖI] Không có Discord webhook nào!")
            print("  Cách 1: Thêm vào .env:")
            print("    DISCORD_WEBHOOK_1=https://discord.com/api/webhooks/...")
            print("    DISCORD_WEBHOOK_2=https://discord.com/api/webhooks/...")
            print("  Cách 2: Dùng --webhook-1 và --webhook-2")
            print("  Cách 3: Dùng --dry-run để test (không gửi Discord)")
            sys.exit(1)

    # --- Setup logging ---
    log_file = Path(config.log_dir) / config.monitor_log_file
    logger = setup_logging(log_file)

    # --- Header ---
    logger.info("=" * 60)
    logger.info("  MONITOR TRAINING — STYLETTS2")
    logger.info("=" * 60)
    logger.info(f"Log dir              : {config.log_dir}")
    logger.info(f"Stage                : {config.stage} — {config.stage_name}")
    logger.info(f"Primary metric       : {config.primary_metric}")
    logger.info(f"Secondary metric     : {config.secondary_metric}")
    logger.info(f"Patience             : {config.patience} epochs")
    logger.info(f"Min delta            : {config.min_delta}")
    logger.info(f"Progress interval    : {config.progress_report_interval} epochs")
    logger.info(f"Poll interval        : {config.poll_interval_s}s")
    logger.info(f"Webhook #1           : {'✓ đã cấu hình' if config.discord_webhook_1 else '✗ CHƯA CÓ'}")
    logger.info(f"Webhook #2           : {'✓ đã cấu hình' if config.discord_webhook_2 else '✗ CHƯA CÓ'}")
    logger.info(f"Dry run              : {config.dry_run}")

    # --- Run ---
    try:
        run_monitor(config, logger)
    except Exception as e:
        logger.exception(f"Monitor FAILED: {e}")
        broadcast(
            config,
            title="Monitor Crashed",
            description=f"Monitor script gặp lỗi:\n```{str(e)[:500]}```",
            level="critical",
            logger=logger,
        )
        sys.exit(1)

if __name__ == "__main__":
    main()