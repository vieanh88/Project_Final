"""
=============================================================================
  MONITOR TRAINING — Giám sát StyleTTS2 qua TensorBoard + Discord + Wandb
=============================================================================
Mục tiêu: Chạy SONG SONG với train_wrapper.py để:
  1. Đọc TensorBoard events file realtime
  2. Phát hiện plateau của val loss → gợi ý early-stop thủ công
  3. Cảnh báo NaN/Inf loss (training sẽ hỏng)
  4. Cảnh báo overfitting (val loss tăng liên tiếp)
  5. Báo cáo tiến độ mỗi N epochs
  6. Gửi notification qua Discord Webhook tới 2 thiết bị
  7. Stream ALL scalars + alerts sang Wandb (3 namespace với custom x-axis)
  8. Print ra console

CHÚ Ý: Script CHỈ cảnh báo, KHÔNG tự dừng training.
        Bạn tự quyết định Ctrl+C khi nhận notification.

Setup Discord Webhook:
  1. Discord → Server Settings → Integrations → Webhooks → New Webhook
  2. Copy URL dạng: https://discord.com/api/webhooks/XXXXX/YYYYY
  3. Paste vào .env hoặc CLI flag

Setup Wandb (tự động enable khi có env var):
  export WANDB_API_KEY=your_api_key_here
  # hoặc thêm WANDB_API_KEY=... vào .env

3 NAMESPACE METRICS với custom x-axis (wandb.define_metric):
  - train/*  → step_metric = train/iter   (số iter trong epoch)
  - eval/*   → step_metric = eval/epoch   (epoch number)
  - monitor/* → step_metric = monitor/poll (số lần poll script)
KHÔNG truyền step= cho wandb.log — wandb tự dùng step metric đã define.

Chạy lệnh (Terminal 2 — song song với train):
    # Cả Discord + Wandb (default, nếu có cả 2 secrets)
    python monitor_training.py --log-dir "Models/VietnameseBase"

    # Chỉ Wandb (tắt Discord)
    python monitor_training.py --log-dir "..." --no-discord

    # Chỉ Discord (không có WANDB_API_KEY → wandb tự skip)
    python monitor_training.py --log-dir "..."

    # Resume wandb run cũ
    python monitor_training.py --log-dir "..." --wandb-resume <run_id>

    # Dry-run (không gửi gì, chỉ print)
    python monitor_training.py --log-dir "..." --dry-run

Tagging metrics ghi bởi train_*.py (đã verify từ source):
  STAGE 1 (train_first.py):
    train/mel_loss, train/gen_loss, train/d_loss,
    train/mono_loss, train/s2s_loss, train/slm_loss
    eval/mel_loss
  STAGE 2 (train_second.py):
    train/mel_loss, train/gen_loss, train/d_loss,
    train/ce_loss, train/dur_loss, train/slm_loss,
    train/norm_loss, train/F0_loss, train/sty_loss,
    train/diff_loss, train/d_loss_slm, train/gen_loss_slm
    eval/mel_loss, eval/dur_loss, eval/F0_loss
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
from datetime import datetime, timezone

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

# =============================================================================
# WANDB CONFIGURATION TEMPLATES — USER CUSTOMIZE
# =============================================================================
# Run name template — sẽ được format với các biến:
#   {stage_short} : "stage1" / "stage2" / "stage3"
#   {stage_name}  : "Stage 1 - Acoustic & Alignment" / etc.
#   {timestamp}   : "20260515_163022" (year+month+day_hour+min+sec)
# VÍ DỤ output: "stage1_20260515_163022"
# Bạn có thể override bằng --wandb-run-name <custom_name> trên CLI
# Hoặc sửa template dưới đây để có format khác:
WANDB_RUN_NAME_TEMPLATE = "{stage_short}_{timestamp}"  # ← USER CUSTOMIZE

# Default tags để filter/search runs trên wandb dashboard.
# Tags stage-specific (stage1/stage2/stage3) sẽ TỰ ĐỘNG thêm vào.
# VÍ DỤ TAGS HỮU ÍCH:
#   "styletts2"     — model name
#   "vietnamese"    — language
#   "vivoice"       — dataset name
#   "ngan"          — Bác Ngạn (chỉ Stage 3)
#   "rtx4080s"      — GPU name
#   "vastai"        — platform
# Bạn có thể override bằng --wandb-tags "tag1,tag2" trên CLI
WANDB_DEFAULT_TAGS = ["styletts2", "vietnamese", "vivoice"]  # ← USER CUSTOMIZE

# Wandb project name mặc định — override bằng --wandb-project
WANDB_DEFAULT_PROJECT = "story-ai-narrator"  # ← USER CUSTOMIZE


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
    # Tags THỰC TẾ ghi bởi train_first.py / train_second.py / train_finetune.py:
    #
    #   STAGE 1 (train_first.py):
    #     train/mel_loss, train/gen_loss, train/d_loss,
    #     train/mono_loss, train/s2s_loss, train/slm_loss
    #     eval/mel_loss   ← CHỈ CÓ 1 eval tag
    #
    #   STAGE 2 (train_second.py):
    #     train/mel_loss, train/gen_loss, train/d_loss,
    #     train/ce_loss, train/dur_loss, train/slm_loss,
    #     train/norm_loss, train/F0_loss, train/sty_loss,
    #     train/diff_loss, train/d_loss_slm, train/gen_loss_slm
    #     eval/mel_loss, eval/dur_loss, eval/F0_loss
    #
    #   STAGE 3 (train_finetune.py): tương tự Stage 2
    #
    # PRIMARY: luôn dùng eval/mel_loss (loss chính, có ở mọi stage)
    # SECONDARY: được auto-set theo stage trong main() (None để trigger logic):
    #   Stage 1 → train/mel_loss (vì stage 1 chỉ có 1 eval tag)
    #   Stage 2/3 → eval/dur_loss (loss expressive quan trọng)
    primary_metric: str = "eval/mel_loss"
    secondary_metric: Optional[str] = None  # auto-set trong main() theo stage

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
    no_discord: bool = False           # Skip Discord notifications (chỉ wandb)

    # --- Wandb ---
    # KÍCH HOẠT TỰ ĐỘNG khi có env var WANDB_API_KEY (config sẽ check trong main())
    wandb_enabled: bool = False
    wandb_project: str = WANDB_DEFAULT_PROJECT
    wandb_run_name: str = ""           # Auto-gen từ template nếu rỗng
    wandb_resume_id: str = ""          # Run ID để resume, rỗng = tạo run mới
    wandb_tags: List[str] = field(default_factory=lambda: list(WANDB_DEFAULT_TAGS))

    # --- Tùy chọn ---
    dry_run: bool = False              # Không thực sự gửi Discord/wandb (chỉ print)

    # --- Log file cho monitor ---
    monitor_log_file: str = "monitor_training.log"

    @classmethod
    def from_env(cls) -> "MonitorConfig":
        """Load secrets từ .env / environment."""
        config = cls()
        config.discord_webhook_1 = os.environ.get("DISCORD_WEBHOOK_1", "")
        config.discord_webhook_2 = os.environ.get("DISCORD_WEBHOOK_2", "")
        # Wandb: enable tự động nếu có WANDB_API_KEY
        config.wandb_enabled = bool(os.environ.get("WANDB_API_KEY", "").strip())
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

# =============================================================================
# WANDB MANAGER
# =============================================================================

class WandbManager:
    """
    Encapsulate toàn bộ wandb logic. Lazy import (chỉ import khi enabled).

    3 NAMESPACE METRICS với custom x-axis (define_metric):
      - train/* → step_metric = train/iter   (số iter, ~50k/epoch ở Stage 1)
      - eval/*  → step_metric = eval/epoch   (epoch number, 1-30)
      - monitor/* → step_metric = monitor/poll (số lần poll script)

    KHÔNG truyền step= cho wandb.log() — wandb tự dùng custom step metric.

    Lifecycle:
      __init__: chỉ lưu config, KHÔNG init wandb
      init(): thật sự init wandb run + define_metric
      log_scalars(): log batch metrics
      alert(): trigger wandb.alert + custom monitor scalars
      finish(): cleanup (gọi từ KeyboardInterrupt handler)
    """

    def __init__(self, config: "MonitorConfig", logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.run = None
        self.wandb_module = None  # lazy import
        self.poll_counter = 0  # số lần đã log monitor scalars

        # Track: tag train đã có entry chưa (để biết khi nào cần define_metric)
        # Wandb yêu cầu định nghĩa step_metric TRƯỚC khi metrics dùng nó được log.
        # Vì ta đã define_metric với glob "train/*" / "eval/*" / "monitor/*",
        # không cần track thêm.
        self._initialized = False

    def init(self) -> bool:
        """
        Init wandb run. Trả về True nếu thành công, False nếu fail/disabled.
        Lỗi ở wandb KHÔNG được crash monitor.
        """
        if not self.config.wandb_enabled:
            self.logger.info("Wandb: DISABLED (không có WANDB_API_KEY env var)")
            return False

        if self.config.dry_run:
            self.logger.info("Wandb: DRY-RUN mode (skip init)")
            return False

        # Lazy import
        try:
            import wandb
            self.wandb_module = wandb
        except ImportError:
            self.logger.warning(
                "Wandb: package chưa cài. Cài bằng: pip install wandb"
            )
            return False

        # Build run name
        run_name = self.config.wandb_run_name
        if not run_name:
            stage_short = f"stage{self.config.stage}"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                run_name = WANDB_RUN_NAME_TEMPLATE.format(
                    stage_short=stage_short,
                    stage_name=self.config.stage_name,
                    timestamp=timestamp,
                )
            except KeyError as e:
                self.logger.warning(
                    f"Wandb: run_name template có biến không hợp lệ: {e}. "
                    f"Dùng fallback name."
                )
                run_name = f"{stage_short}_{timestamp}"

        # Tags: default + stage-specific
        tags = list(self.config.wandb_tags)
        stage_tag = f"stage{self.config.stage}"
        if stage_tag not in tags:
            tags.append(stage_tag)

        try:
            init_kwargs = {
                "project": self.config.wandb_project,
                "name": run_name,
                "tags": tags,
                "config": {
                    "stage": self.config.stage,
                    "stage_name": self.config.stage_name,
                    "primary_metric": self.config.primary_metric,
                    "secondary_metric": self.config.secondary_metric,
                    "patience": self.config.patience,
                    "min_delta": self.config.min_delta,
                    "log_dir": self.config.log_dir,
                },
            }
            if self.config.wandb_resume_id:
                init_kwargs["id"] = self.config.wandb_resume_id
                init_kwargs["resume"] = "must"
                self.logger.info(
                    f"Wandb: resume run_id={self.config.wandb_resume_id}"
                )

            self.run = self.wandb_module.init(**init_kwargs)
        except Exception as e:
            self.logger.warning(f"Wandb init thất bại: {e}. Tiếp tục không có wandb.")
            self.run = None
            return False

        # === DEFINE METRICS — 3 namespace với custom x-axis ===
        # Pattern: define step metric trước, sau đó bind glob pattern
        try:
            # Namespace 1: TRAIN (step_metric = train/iter)
            self.wandb_module.define_metric("train/iter")
            self.wandb_module.define_metric("train/*", step_metric="train/iter")

            # Namespace 2: EVAL (step_metric = eval/epoch)
            self.wandb_module.define_metric("eval/epoch")
            self.wandb_module.define_metric("eval/*", step_metric="eval/epoch")

            # Namespace 3: MONITOR (step_metric = monitor/poll)
            self.wandb_module.define_metric("monitor/poll")
            self.wandb_module.define_metric("monitor/*", step_metric="monitor/poll")
        except Exception as e:
            self.logger.warning(f"Wandb define_metric thất bại: {e}")
            # Vẫn tiếp tục — không define_metric thì wandb dùng step ngầm định

        self._initialized = True
        self.logger.info(
            f"Wandb: ✓ run={run_name} project={self.config.wandb_project} "
            f"tags={tags}"
        )
        if self.run is not None and hasattr(self.run, "url"):
            self.logger.info(f"Wandb URL: {self.run.url}")
        return True

    def log_scalars(self, new_data: Dict[str, List]):
        """
        Log batch scalars sang wandb.

        Args:
            new_data: dict {tag_name: List[(step, value, wall_time)]}
                      Format giống output của TensorBoardReader.scan_new_metrics()

        Logic:
          - train/*  → log mỗi entry, kèm "train/iter" = step
          - eval/*   → log mỗi entry, kèm "eval/epoch" = step
          - khác     → bỏ qua (chỉ track namespaces đã define)

        KHÔNG truyền step= (wandb sẽ dùng custom step metric đã define).
        """
        if not self._initialized or self.run is None:
            return

        # Mỗi entry phải log riêng vì step KHÁC NHAU.
        # Log batch theo từng entry → wandb tạo điểm chart chính xác.
        for tag, entries in new_data.items():
            for step, value, _wall_time in entries:
                payload = {tag: value}

                # Bind step vào đúng namespace
                if tag.startswith("train/"):
                    payload["train/iter"] = step
                elif tag.startswith("eval/"):
                    payload["eval/epoch"] = step
                else:
                    # Tag không phải train/* hay eval/* — vẫn log nhưng không có x-axis riêng
                    pass

                try:
                    self.wandb_module.log(payload)
                except Exception as e:
                    self.logger.warning(f"Wandb log thất bại cho {tag}: {e}")
                    return  # tránh spam warning với cùng error

    def log_monitor_state(
        self,
        primary_tracker: "MetricTracker",
        secondary_tracker: "MetricTracker",
    ):
        """
        Log các "monitor/*" custom scalars: best_value, stale_count, increasing_count.

        Trigger 1 lần/poll, dùng poll_counter làm step.
        """
        if not self._initialized or self.run is None:
            return

        self.poll_counter += 1
        payload = {"monitor/poll": self.poll_counter}

        for tracker in (primary_tracker, secondary_tracker):
            if not tracker.history:
                continue
            # Tên metric: dùng tag gốc + suffix
            # Ví dụ: monitor/eval_mel_loss_best, monitor/eval_mel_loss_stale
            safe_name = tracker.name.replace("/", "_")
            if tracker.best_value is not None:
                payload[f"monitor/{safe_name}_best"] = tracker.best_value
            payload[f"monitor/{safe_name}_stale"] = tracker.stale_count
            payload[f"monitor/{safe_name}_increasing"] = tracker.increasing_count

        try:
            self.wandb_module.log(payload)
        except Exception as e:
            self.logger.warning(f"Wandb log_monitor_state thất bại: {e}")

    def alert(self, title: str, text: str, level: str):
        """
        Trigger wandb.alert (gửi email tới owner của wandb account).

        Level mapping:
          critical → wandb.AlertLevel.ERROR
          warning  → wandb.AlertLevel.WARN
          info/progress → wandb.AlertLevel.INFO

        Rate limit: 30 emails/run/24h (theo wandb docs). Plateau alert có
        flag `plateau_alerted=True` để tránh spam → KHÔNG chạm limit.
        """
        if not self._initialized or self.run is None:
            return

        try:
            level_map = {
                "critical": self.wandb_module.AlertLevel.ERROR,
                "warning": self.wandb_module.AlertLevel.WARN,
                "info": self.wandb_module.AlertLevel.INFO,
                "progress": self.wandb_module.AlertLevel.INFO,
            }
            wandb_level = level_map.get(level, self.wandb_module.AlertLevel.INFO)
            self.wandb_module.alert(title=title, text=text, level=wandb_level)
        except Exception as e:
            # Không spam warning cho mọi alert; chỉ log debug
            self.logger.debug(f"Wandb alert thất bại: {e}")

    def finish(self, exit_code: int = 0):
        """Cleanup wandb run. Gọi từ KeyboardInterrupt handler hoặc end-of-main."""
        if not self._initialized or self.run is None:
            return
        try:
            self.wandb_module.finish(exit_code=exit_code)
            self.logger.info("Wandb: run finished cleanly.")
        except Exception as e:
            self.logger.warning(f"Wandb finish thất bại: {e}")


# =============================================================================
# DISCORD NOTIFICATION
# =============================================================================
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
    wandb_mgr: Optional["WandbManager"] = None,
):
    """
    Gửi notification tới: console (luôn) + Discord (nếu có) + wandb alert (nếu có).

    Behavior matrix:
      dry_run = True       → chỉ console, KHÔNG gửi Discord/wandb
      no_discord = True    → console + wandb (skip Discord)
      no_discord = False   → console + Discord + wandb
    """
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

    # Dry-run: skip cả Discord và wandb
    if config.dry_run:
        logger.info("  [DRY RUN] Không gửi Discord/wandb thực tế.")
        return

    # Discord (trừ khi --no-discord)
    if not config.no_discord:
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
        logger.debug("  Discord: SKIPPED (--no-discord)")

    # Wandb alert (nếu wandb được init)
    if wandb_mgr is not None:
        # Build text cho wandb alert (gộp description + fields)
        wandb_text = description
        if fields:
            wandb_text += "\n\n"
            for f in fields:
                wandb_text += f"• {f['name']}: {f['value']}\n"
        wandb_mgr.alert(title=title, text=wandb_text, level=level)


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

    # === WANDB INIT (lazy — chỉ init nếu enabled) ===
    wandb_mgr = WandbManager(config, logger)
    wandb_mgr.init()  # silent fail nếu wandb không enabled hoặc lỗi

    # --- Reader ---
    reader = TensorBoardReader(tensorboard_dir, logger)

    logger.info(f"Chờ TensorBoard events file tại: {tensorboard_dir}")
    if not reader.wait_for_events_file(max_wait_s=1800):
        logger.error("Timeout — không tìm thấy events file. Thoát.")
        wandb_mgr.finish(exit_code=1)
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
            f"**Primary**: `{config.primary_metric}`\n"
            f"**Secondary**: `{config.secondary_metric}`\n"
            f"**Patience**: {config.patience} epochs\n"
            f"**Min delta**: {config.min_delta}\n"
            f"**Progress report**: mỗi {config.progress_report_interval} epochs\n"
            f"**Wandb**: {'✓ ENABLED' if wandb_mgr._initialized else '✗ disabled'}"
        ),
        level="info",
        logger=logger,
        wandb_mgr=wandb_mgr,
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
    exit_code = 0
    try:
        while True:
            iteration += 1
            time.sleep(config.poll_interval_s)

            # Scan metrics mới
            new_data = reader.scan_new_metrics()

            if not new_data:
                logger.info(f"[Poll #{iteration}] Không có data mới.")
                continue

            logger.info(
                f"[Poll #{iteration}] Nhận {sum(len(v) for v in new_data.values())} "
                f"entries mới ({len(new_data)} tags)"
            )

            # === WANDB: Log TẤT CẢ scalars mới (train/* và eval/*) ===
            # Chạy đầu tiên trong poll để wandb cập nhật REALTIME, không bị
            # delay bởi xử lý plateau/overfitting bên dưới.
            wandb_mgr.log_scalars(new_data)

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
                        wandb_mgr=wandb_mgr,
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
                        wandb_mgr=wandb_mgr,
                    )

            # === WANDB: Log monitor/* state (best, stale, increasing) ===
            wandb_mgr.log_monitor_state(primary_tracker, secondary_tracker)

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
                    wandb_mgr=wandb_mgr,
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
                    wandb_mgr=wandb_mgr,
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
            wandb_mgr=wandb_mgr,
        )
    except Exception:
        # Re-raise sau khi finish wandb với exit_code != 0
        exit_code = 1
        wandb_mgr.finish(exit_code=exit_code)
        raise
    finally:
        wandb_mgr.finish(exit_code=exit_code)


def _handle_flags(
    config: MonitorConfig,
    logger: logging.Logger,
    tracker: MetricTracker,
    flags: Dict[str, bool],
    step: int,
    value: float,
    wandb_mgr: Optional["WandbManager"] = None,
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
            wandb_mgr=wandb_mgr,
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

    # --- BUG 2 FIX: Plateau detection (PER-METRIC alert) ---
    # Trước đây: tracker set flag is_plateau=True nhưng nobody trigger broadcast
    # Giờ: trigger alert ngay khi metric cụ thể plateau (chưa cần CẢ HAI plateau)
    if flags["is_plateau"]:
        broadcast(
            config,
            title=f"📉 PLATEAU — {tracker.name}",
            description=(
                f"**{config.stage_name}**\n\n"
                f"Metric `{tracker.name}` đã PLATEAU "
                f"(không giảm > {config.min_delta} trong "
                f"{tracker.patience} epochs liên tiếp).\n\n"
                f"Giá trị hiện tại: `{value:.4f}` @ epoch {step}\n"
                f"Best value: `{tracker.best_value:.4f}` @ epoch {tracker.best_step}\n\n"
                f"🔔 **Đề xuất**: chờ thêm metric thứ 2 plateau "
                f"trước khi dừng (combined alert sẽ trigger sau)."
            ),
            level="warning",
            logger=logger,
            wandb_mgr=wandb_mgr,
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
            wandb_mgr=wandb_mgr,
        )


def _send_progress_report(
    config: MonitorConfig,
    logger: logging.Logger,
    primary_tracker: MetricTracker,
    secondary_tracker: MetricTracker,
    current_epoch: int,
    wandb_mgr: Optional["WandbManager"] = None,
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
        wandb_mgr=wandb_mgr,
    )

# MAIN
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Monitor StyleTTS2 training qua TensorBoard + Discord + wandb. "
            "Wandb tự động enable nếu có WANDB_API_KEY env var."
        )
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
        default=None,
        help=(
            "Metric phụ. Mặc định auto-set theo stage: "
            "Stage 1 → 'train/mel_loss', Stage 2/3 → 'eval/dur_loss'"
        ),
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

    # --- Discord ---
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
        "--no-discord",
        action="store_true",
        help=(
            "TẮT Discord notifications (vẫn dùng wandb nếu enabled). "
            "Hữu ích khi chỉ muốn dùng wandb thuần."
        ),
    )

    # --- Wandb ---
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help=f"Wandb project name (mặc định: '{WANDB_DEFAULT_PROJECT}')",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help=(
            "Wandb run name. Mặc định auto-gen từ template "
            f"'{WANDB_RUN_NAME_TEMPLATE}' (xem WANDB_RUN_NAME_TEMPLATE trong file)."
        ),
    )
    parser.add_argument(
        "--wandb-resume",
        type=str,
        default=None,
        metavar="RUN_ID",
        help="Resume wandb run cụ thể bằng run_id (mặc định: tạo run mới)",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default=None,
        help=(
            "Tags cho wandb run, format CSV. "
            f"VÍ DỤ: --wandb-tags 'rtx4080s,vastai'. "
            f"Default tags: {WANDB_DEFAULT_TAGS} (luôn thêm 'stage{{N}}')."
        ),
    )

    # --- Tùy chọn khác ---
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Không gửi Discord/wandb thực tế (chỉ print ra console)",
    )
    args = parser.parse_args()

    # --- Load .env ---
    env_candidates = [Path(".env"), Path("../.env"), Path("../../.env"), Path("../../../.env")]
    for ep in env_candidates:
        if ep.exists():
            load_dotenv(str(ep))
            break

    # --- Build config ---
    config = MonitorConfig.from_env()  # đã set wandb_enabled nếu có WANDB_API_KEY

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
    config.progress_report_interval = args.progress_report_interval
    config.poll_interval_s = args.poll_interval
    config.dry_run = args.dry_run

    # === BUG 1 FIX: Auto-set secondary_metric theo stage ===
    # Tag thật của StyleTTS2 (đã verify):
    #   Stage 1 (train_first.py)  → eval/* CHỈ có 'eval/mel_loss' duy nhất
    #     → secondary = 'train/mel_loss' (smooth train mel loss để check trend)
    #   Stage 2 (train_second.py) → eval/* có mel_loss, dur_loss, F0_loss
    #     → secondary = 'eval/dur_loss' (expressive metric quan trọng)
    #   Stage 3 (train_finetune.py) → kế thừa từ train_second
    #     → secondary = 'eval/dur_loss'
    if args.secondary_metric is None:
        if args.stage == 1:
            config.secondary_metric = "train/mel_loss"
        else:  # Stage 2, 3
            config.secondary_metric = "eval/dur_loss"
    else:
        config.secondary_metric = args.secondary_metric

    # Discord overrides
    if args.webhook_1:
        config.discord_webhook_1 = args.webhook_1
    if args.webhook_2:
        config.discord_webhook_2 = args.webhook_2
    config.no_discord = args.no_discord

    # Wandb overrides
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.wandb_run_name:
        config.wandb_run_name = args.wandb_run_name
    if args.wandb_resume:
        config.wandb_resume_id = args.wandb_resume
    if args.wandb_tags:
        config.wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

    # --- Validate: phải có ít nhất 1 kênh notification (trừ khi dry-run) ---
    if not config.dry_run:
        has_discord = (
            not config.no_discord
            and (config.discord_webhook_1 or config.discord_webhook_2)
        )
        has_wandb = config.wandb_enabled
        if not has_discord and not has_wandb:
            print("[LỖI] Không có kênh notification nào!")
            print("  Bạn cần CÍT NHẤT 1 trong các option:")
            print("  ────────────────────────────────────────")
            print("  Option A — Wandb (KHUYẾN NGHỊ):")
            print("    export WANDB_API_KEY=your_key_here")
            print("    Hoặc thêm WANDB_API_KEY=... vào .env")
            print("  ────────────────────────────────────────")
            print("  Option B — Discord:")
            print("    export DISCORD_WEBHOOK_1=https://discord.com/api/webhooks/...")
            print("    Hoặc dùng --webhook-1 / --webhook-2")
            print("  ────────────────────────────────────────")
            print("  Option C — Dry-run (test, không gửi gì):")
            print("    python monitor_training.py --dry-run ...")
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
    logger.info(f"Discord              : {'OFF (--no-discord)' if config.no_discord else 'ON'}")
    if not config.no_discord:
        logger.info(f"  Webhook #1         : {'✓ đã cấu hình' if config.discord_webhook_1 else '✗ CHƯA CÓ'}")
        logger.info(f"  Webhook #2         : {'✓ đã cấu hình' if config.discord_webhook_2 else '✗ CHƯA CÓ'}")
    logger.info(f"Wandb                : {'✓ ENABLED' if config.wandb_enabled else '✗ disabled (no WANDB_API_KEY)'}")
    if config.wandb_enabled:
        logger.info(f"  Project            : {config.wandb_project}")
        logger.info(f"  Run name           : {config.wandb_run_name or '(auto-gen từ template)'}")
        logger.info(f"  Resume run_id      : {config.wandb_resume_id or '(N/A - tạo run mới)'}")
        logger.info(f"  Tags (default)     : {config.wandb_tags}")
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
            wandb_mgr=None,  # wandb_mgr đã được run_monitor finish trong except block
        )
        sys.exit(1)

if __name__ == "__main__":
    main()