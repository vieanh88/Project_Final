"""
=============================================================================
  NLP GENERATOR — Phase 1: Đạo diễn Kịch bản (Module NLP với Qwen)
=============================================================================
Mục tiêu: Đọc kịch bản truyện ma (.txt), gửi tới Qwen3.5-27B-AWQ đang
          chạy qua vLLM local API, nhận về JSON array chứa:
            - id: thứ tự câu
            - role: "narrator" hoặc "character"
            - text: nội dung câu thoại
            - pause_after_ms: khoảng lặng kinh dị (ms)

          Lưu kết quả ra file script.json cho Phase 2 (TTS).

Quy trình:
  1. [Terminal 1] Khởi động vLLM server (thủ công):
     python -m vllm.entrypoints.openai.api_server \
         --model QuantTrio/Qwen3.5-27B-AWQ \
         --quantization awq \
         --max-model-len 4096 \
         --gpu-memory-utilization 0.8 \
         --port 8000

  2. [Terminal 2] Chạy script này:
     python nlp_generator.py --input "truyen_ma.txt"

  3. Sau khi có script.json → TẮT Terminal 1 (Ctrl+C) để giải phóng VRAM

Chạy lệnh:
    python nlp_generator.py --input "story.txt"
    python nlp_generator.py --input "story.txt" --output "script.json"
    python nlp_generator.py --input "story.txt" --api-url "http://localhost:8000"
    python nlp_generator.py --config config.yaml
=============================================================================
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import yaml
from dotenv import load_dotenv

# =============================================================================
# KHẮC PHỤC LỖI ENCODING TRÊN WINDOWS
# =============================================================================
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"


# =============================================================================
# CONFIGURATION
# =============================================================================

MASTER_PROMPT = """Bạn là trợ lý AI chuyên phân tích kịch bản truyện ma tiếng Việt. Nhiệm vụ của bạn là xử lý văn bản thô và trích xuất thành MẢNG JSON.

Quy tắc:
1. Phân đoạn: Cắt đoạn dài thành các câu ngắn phù hợp (10-20 từ) để nhịp thở tự nhiên.
2. Khoảng lặng (pause_after_ms), tuân theo các quy tắc để tạo không khí kinh dị:
   - Dấu phẩy (,): 200 - 300 ms.
   - Dấu (. ? !): 500 - 800 ms.
   - Chuyển cảnh, cao trào, rùng rợn: 1500 - 2500 ms.
3. Vai trò (role): "narrator" (lời dẫn) hoặc "character" (lời thoại).

Output BẮT BUỘC CHỈ LÀ MỘT MẢNG JSON.
Định dạng object: {"id": int, "role": "string", "text": "string", "pause_after_ms": int}
KHÔNG in ra markdown, KHÔNG giải thích, KHÔNG bọc trong ```json```."""


@dataclass
class NLPConfig:
    """Cấu hình cho NLP Generator."""

    # File input (kịch bản truyện ma .txt)
    input_file: str = ""

    # File output
    output_file: str = "script.json"

    # vLLM API
    api_url: str = "http://localhost:8000"
    model_name: str = "QuantTrio/Qwen3.5-27B-AWQ"

    # Generation params
    max_tokens: int = 4096
    temperature: float = 0.3       # Thấp → output nhất quán, ít sáng tạo ngẫu nhiên
    top_p: float = 0.9

    # Chunk size (số ký tự tối đa mỗi lần gửi API)
    # Tránh vượt context window 4096 tokens (~12000 ký tự tiếng Việt)
    chunk_size: int = 6000

    # Retry
    max_retries: int = 3
    retry_delay: float = 5.0

    # Work dir (log)
    work_dir: str = "./workdir"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "NLPConfig":
        """Load config từ file YAML."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        nlp = full_config.get("nlp_generator", {})

        config = cls()
        config.input_file = nlp.get("input_file", config.input_file)
        config.output_file = nlp.get("output_file", config.output_file)
        config.api_url = nlp.get("api_url", config.api_url)
        config.model_name = nlp.get("model_name", config.model_name)
        config.max_tokens = nlp.get("max_tokens", config.max_tokens)
        config.temperature = nlp.get("temperature", config.temperature)
        config.chunk_size = nlp.get("chunk_size", config.chunk_size)
        config.work_dir = nlp.get("work_dir", full_config.get("paths", {}).get("work_dir", config.work_dir))

        return config


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "nlp_generator.log"

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), encoding="utf-8", mode="w"),
        ],
    )
    return logging.getLogger("nlp_generator")


# =============================================================================
# TEXT CHUNKING
# =============================================================================

def chunk_text(text: str, chunk_size: int) -> list:
    """
    Tách văn bản dài thành nhiều chunk nhỏ hơn.
    Ưu tiên tách tại dấu xuống dòng kép (paragraph break) hoặc dấu chấm.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    remaining = text

    while len(remaining) > chunk_size:
        # Tìm điểm cắt tốt nhất trong phạm vi chunk_size
        cut_point = chunk_size

        # Ưu tiên 1: Paragraph break (\n\n)
        para_break = remaining.rfind("\n\n", 0, chunk_size)
        if para_break > chunk_size * 0.3:
            cut_point = para_break + 2
        else:
            # Ưu tiên 2: Dấu chấm câu + space
            for delim in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                pos = remaining.rfind(delim, 0, chunk_size)
                if pos > chunk_size * 0.3:
                    cut_point = pos + len(delim)
                    break
            else:
                # Ưu tiên 3: Xuống dòng đơn
                newline = remaining.rfind("\n", 0, chunk_size)
                if newline > chunk_size * 0.3:
                    cut_point = newline + 1

        chunk = remaining[:cut_point].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut_point:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


# =============================================================================
# API COMMUNICATION
# =============================================================================

def check_api_health(api_url: str, logger: logging.Logger) -> bool:
    """Kiểm tra vLLM server có đang chạy không."""
    import requests

    try:
        resp = requests.get(f"{api_url}/v1/models", timeout=10)
        if resp.status_code == 200:
            models = resp.json()
            logger.info(f"vLLM server OK. Models: {models}")
            return True
        else:
            logger.error(f"vLLM server trả về status {resp.status_code}")
            return False
    except requests.ConnectionError:
        logger.error(f"Không kết nối được tới vLLM server tại {api_url}")
        logger.error("Hãy khởi động vLLM server trước (xem hướng dẫn ở đầu file)")
        return False
    except Exception as e:
        logger.error(f"Lỗi kiểm tra API: {e}")
        return False


def call_qwen_api(
    text_chunk: str,
    config: NLPConfig,
    chunk_index: int,
    total_chunks: int,
    logger: logging.Logger,
) -> list:
    """
    Gửi 1 chunk text tới Qwen API, nhận về list of dicts.
    Có retry logic nếu JSON parse thất bại.
    """
    import requests

    endpoint = f"{config.api_url}/v1/chat/completions"

    payload = {
        "model": config.model_name,
        "messages": [
            {"role": "system", "content": MASTER_PROMPT},
            {"role": "user", "content": text_chunk},
        ],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }

    for attempt in range(1, config.max_retries + 1):
        try:
            logger.info(
                f"  Chunk {chunk_index}/{total_chunks} | "
                f"Attempt {attempt}/{config.max_retries} | "
                f"{len(text_chunk)} chars..."
            )

            resp = requests.post(endpoint, json=payload, timeout=300)
            resp.raise_for_status()

            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()

            # Parse JSON — xử lý trường hợp Qwen bọc trong ```json```
            content = _clean_json_response(content)
            items = json.loads(content)

            if not isinstance(items, list):
                raise ValueError(f"Expected JSON array, got {type(items).__name__}")

            # Validate structure
            validated = []
            for item in items:
                validated.append({
                    "id": item.get("id", 0),
                    "role": item.get("role", "narrator"),
                    "text": str(item.get("text", "")).strip(),
                    "pause_after_ms": int(item.get("pause_after_ms", 500)),
                })

            # Lọc bỏ item text rỗng
            validated = [v for v in validated if v["text"]]

            logger.info(f"    → Nhận {len(validated)} items")
            return validated

        except json.JSONDecodeError as e:
            logger.warning(f"    JSON parse error: {e}")
            logger.warning(f"    Raw response (500 chars): {content[:500]}")
            if attempt < config.max_retries:
                logger.info(f"    Retry sau {config.retry_delay}s...")
                time.sleep(config.retry_delay)

        except requests.RequestException as e:
            logger.warning(f"    Request error: {e}")
            if attempt < config.max_retries:
                logger.info(f"    Retry sau {config.retry_delay}s...")
                time.sleep(config.retry_delay)

        except Exception as e:
            logger.warning(f"    Unexpected error: {e}")
            if attempt < config.max_retries:
                time.sleep(config.retry_delay)

    logger.error(f"  Chunk {chunk_index}: THẤT BẠI sau {config.max_retries} lần thử!")
    return []


def _clean_json_response(content: str) -> str:
    """
    Làm sạch response từ LLM:
    - Xóa markdown code block (```json ... ```)
    - Xóa text thừa trước/sau JSON array
    """
    # Xóa markdown fence
    content = content.strip()
    if content.startswith("```"):
        # Tìm dòng đầu tiên sau ```
        first_newline = content.find("\n")
        if first_newline > 0:
            content = content[first_newline + 1:]
        # Xóa ``` cuối
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    # Tìm JSON array trong content (bắt đầu bằng [ và kết thúc bằng ])
    start = content.find("[")
    end = content.rfind("]")
    if start >= 0 and end > start:
        content = content[start:end + 1]

    return content


# =============================================================================
# CORE LOGIC
# =============================================================================

def generate_script(config: NLPConfig, logger: logging.Logger):
    """
    Quy trình chính:
    1. Đọc file truyện ma
    2. Chunk text
    3. Gửi từng chunk tới Qwen API
    4. Gộp kết quả, re-index IDs
    5. Lưu script.json
    """
    import requests  # noqa: F811 — verify import sớm

    input_path = Path(config.input_file)
    output_path = Path(config.output_file)

    # --- Đọc file input ---
    if not input_path.exists():
        logger.error(f"Không tìm thấy file input: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    logger.info(f"Đọc file: {input_path.name} ({len(raw_text):,} ký tự)")

    # --- Kiểm tra API ---
    logger.info(f"Kiểm tra vLLM server tại {config.api_url}...")
    if not check_api_health(config.api_url, logger):
        logger.error("")
        logger.error("vLLM server chưa chạy! Khởi động bằng lệnh:")
        logger.error("  python -m vllm.entrypoints.openai.api_server \\")
        logger.error(f"      --model {config.model_name} \\")
        logger.error("      --quantization awq \\")
        logger.error("      --max-model-len 4096 \\")
        logger.error("      --gpu-memory-utilization 0.8 \\")
        logger.error("      --port 8000")
        return

    # --- Chunk text ---
    chunks = chunk_text(raw_text, config.chunk_size)
    logger.info(f"Tách thành {len(chunks)} chunk(s)")

    # --- Gửi API ---
    start_time = time.time()
    all_items = []

    for i, chunk in enumerate(chunks, 1):
        items = call_qwen_api(chunk, config, i, len(chunks), logger)
        all_items.extend(items)

    elapsed = time.time() - start_time

    if not all_items:
        logger.error("Không nhận được kết quả nào từ API!")
        return

    # --- Re-index IDs (đảm bảo liên tục từ 1) ---
    for idx, item in enumerate(all_items, 1):
        item["id"] = idx

    # --- Lưu script.json ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)

    # --- Thống kê ---
    narrator_count = sum(1 for it in all_items if it["role"] == "narrator")
    character_count = sum(1 for it in all_items if it["role"] == "character")
    pause_values = [it["pause_after_ms"] for it in all_items]
    total_pause_s = sum(pause_values) / 1000.0

    logger.info("")
    logger.info("=" * 60)
    logger.info("  KẾT QUẢ NLP GENERATOR")
    logger.info("=" * 60)
    logger.info(f"  Thời gian       : {elapsed:.1f}s")
    logger.info(f"  Tổng items      : {len(all_items)}")
    logger.info(f"    Narrator      : {narrator_count}")
    logger.info(f"    Character     : {character_count}")
    logger.info(f"  Khoảng lặng     :")
    logger.info(f"    Tổng          : {total_pause_s:.1f}s")
    logger.info(f"    Min           : {min(pause_values)} ms")
    logger.info(f"    Max           : {max(pause_values)} ms")
    logger.info(f"    Trung bình    : {sum(pause_values) / len(pause_values):.0f} ms")
    logger.info(f"  Output file     : {output_path}")

    # Mẫu kiểm tra
    logger.info("")
    logger.info("  Mẫu (5 items đầu):")
    for item in all_items[:5]:
        text_preview = item["text"][:50] + "..." if len(item["text"]) > 50 else item["text"]
        logger.info(
            f"    [{item['id']:3d}] [{item['role']:9s}] "
            f"{text_preview} | pause={item['pause_after_ms']}ms"
        )

    logger.info("")
    logger.info("=" * 60)
    logger.info("  PHASE 1 HOÀN TẤT!")
    logger.info("=" * 60)
    logger.info("  QUAN TRỌNG: Hãy TẮT vLLM server (Ctrl+C ở Terminal 1)")
    logger.info("  để giải phóng VRAM cho Phase 2 (TTS).")
    logger.info("")
    logger.info("  Bước tiếp theo:")
    logger.info("    python create_mean_style.py   (nếu chưa chạy)")
    logger.info("    python tts_generator.py        (Phase 2: TTS → audiobook)")
    logger.info("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NLP Generator — Phase 1: Qwen phân tích kịch bản truyện ma → script.json"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Đường dẫn file kịch bản truyện ma (.txt)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Đường dẫn file output (mặc định: script.json)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Đường dẫn config YAML",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Override URL vLLM API (mặc định: http://localhost:8000)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk size (ký tự)",
    )
    args = parser.parse_args()

    # Load .env
    env_candidates = [Path(".env"), Path("../.env")]
    for env_path in env_candidates:
        if env_path.exists():
            load_dotenv(str(env_path))
            break

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = NLPConfig.from_yaml(str(config_path))
    else:
        config = NLPConfig()

    # Override từ CLI
    if args.input:
        config.input_file = args.input
    if args.output:
        config.output_file = args.output
    if args.api_url:
        config.api_url = args.api_url
    if args.chunk_size:
        config.chunk_size = args.chunk_size

    # Validate
    if not config.input_file:
        print("[LỖI] Chưa chỉ định file input!")
        print("  python nlp_generator.py --input truyen_ma.txt")
        sys.exit(1)

    # Setup logging
    log_dir = Path(config.work_dir) / "logs"
    logger = setup_logging(log_dir)

    # Header
    logger.info("=" * 60)
    logger.info("  NLP GENERATOR — PHASE 1: ĐẠO DIỄN KỊCH BẢN")
    logger.info("=" * 60)
    logger.info(f"Config       : {config_path}")
    logger.info(f"Input file   : {config.input_file}")
    logger.info(f"Output file  : {config.output_file}")
    logger.info(f"API URL      : {config.api_url}")
    logger.info(f"Model        : {config.model_name}")
    logger.info(f"Chunk size   : {config.chunk_size} chars")
    logger.info(f"Temperature  : {config.temperature}")

    # Run
    try:
        generate_script(config, logger)
    except Exception as e:
        logger.exception(f"THẤT BẠI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()