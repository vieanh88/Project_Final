"""
=============================================================
  D2: NLP GENERATOR — Text truyện ma → script.json
=============================================================
Mục đích:
  Phase 1 của inference pipeline. Đọc file .txt truyện ma tiếng Việt,
  gọi Gemini 2.5 Flash API để:
    1. Cắt thành các câu ngắn (10-25 từ)
    2. Gán role: narrator / character_male / character_female
    3. Tính pause_after_ms theo nhịp điệu horror (200-3000ms)
  Output: script.json (input cho D3 tts_generator.py)

Quan trọng:
  - Dùng SDK MỚI: google-genai (KHÔNG phải google-generativeai cũ)
  - Model: gemini-2.5-flash (Gemini 2.0 Flash đã shutdown 1/6/2026)
  - Structured output qua response_schema -> Pydantic model
    -> Gemini ĐẢM BẢO output đúng JSON schema, không cần parse markdown

Setup:
  1. Tạo Gemini API key: https://aistudio.google.com/apikey (free)
  2. Lưu vào .env (cùng folder TTS_StyleTTS2-lite-vi/):
        GEMINI_API_KEY=AIzaSy...
  3. Cài deps:
        pip install google-genai python-dotenv pydantic

Cách dùng (mọi đường dẫn & tham số đọc từ inference_config.yaml, section `nlp:`):
    python inference/nlp_generator.py --config inference/inference_config.yaml

Override nhanh qua CLI (ưu tiên CLI > config > default):
    python inference/nlp_generator.py --config inference/inference_config.yaml \\
        --input data/raw_stories/story2.txt --no-thinking --dry-run

Các tham số chỉnh trong section `nlp:` của config:
    input, output, model, env, chunk_size, max_retries, thinking, dry_run
=============================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

# Tham số & đường dẫn tập trung trong inference_config.yaml (xem config_loader.py)
from config_loader import (
    PROJECT_ROOT,
    DEFAULT_CONFIG_PATH,
    load_config,
    cfg_value,
    resolve_path,
)

# Fallback .env mặc định (chỉ dùng nếu config + CLI đều không set)
DEFAULT_ENV = PROJECT_ROOT.parent / ".env"


# ============================================================
# 1. PYDANTIC SCHEMA — Gemini sẽ enforce
# ============================================================
class ScriptLine(BaseModel):
    """Một dòng trong script. Gemini guaranteed trả về đúng format này."""
    id: int = Field(
        description="Số thứ tự câu, bắt đầu từ 1, tăng liên tục.",
        ge=1,
    )
    role: Literal["narrator", "character_male", "character_female"] = Field(
        description=(
            "Vai đọc câu này. "
            "narrator = lời dẫn truyện (không phải lời thoại của nhân vật). "
            "character_male = lời thoại của nhân vật nam (hoặc giới tính không rõ). "
            "character_female = lời thoại CỦA NHÂN VẬT NỮ (phải rõ ràng là nữ)."
        ),
    )
    text: str = Field(
        description=(
            "Câu tiếng Việt, từ 5-30 từ. Phải có dấu câu cuối (. ! ?). "
            "KHÔNG được chứa các ký tự đặc biệt như #, *, [, ]. "
            "KHÔNG được chứa số (vd: '1990' phải viết 'một nghìn chín trăm chín mươi')."
        ),
        min_length=3,
    )
    pause_after_ms: int = Field(
        description=(
            "Khoảng lặng sau câu, theo ms. "
            "Sau dấu phẩy giữa câu: 200-300. "
            "Sau dấu . ? ! kết câu bình thường: 500-800. "
            "Sau câu chuyển cảnh / cao trào / rùng rợn: 1500-2500."
        ),
        ge=100,
        le=3000,
    )


# ============================================================
# 2. MASTER PROMPT
# ============================================================
MASTER_PROMPT = """Bạn là một biên kịch audiobook chuyên về thể loại TRUYỆN MA / HORROR tiếng Việt.

Nhiệm vụ: phân tích văn bản truyện ma được cung cấp và chuyển thành kịch bản đọc thoại có nhịp điệu kinh dị.

## QUY TẮC PHÂN CÂU
1. Mỗi câu trong kịch bản dài 5-25 từ (đủ ngắn để 1 hơi thở).
2. Câu gốc dài > 25 từ -> CẮT thành 2-3 câu, ngắt ở dấu phẩy hoặc liên từ.
3. Câu gốc quá ngắn (< 5 từ) -> GIỮ NGUYÊN nếu nó là câu cảm thán/hỏi gọn. Không gộp.
4. GIỮ TOÀN BỘ NỘI DUNG văn bản gốc, không tự thêm/bớt/diễn giải.
5. Loại bỏ chữ số (vd "1990" -> "một nghìn chín trăm chín mươi" hoặc bỏ đi nếu không quan trọng).

## QUY TẮC GÁN ROLE
- "narrator": lời dẫn chuyện (mô tả cảnh, hành động, suy nghĩ ngầm), KHÔNG phải lời thoại.
- "character_male": lời thoại trong dấu ngoặc kép HOẶC có dấu "-" đầu câu, mà NGƯỜI NÓI LÀ NAM hoặc không rõ giới tính.
- "character_female": lời thoại mà NGƯỜI NÓI rõ ràng là NỮ (qua context: tên nhân vật, đại từ "cô", "bà", "chị", "em gái", ...).
- Nếu không chắc giới tính -> mặc định "character_male" (vì giọng chính của hệ thống là giọng nam Ngạn).

## QUY TẮC PAUSE_AFTER_MS (kiến tạo không khí kinh dị)
- Câu thường (kết thúc bằng . , ? ! ở GIỮA đoạn): 500-800 ms.
- Câu nối tiếp ý (kết thúc bằng dấu phẩy nội bộ nếu được giữ): 200-300 ms.
- Câu CHUYỂN CẢNH (kết thúc 1 phân đoạn, mở cảnh mới): 1500-2000 ms.
- Câu CAO TRÀO / RÙNG RỢN / hé lộ sự kiện kinh dị: 2000-2500 ms.
- Câu CUỐI ĐOẠN văn bản: 2500-3000 ms.

## EXAMPLE
Input:
"Tôi mở cửa, một cơn gió lạnh ùa vào. Bỗng nhiên, từ trong góc tối, có giọng nói thì thầm: \"Đừng quay lại...\". Tôi sợ hãi, cô gái bên cạnh thì thào: \"Anh ơi, có ai đó trong phòng.\""

Output JSON:
[
  {"id": 1, "role": "narrator", "text": "Tôi mở cửa, một cơn gió lạnh ùa vào.", "pause_after_ms": 1500},
  {"id": 2, "role": "narrator", "text": "Bỗng nhiên, từ trong góc tối, có giọng nói thì thầm:", "pause_after_ms": 800},
  {"id": 3, "role": "character_male", "text": "Đừng quay lại...", "pause_after_ms": 2200},
  {"id": 4, "role": "narrator", "text": "Tôi sợ hãi, cô gái bên cạnh thì thào:", "pause_after_ms": 700},
  {"id": 5, "role": "character_female", "text": "Anh ơi, có ai đó trong phòng.", "pause_after_ms": 2500}
]

## VĂN BẢN TRUYỆN MA CẦN XỬ LÝ

{INPUT_TEXT}

## YÊU CẦU CUỐI
Trả về MẢNG JSON đúng schema. ID phải bắt đầu từ {START_ID} và tăng liên tục.
KHÔNG kèm theo bất kỳ markdown, giải thích, hay text nào khác ngoài JSON array.
"""


# ============================================================
# 3. ENV + CLIENT SETUP
# ============================================================
def load_api_key(env_path: Optional[Path] = None) -> str:
    """Load GEMINI_API_KEY từ .env hoặc environment."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        raise ImportError(
            "Thiếu 'python-dotenv'. Cài: pip install python-dotenv"
        )

    if env_path is None:
        env_path = DEFAULT_ENV

    if env_path.exists():
        load_dotenv(env_path)
        print(f"  Loaded .env từ: {env_path}")
    else:
        print(f"  ⚠️  .env không tìm thấy ở {env_path} — thử dùng env variable trực tiếp")

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY không có. Hãy:\n"
            f"  1. Tạo key: https://aistudio.google.com/apikey\n"
            f"  2. Lưu vào {env_path}:\n"
            f"     GEMINI_API_KEY=AIzaSy...\n"
        )
    return api_key


def build_client(api_key: str):
    """Build google-genai client."""
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "Thiếu 'google-genai'. Cài: pip install google-genai\n"
            "Note: KHÔNG phải 'google-generativeai' (deprecated)."
        )
    return genai.Client(api_key=api_key)


# ============================================================
# 4. TEXT PREPROCESSING
# ============================================================
def read_input_text(path: Path) -> str:
    """Read text file với UTF-8 auto-fallback."""
    if not path.exists():
        raise FileNotFoundError(f"Input file không tồn tại: {path}")

    for encoding in ("utf-8", "utf-8-sig", "cp1258", "latin-1"):
        try:
            text = path.read_text(encoding=encoding)
            if encoding != "utf-8":
                print(f"  ⚠️  File đọc bằng {encoding} (không phải utf-8)")
            return text
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Không đọc được {path} với bất kỳ encoding nào")


def chunk_text_by_paragraph(text: str, max_chars: int = 8000) -> List[str]:
    """
    Chia text dài thành các chunk ≤ max_chars, cắt theo đoạn.
    Tránh cắt giữa câu.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    # Split theo đoạn (double newline)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        # Fallback: split theo single newline
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= max_chars:
            current = current + "\n\n" + p if current else p
        else:
            if current:
                chunks.append(current)
            # Nếu 1 đoạn riêng vẫn > max_chars, force split
            if len(p) > max_chars:
                while len(p) > max_chars:
                    # Split tại dấu chấm gần nhất trong [max_chars*0.8, max_chars]
                    cut = p.rfind(".", int(max_chars * 0.8), max_chars)
                    if cut == -1:
                        cut = max_chars
                    chunks.append(p[:cut + 1].strip())
                    p = p[cut + 1:].strip()
                current = p
            else:
                current = p
    if current:
        chunks.append(current)
    return chunks


# ============================================================
# 5. GEMINI API CALL
# ============================================================
def call_gemini_chunk(
    client,
    model: str,
    chunk_text: str,
    start_id: int,
    use_thinking: bool,
    max_retries: int,
) -> List[dict]:
    """
    Gọi Gemini cho 1 chunk text. Returns list of dict (validated).
    Retry với exponential backoff nếu fail.
    """
    from google.genai import types

    prompt = (
        MASTER_PROMPT
        .replace("{INPUT_TEXT}", chunk_text)
        .replace("{START_ID}", str(start_id))
    )

    # Config với structured output
    config_kwargs = {
        "response_mime_type": "application/json",
        "response_schema": list[ScriptLine],
        "temperature": 0.3,    # thấp để output deterministic
        "max_output_tokens": 8192,
    }

    # Tắt thinking cho gemini-2.5-flash (mặc định bật)
    # -> nhanh hơn ~2x cho task không cần reasoning sâu
    if not use_thinking:
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except (AttributeError, TypeError):
            # Model cũ không support thinking_config -> skip silently
            pass

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"    [Attempt {attempt}/{max_retries}] Calling {model}...")
            t0 = time.time()
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            elapsed = time.time() - t0
            print(f"    ✅ Response trong {elapsed:.1f}s")

            # response.text là JSON string đảm bảo đúng schema
            if not response.text:
                raise ValueError("Response empty")

            data = json.loads(response.text)
            if not isinstance(data, list):
                raise ValueError(f"Expected list, got {type(data).__name__}")

            # Validate từng item qua Pydantic
            validated = []
            for item in data:
                try:
                    line = ScriptLine(**item)
                    validated.append(line.model_dump())
                except ValidationError as ve:
                    print(f"    ⚠️  Skip line không hợp lệ: {item} ({ve})")

            return validated

        except Exception as e:
            last_error = e
            wait = 2 ** attempt
            print(f"    ❌ Fail attempt {attempt}: {type(e).__name__}: {e}")
            if attempt < max_retries:
                print(f"    ⏳ Wait {wait}s rồi retry...")
                time.sleep(wait)

    raise RuntimeError(
        f"Gemini API fail sau {max_retries} attempts. Last error: {last_error}"
    )


# ============================================================
# 6. POST-PROCESSING
# ============================================================
def post_process_script(lines: List[dict]) -> List[dict]:
    """
    Sau khi merge từ nhiều chunk:
      1. Re-assign ID liên tục từ 1
      2. Verify không có gap/duplicate
      3. Strip whitespace
    """
    cleaned = []
    for idx, line in enumerate(lines, start=1):
        line["id"] = idx
        line["text"] = line["text"].strip()
        cleaned.append(line)
    return cleaned


def estimate_audio_duration_sec(lines: List[dict]) -> float:
    """
    Ước tính tổng duration audio cuối:
      - Mỗi câu: ~3-5 giây đọc + pause sau
      - Trung bình 4 giây / câu + pause_after_ms / 1000
    """
    total = 0.0
    for line in lines:
        # Đọc: ~0.3 giây / từ (giọng kể chậm rãi)
        n_words = len(line["text"].split())
        speech_sec = n_words * 0.3
        pause_sec = line["pause_after_ms"] / 1000.0
        total += speech_sec + pause_sec
    return total


# ============================================================
# 7. MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # File config YAML — nguồn chính của mọi đường dẫn & tham số
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH),
        help=f"Path tới inference config YAML (default: {DEFAULT_CONFIG_PATH.name})",
    )
    # ---- Override CLI (None = không truyền -> dùng config -> default) ----
    parser.add_argument("--input", type=str, default=None,
                        help="[nlp.input] file .txt truyện ma")
    parser.add_argument("--output", type=str, default=None,
                        help="[nlp.output] file/folder output. Bỏ trống -> output/nlp/<tên-input>.json")
    parser.add_argument("--model", type=str, default=None,
                        help="[nlp.model] tên model Gemini")
    parser.add_argument("--env", type=str, default=None,
                        help="[nlp.env] path tới .env chứa GEMINI_API_KEY")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="[nlp.chunk_size] số ký tự tối đa mỗi chunk")
    parser.add_argument("--max-retries", type=int, default=None,
                        help="[nlp.max_retries] số lần retry khi API fail")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None,
                        help="[nlp.thinking] --thinking / --no-thinking (tắt nhanh hơn ~2x)")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None,
                        help="[nlp.dry_run] --dry-run = in prompt, KHÔNG gọi API")
    args = parser.parse_args()

    print("=" * 60)
    print("D2 — NLP GENERATOR (Gemini -> script.json)")
    print("=" * 60)

    # ===== Resolve config (CLI > YAML > default) =====
    cfg = load_config(args.config)

    input_path = resolve_path(cfg_value(cfg, "nlp", "input", args.input))
    model = cfg_value(cfg, "nlp", "model", args.model)
    chunk_size = cfg_value(cfg, "nlp", "chunk_size", args.chunk_size)
    max_retries = cfg_value(cfg, "nlp", "max_retries", args.max_retries)
    use_thinking = cfg_value(cfg, "nlp", "thinking", args.thinking)
    dry_run = cfg_value(cfg, "nlp", "dry_run", args.dry_run)

    # .env: CLI > config > DEFAULT_ENV
    env_path = resolve_path(cfg_value(cfg, "nlp", "env", args.env)) or DEFAULT_ENV.resolve()

    # Output: ưu tiên CLI > config; None -> tự sinh output/nlp/<tên-input>.json;
    #         nếu trỏ folder -> đặt file theo tên input trong folder đó.
    default_output_name = f"{input_path.stem}.json"
    output_arg = cfg_value(cfg, "nlp", "output", args.output)
    if output_arg is None:
        output_path = (PROJECT_ROOT / "output" / "nlp" / default_output_name).resolve()
    else:
        resolved = resolve_path(output_arg)
        if resolved.suffix.lower() == ".json":
            output_path = resolved
        else:
            output_path = (resolved / default_output_name).resolve()

    print(f"Input file  : {input_path}")
    print(f"Output file : {output_path}")
    print(f"Model       : {model}")
    print(f"Thinking    : {'ON' if use_thinking else 'OFF'}")
    print(f"Dry run     : {dry_run}")

    # 1. Read text
    print(f"\n[1/5] Reading input text...")
    text = read_input_text(input_path)
    n_chars = len(text)
    n_words = len(text.split())
    print(f"  Length: {n_chars:,} chars, {n_words:,} words")

    # 2. Chunk
    print(f"\n[2/5] Chunking (max {chunk_size} chars/chunk)...")
    chunks = chunk_text_by_paragraph(text, max_chars=chunk_size)
    print(f"  {len(chunks)} chunk(s)")

    # 3. Dry run?
    if dry_run:
        print(f"\n[DRY RUN] Sample prompt cho chunk 1:")
        sample_prompt = (
            MASTER_PROMPT
            .replace("{INPUT_TEXT}", chunks[0][:500] + ("..." if len(chunks[0]) > 500 else ""))
            .replace("{START_ID}", "1")
        )
        print("=" * 60)
        print(sample_prompt)
        print("=" * 60)
        print("\n(Dry-run: KHÔNG gọi API)")
        return

    # 4. Setup API
    print(f"\n[3/5] Setup Gemini client...")
    api_key = load_api_key(env_path)
    print(f"  API key: {api_key[:8]}...{api_key[-4:]} (length {len(api_key)})")
    client = build_client(api_key)

    # 5. Call API per chunk
    print(f"\n[4/5] Calling Gemini API ({len(chunks)} chunks)...")
    all_lines = []
    current_id = 1
    for idx, chunk in enumerate(chunks, start=1):
        print(f"\n  Chunk {idx}/{len(chunks)}  ({len(chunk):,} chars)")
        try:
            lines = call_gemini_chunk(
                client,
                model=model,
                chunk_text=chunk,
                start_id=current_id,
                use_thinking=use_thinking,
                max_retries=max_retries,
            )
        except Exception as e:
            print(f"  ❌ Chunk {idx} FAIL hoàn toàn: {e}")
            print(f"     Skip chunk này, tiếp tục với chunks còn lại.")
            continue

        print(f"    Sinh ra {len(lines)} script lines")
        all_lines.extend(lines)
        current_id = len(all_lines) + 1

    if not all_lines:
        print("\n❌ KHÔNG sinh được dòng nào. Check log + retry.")
        sys.exit(1)

    # 6. Post-process
    print(f"\n[5/5] Post-process + save...")
    all_lines = post_process_script(all_lines)

    # Stats
    role_counts = {}
    for line in all_lines:
        role_counts[line["role"]] = role_counts.get(line["role"], 0) + 1
    duration_est = estimate_audio_duration_sec(all_lines)

    # Save script.json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_lines, f, ensure_ascii=False, indent=2)

    # Save metadata.json
    metadata_path = output_path.parent / f"{input_path.name}.metadata.json"
    metadata = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "stats": {
            "input_chars": n_chars,
            "input_words": n_words,
            "n_chunks": len(chunks),
            "n_lines": len(all_lines),
            "role_distribution": role_counts,
            "estimated_audio_duration_sec": round(duration_est, 1),
            "estimated_audio_duration_min": round(duration_est / 60, 1),
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"✅ SUCCESS")
    print(f"{'=' * 60}")
    print(f"  Script  : {output_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"\n  Stats:")
    print(f"    Lines           : {len(all_lines)}")
    print(f"    Roles           : {role_counts}")
    print(f"    Est. audio dur. : {duration_est:.1f}s ({duration_est/60:.1f} min)")

    # Print preview 3 dòng đầu
    print(f"\n  Preview 3 dòng đầu:")
    for line in all_lines[:3]:
        print(f"    [{line['id']:3d}] [{line['role']:18s}] [{line['pause_after_ms']:4d}ms] {line['text']}")

    print(f"\n👉 Bước tiếp theo:")
    print(f"  - Mở {output_path} để review script (đặc biệt: role có đúng không?)")
    print(f"  - Nếu OK -> chạy D3 (tts_generator.py) để sinh audio")
    print(f"  - Nếu role sai nhiều -> sửa thủ công trong JSON hoặc retry với chunk nhỏ hơn")

if __name__ == "__main__":
    main()