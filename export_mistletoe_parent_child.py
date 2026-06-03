"""
将 Markdown(内含 HTML) 解析为父子块，并构建 FAISS 向量库。
CPU 环境使用分批编码，避免一次性占用过多内存。
"""
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

_CURSOR_DIR = Path(__file__).resolve().parent
if str(_CURSOR_DIR) not in sys.path:
    sys.path.insert(0, str(_CURSOR_DIR))

from html_parent_child_chunker import (
    OVERLAP_CHILD_COUNT,
    PARENT_MAX_CHARS,
    chunk_markdown_html,
    optional_build_faiss,
)


SOURCE = Path(r"D:\OllamaYXX\RAG_Project\doc_process\mistletoe解析结果_LLM优化后.md")
OUT_DIR = Path(r"D:\OllamaYXX\RAG_Project\segment_processing_500")
OUT_JSON = OUT_DIR / "mistletoe解析结果_LLM优化后_parent_child.json"
OUT_PICKLE = OUT_DIR / "mistletoe解析结果_LLM优化后_parent_child.pkl"

FAISS_INDEX_PATH = OUT_DIR / "faiss_parent_child_index.bin"
CHILD_DATA_PATH = OUT_DIR / "child_data.pkl"
PARENT_DATA_PATH = OUT_DIR / "parent_data.pkl"

EMBED_MODEL_PATH = r"D:\anaconda\conda_envs\qwen-agent\models\Qwen3-Embedding-0.6B"

# CPU 批处理大小：需要在内存和速度之间折中
BATCH_SIZE = 32


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    text = SOURCE.read_text(encoding="utf-8")
    result = chunk_markdown_html(text)

    kids = result["child_blocks"]
    export = {
        "source": str(SOURCE.resolve()),
        "parent_max_chars": PARENT_MAX_CHARS,
        "overlap_child_count": OVERLAP_CHILD_COUNT,
        "num_child_blocks": len(kids),
        "num_parents": len(result["parent_texts"]),
        "child_blocks": [asdict(c) for c in kids],
        "parent_texts": result["parent_texts"],
        "parent_child_indices": result["parent_child_indices"],
        "child_to_parent": result["child_to_parent"],
    }

    OUT_JSON.write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")
    with OUT_PICKLE.open("wb") as f:
        pickle.dump(result, f)

    optional_build_faiss(
        result=result,
        embed_model_path=EMBED_MODEL_PATH,
        faiss_index_path=str(FAISS_INDEX_PATH),
        child_data_path=str(CHILD_DATA_PATH),
        parent_data_path=str(PARENT_DATA_PATH),
        device="cpu",
        batch_size=BATCH_SIZE,
    )

    print(f"子块数: {len(kids)}")
    print(f"父块数: {len(result['parent_texts'])}")
    print(f"JSON: {OUT_JSON}")
    print(f"Pickle: {OUT_PICKLE}")
    print(f"FAISS索引: {FAISS_INDEX_PATH}")
    print(f"子块数据: {CHILD_DATA_PATH}")
    print(f"父块数据: {PARENT_DATA_PATH}")


if __name__ == "__main__":
    main()
