
"""Day 1 sanity check: prints the active configuration and which libraries import correctly."""

import importlib
import platform
import sys

from crag.config import get_settings

MODULES = [
    "pydantic",
    "pydantic_settings",
    "openai",
    "sentence_transformers",
    "chromadb",
    "rank_bm25",
    "pymupdf",
    "langgraph",
    "tavily",
    "fastapi",
    "uvicorn",
    "numpy",
    "httpx",
]


def main() -> int:
    s = get_settings()
    print(f"python            : {platform.python_version()} ({sys.executable})")
    print(f"llm_model         : {s.llm_model}")
    print(f"llm_base_url      : {s.llm_base_url or '(OpenAI default)'}")
    print(f"llm_api_key set   : {bool(s.llm_api_key)}")
    print(f"tavily key set    : {bool(s.tavily_api_key)}")
    print(f"embedding_model   : {s.embedding_model}")
    print(f"data_dir          : {s.data_dir}")

    missing = []

    for name in MODULES:
        try:
            importlib.import_module(name)
            print(f"  import {name:<22} OK")
        except Exception as exc:
            missing.append(name)
            print(f"  import {name:<22} FAILED ({exc.__class__.__name__})")

    if missing:
        print(f"\nMissing: {', '.join(missing)}. Re-run: pip install -r requirements.txt")
        return 1

    print("\nAll imports OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
