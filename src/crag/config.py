
"""Central configuration. Every tunable value lives here; nothing is hard-coded elsewhere."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# src/crag/config.py -> parents[0]=crag, [1]=src, [2]=repository root
ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM (Groq) ----
    llm_api_key: str = Field(
        default="",
        validation_alias="GROQ_API_KEY",
    )
    llm_base_url: str | None = None

    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias="GROQ_MODEL",
    )

    vision_model: str = Field(
        default="qwen/qwen3.8-27b",
        validation_alias="GROQ_VISION_MODEL",
    )

    judge_model: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias="GROQ_JUDGE_MODEL",
    )

    # ---- Web fallback ----
    tavily_api_key: str = ""
    enable_web: bool = True
    web_max_results: int = 3
    web_chunk_chars: int = 1200

    # ---- Local models ----
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ---- Paths ----
    data_dir: Path = ROOT / "data"

    # ---- Ingestion ----
    caption_figures: bool = True
    min_figure_px: int = 150
    child_chunk_chars: int = 600
    chunk_overlap_chars: int = 100
    parent_max_chars: int = 2500
    max_table_chars: int = 3000

    # ---- Retrieval ----
    top_k_dense: int = 20
    top_k_bm25: int = 20
    rerank_candidates: int = 30
    final_k: int = 6

    # ---- Agent (CRAG) ----
    ce_high: float = 0.70
    ce_low: float = 0.15
    max_retries: int = 2
    guardrail_threshold: float = 0.80

    # ---- Generation ----
    max_images_in_prompt: int = 2

    # ---- Cache ----
    cache_threshold: float = 0.95

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def figures_dir(self) -> Path:
        return self.processed_dir / "figures"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


@lru_cache
def get_settings() -> Settings:
    return Settings()
