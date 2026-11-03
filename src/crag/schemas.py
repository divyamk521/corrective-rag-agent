
"""Shared data models. Every module imports its types from here."""

from typing import Literal

from pydantic import BaseModel, Field

BlockKind = Literal["text", "table", "figure"]
Grade = Literal["correct", "ambiguous", "incorrect"]
SourceType = Literal["internal", "web"]


class Block(BaseModel):
    """A raw piece of a parsed PDF page (before chunking)."""

    doc_id: str
    page: int
    kind: BlockKind
    text: str = ""
    section: str = ""
    is_heading: bool = False
    image_path: str | None = None


class Parent(BaseModel):
    """A larger section of text. Children are searched; the parent is given to the LLM."""

    id: str
    doc_id: str
    page: int
    section: str = ""
    text: str


class Chunk(BaseModel):
    """The unit that is indexed, retrieved, cited and verified."""

    id: str
    doc_id: str
    page: int
    section: str = ""
    kind: BlockKind = "text"
    text: str
    parent_id: str | None = None
    image_path: str | None = None
    source_type: SourceType = "internal"
    source_uri: str = ""
    score: float = 0.0
    rerank_score: float | None = None

    def index_text(self) -> str:
        """Text used for embedding and BM25: the chunk plus its section title as context."""
        return f"{self.section}\n{self.text}" if self.section else self.text


class Citation(BaseModel):
    label: str  # "C1"
    chunk_id: str
    doc_id: str
    page: int
    section: str = ""
    kind: BlockKind = "text"
    source_type: SourceType = "internal"
    source_uri: str = ""
    image_path: str | None = None
    snippet: str = ""


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    invalid_labels: list[str] = Field(default_factory=list)
    citation_coverage: float = 1.0


class PipelineResult(BaseModel):
    """What every pipeline (baseline or CRAG) returns, so evaluation can treat them alike."""

    question: str
    answer: Answer
    contexts: list[Chunk] = Field(default_factory=list)
    retrieved: list[Chunk] = Field(default_factory=list)
    latency_s: float = 0.0
    retries: int = 0
    used_web: bool = False
    evaluation: str | None = None
    guardrail_score: float | None = None
    guardrail_verdict: str | None = None
    trace: list[dict] = Field(default_factory=list)
