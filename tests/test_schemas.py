
from crag.schemas import Answer, Chunk


def test_index_text_includes_section():
    c = Chunk(id="a", doc_id="d", page=1, section="Methods", text="We train a model.")
    assert c.index_text() == "Methods\nWe train a model."


def test_index_text_without_section():
    c = Chunk(id="a", doc_id="d", page=1, text="Plain text.")
    assert c.index_text() == "Plain text."


def test_defaults():
    c = Chunk(id="a", doc_id="d", page=2, text="x")
    assert c.kind == "text" and c.source_type == "internal" and c.rerank_score is None
    a = Answer(text="hi")
    assert a.citations == [] and a.refused is False and a.citation_coverage == 1.0
