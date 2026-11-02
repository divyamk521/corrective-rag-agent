
from crag.config import ROOT, Settings


def test_defaults_are_sane():
    s = Settings(_env_file=None)
    assert s.llm_model == "gpt-4o-mini"
    assert 0.0 < s.ce_low < s.ce_high < 1.0
    assert s.child_chunk_chars < s.parent_max_chars
    assert s.max_retries >= 1


def test_paths_are_under_data_dir():
    s = Settings(_env_file=None)
    assert s.raw_dir == s.data_dir / "raw"
    assert s.index_dir == s.data_dir / "index"
    assert s.figures_dir == s.data_dir / "processed" / "figures"
    assert s.data_dir == ROOT / "data"


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "my-model")
    monkeypatch.setenv("ENABLE_WEB", "false")
    s = Settings(_env_file=None)
    assert s.llm_model == "my-model"
    assert s.enable_web is False
