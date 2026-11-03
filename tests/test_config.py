
from crag.config import ROOT, Settings


def test_defaults_are_sane():
    settings = Settings(_env_file=None)

    assert settings.llm_model == "llama-3.3-70b-versatile"
    assert settings.ce_high > settings.ce_low
    assert settings.child_chunk_chars < settings.parent_max_chars
    assert settings.max_retries >= 1


def test_paths_are_under_data_dir():
    settings = Settings(_env_file=None)

    assert settings.raw_dir == settings.data_dir / "raw"
    assert settings.index_dir == settings.data_dir / "index"
    assert settings.figures_dir == settings.data_dir / "processed" / "figures"
    assert settings.data_dir == ROOT / "data"


def test_env_override(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "my-model")
    monkeypatch.setenv("ENABLE_WEB", "false")

    settings = Settings(_env_file=None)

    assert settings.llm_model == "my-model"
    assert settings.enable_web is False
