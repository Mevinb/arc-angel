"""Config loading: defaults, YAML merge, env overrides."""

from __future__ import annotations

from pathlib import Path

from arc.config import load_config


def test_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_DATA_DIR", raising=False)
    config = load_config(project_root=tmp_path)
    assert config.llm.base_url == "http://localhost:20128/v1"
    assert config.llm.model("fast") == "auto/fast"
    assert config.llm.model("reasoning") == "auto"
    assert config.safety_mode == "interactive"
    assert config.internships["sources"] == ["remoteok", "arbeitnow", "hackernews"]
    assert config.db_path == config.data_dir / "arc.db"
    assert config.data_dir.is_dir()  # created eagerly


def test_model_chain_deduplicates(tmp_path):
    config = load_config(project_root=tmp_path)
    chain = config.llm.chain("fast")  # auto/fast + fallback [auto]
    assert chain == ["auto/fast", "auto"]
    assert config.llm.chain("vision") == ["auto"]  # primary == fallback → dedup


def test_yaml_override(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "llm:\n  base_url: http://my-gateway:9000/v1\nsafety:\n  mode: yolo\n")
    config = load_config(project_root=tmp_path)
    assert config.llm.base_url == "http://my-gateway:9000/v1"
    assert config.safety_mode == "yolo"
    # untouched defaults survive the merge
    assert config.llm.model("fast") == "auto/fast"


def test_env_overrides_beat_yaml(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "llm:\n  base_url: http://from-yaml/v1\n")
    monkeypatch.setenv("ARC_LLM_BASE_URL", "http://from-env/v1")
    monkeypatch.setenv("ARC_MODEL_FAST", "gpt-fast")
    config = load_config(project_root=tmp_path)
    assert config.llm.base_url == "http://from-env/v1"
    assert config.llm.model("fast") == "gpt-fast"


def test_env_file_loaded(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_LLM_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "# comment\nARC_LLM_API_KEY=secret123\nARC_LLM_BASE_URL=http://dotenv/v1\n")
    config = load_config(project_root=tmp_path)
    assert config.llm.api_key == "secret123"
    assert config.llm.base_url == "http://dotenv/v1"


def test_legacy_env_vars_still_work(tmp_path, monkeypatch):
    """Pre-rename JARVIS_* vars are honoured as fallbacks."""
    monkeypatch.delenv("ARC_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("ARC_MODEL_FAST", raising=False)
    monkeypatch.setenv("JARVIS_LLM_BASE_URL", "http://legacy/v1")
    monkeypatch.setenv("JARVIS_MODEL_FAST", "legacy-fast")
    config = load_config(project_root=tmp_path)
    assert config.llm.base_url == "http://legacy/v1"
    assert config.llm.model("fast") == "legacy-fast"


def test_arc_env_beats_jarvis_env(tmp_path, monkeypatch):
    """ARC_* always wins when both the new and legacy names are set."""
    monkeypatch.setenv("ARC_LLM_BASE_URL", "http://arc/v1")
    monkeypatch.setenv("JARVIS_LLM_BASE_URL", "http://jarvis/v1")
    monkeypatch.setenv("ARC_MODEL_FAST", "arc-fast")
    monkeypatch.setenv("JARVIS_MODEL_FAST", "jarvis-fast")
    config = load_config(project_root=tmp_path)
    assert config.llm.base_url == "http://arc/v1"
    assert config.llm.model("fast") == "arc-fast"


def test_legacy_env_file_loaded(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_LLM_API_KEY", raising=False)
    monkeypatch.delenv("JARVIS_LLM_API_KEY", raising=False)
    (tmp_path / ".env").write_text("JARVIS_LLM_API_KEY=legacy-secret\n")
    config = load_config(project_root=tmp_path)
    assert config.llm.api_key == "legacy-secret"


def test_legacy_data_dir_env(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_DATA_DIR", raising=False)
    custom = tmp_path / "legacy-data"
    monkeypatch.setenv("JARVIS_DATA_DIR", str(custom))
    config = load_config(project_root=tmp_path)
    assert config.data_dir == custom
    assert custom.is_dir()


def test_db_migration_copies_legacy_file(tmp_path):
    """data/jarvis.db is copied to data/arc.db on first access."""
    from arc.config import ArcConfig, LLMConfig

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    legacy = data_dir / "jarvis.db"
    legacy.write_bytes(b"legacy-db-content")
    (data_dir / "jarvis.db-wal").write_bytes(b"wal-content")

    cfg = ArcConfig(
        llm=LLMConfig(
            base_url="http://x/v1", api_key="",
            models={"fast": "a", "reasoning": "b", "vision": "c"},
            fallbacks={},
        ),
        safety_mode="interactive",
        internships={}, automation={}, gmail={},
        data_dir=data_dir,
        config_path=tmp_path / "config.yaml",
    )
    assert cfg.db_path == data_dir / "arc.db"
    assert (data_dir / "arc.db").read_bytes() == b"legacy-db-content"
    assert (data_dir / "arc.db-wal").read_bytes() == b"wal-content"
    # legacy file stays as backup
    assert legacy.is_file()
    # second access does not overwrite an existing arc.db
    (data_dir / "arc.db").write_bytes(b"new-content")
    assert cfg.db_path.read_bytes() == b"new-content"
