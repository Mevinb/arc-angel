"""Config loading: defaults, YAML merge, env overrides."""

from __future__ import annotations

from pathlib import Path

from jarvis.config import load_config


def test_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("JARVIS_DATA_DIR", raising=False)
    config = load_config(project_root=tmp_path)
    assert config.llm.base_url == "http://localhost:20128/v1"
    assert config.llm.model("fast") == "auto/fast"
    assert config.llm.model("reasoning") == "auto"
    assert config.safety_mode == "interactive"
    assert config.internships["sources"] == ["remoteok", "arbeitnow", "hackernews"]
    assert config.db_path == config.data_dir / "jarvis.db"
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
    monkeypatch.setenv("JARVIS_LLM_BASE_URL", "http://from-env/v1")
    monkeypatch.setenv("JARVIS_MODEL_FAST", "gpt-fast")
    config = load_config(project_root=tmp_path)
    assert config.llm.base_url == "http://from-env/v1"
    assert config.llm.model("fast") == "gpt-fast"


def test_env_file_loaded(tmp_path, monkeypatch):
    monkeypatch.delenv("JARVIS_LLM_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "# comment\nJARVIS_LLM_API_KEY=secret123\nJARVIS_LLM_BASE_URL=http://dotenv/v1\n")
    config = load_config(project_root=tmp_path)
    assert config.llm.api_key == "secret123"
    assert config.llm.base_url == "http://dotenv/v1"
