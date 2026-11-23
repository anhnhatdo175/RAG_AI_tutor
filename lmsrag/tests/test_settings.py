import pytest

from lmsrag import settings


def test_load_data_config_resolves_paths_and_returns_seeds():
    config = settings.load("data")

    assert config["output_dir"].is_absolute()
    assert config["datasets"]["assist2009"]["path"].is_absolute()
    assert settings.seeds() == [0, 1, 2, 3, 4]
    assert settings.data_seed() == 2026


def test_load_rejects_missing_required_data_keys(tmp_path):
    config_path = tmp_path / "data.yaml"
    config_path.write_text("data_seed: 2026\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required data configuration keys"):
        settings.load(config_path)


def test_load_rejects_invalid_seed_values(tmp_path):
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        "data_seed: 2026\n"
        "seeds: [0, true]\n"
        "split: {history: 0.7, validation: 0.15, test: 0.15}\n"
        "datasets: {assist2009: {path: data.csv}}\n"
        "output_dir: artifacts\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="seeds must be a non-empty list of integers"):
        settings.load(config_path)


def test_load_missing_config_reports_absolute_path():
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        settings.load("does-not-exist")


def test_load_preserves_nested_config_path(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    nested_config = config_dir / "group" / "custom.yaml"
    nested_config.parent.mkdir(parents=True)
    nested_config.write_text("option: value\n", encoding="utf-8")
    monkeypatch.setattr(settings, "REPO", tmp_path)
    monkeypatch.setattr(settings, "CONFIG_DIR", config_dir)

    assert settings.load("group/custom") == {"option": "value"}
