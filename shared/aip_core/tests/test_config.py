from pathlib import Path

from aip_core import DATASETS, paths, settings


def test_repo_root_detected():
    assert (paths.root / "pyproject.toml").exists()
    assert (paths.root / "modules").is_dir()


def test_paths_ensure(tmp_path: Path):
    paths.ensure()
    assert paths.raw.is_dir()


def test_dataset_registry_has_phase1_sets():
    assert {"cmapss", "ims_bearings", "ai4i2020"} <= set(DATASETS)
    for spec in DATASETS.values():
        assert spec.url.startswith("https://")


def test_settings_defaults():
    assert settings.environment in {"local", "laptop", "databricks"}
