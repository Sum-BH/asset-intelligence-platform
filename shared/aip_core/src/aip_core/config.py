"""Central configuration for the Asset Intelligence Platform.

Single source of truth for paths, environment settings, and the dataset
registry. Every module imports from here instead of hard-coding paths.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upwards until we find the repo root (marked by pyproject.toml + modules/)."""
    p = (start or Path(__file__)).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "modules").is_dir():
            return candidate
    raise RuntimeError("Repository root not found. Run from inside the repo.")


REPO_ROOT = find_repo_root()


class Paths(BaseModel):
    root: Path = REPO_ROOT
    data: Path = REPO_ROOT / "data"
    raw: Path = REPO_ROOT / "data" / "raw"
    interim: Path = REPO_ROOT / "data" / "interim"
    processed: Path = REPO_ROOT / "data" / "processed"
    synthetic: Path = REPO_ROOT / "data" / "synthetic"
    configs: Path = REPO_ROOT / "configs"
    docs: Path = REPO_ROOT / "docs"

    def ensure(self) -> None:
        for p in (self.raw, self.interim, self.processed, self.synthetic):
            p.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Runtime settings. Override via environment variables prefixed AIP_,
    e.g. AIP_MLFLOW_TRACKING_URI, or via a .env file at repo root."""

    model_config = SettingsConfigDict(
        env_prefix="AIP_", env_file=str(REPO_ROOT / ".env"), extra="ignore"
    )

    environment: str = "local"  # local | laptop | databricks
    mlflow_tracking_uri: str = "http://localhost:5000"
    duckdb_path: str = str(REPO_ROOT / "data" / "processed" / "aip.duckdb")
    ollama_base_url: str = "http://localhost:11434"
    log_level: str = "INFO"


# --- Dataset registry (Phase 0/1 scope; extended in later phases) -----------


class DatasetSpec(BaseModel):
    key: str
    name: str
    url: str
    archive_name: str
    phase: int
    notes: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "cmapss": DatasetSpec(
        key="cmapss",
        name="NASA C-MAPSS Turbofan Degradation",
        url="https://data.nasa.gov/download/nk8v-ckry/application%2Fzip",
        archive_name="CMAPSSData.zip",
        phase=1,
        notes="Primary Phase 1 dataset: RUL forecasting. If the NASA URL moves, "
        "search 'C-MAPSS turbofan degradation dataset' - it is mirrored on "
        "Kaggle and the NASA Prognostics Data Repository.",
    ),
    "ims_bearings": DatasetSpec(
        key="ims_bearings",
        name="NASA IMS Bearing Run-to-Failure",
        url="https://data.nasa.gov/download/brfb-gzcv/application%2Fzip",
        archive_name="IMS.zip",
        phase=1,
        notes="Vibration run-to-failure; used for envelope/FFT feature demos.",
    ),
    "ai4i2020": DatasetSpec(
        key="ai4i2020",
        name="AI4I 2020 Predictive Maintenance",
        url="https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
        archive_name="ai4i2020.zip",
        phase=1,
        notes="Small tabular PdM set; good for KPI dashboard seeding.",
    ),
}


paths = Paths()
settings = Settings()
