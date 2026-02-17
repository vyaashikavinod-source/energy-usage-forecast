from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]
    dataset_dir: Path = root / "dataset"
    artifacts_dir: Path = root / "artifacts"
    reports_dir: Path = root / "reports"
    images_dir: Path = root / "images"

PATHS = Paths()
