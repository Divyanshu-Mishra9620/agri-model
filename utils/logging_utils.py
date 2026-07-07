from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter

def setup_logging(log_dir: str | Path, name: str = "krishinova_cv", level: int = logging.INFO) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()  

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return logging.getLogger(name)

class EpochLogger:

    def __init__(
        self,
        log_dir: str | Path,
        experiment_name: str,
        use_tensorboard: bool = True,
        use_csv: bool = True,
    ) -> None:
        self.run_dir = Path(log_dir) / experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._writer = (
            SummaryWriter(log_dir=str(self.run_dir / "tensorboard")) if use_tensorboard else None
        )

        self._csv_path = self.run_dir / "metrics.csv" if use_csv else None
        self._csv_header_written = bool(
            self._csv_path and self._csv_path.exists() and self._csv_path.stat().st_size > 0
        )

    def log_epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        if self._writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(key, value, epoch)
            self._writer.flush()

        if self._csv_path is not None:
            row = {"epoch": epoch, **metrics}
            with self._csv_path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                if not self._csv_header_written:
                    writer.writeheader()
                    self._csv_header_written = True
                writer.writerow(row)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
