"""DEAM、PMEmo 或联合音乐 VA 训练入口。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments import run_music_va
from src.training.engine import RunOptions


EXPERIMENT = "joint"  # deam | pmemo | joint
RUN_NAME: str | None = None
RUN = RunOptions(evaluate_test=False)


def main() -> None:
    run_music_va(EXPERIMENT, RUN, RUN_NAME)


if __name__ == "__main__":
    main()
