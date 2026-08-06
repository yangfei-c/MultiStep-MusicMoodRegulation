"""中英文文本 VA 训练入口。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments import run_text_va
from src.training.engine import RunOptions


RUN = RunOptions(evaluate_test=False)


def main() -> None:
    run_text_va(RUN)


if __name__ == "__main__":
    main()
