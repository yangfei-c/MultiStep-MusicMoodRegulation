"""MTG-Jamendo 183 标签训练入口；运行设置直接修改下方变量。"""

import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import MTGDataset, build_dataloader
from src.training.common import get_device, load_yaml, logged_output, set_seed
from src.training.engine import RunOptions, prepare_run, run_experiment
from src.training.factory import build_model, build_optimizer, build_tag_loss
from src.training.tasks import MTG_TASK


RUN = RunOptions(smoke_test=False, resume_from=None, init_weights=None, evaluate_test=False)
DATASET_CONFIG = ROOT / "configs/dataset.yaml"
TRAIN_CONFIG = ROOT / "configs/train.yaml"


def main() -> None:
    dataset_config, train_config = load_yaml(DATASET_CONFIG), load_yaml(TRAIN_CONFIG)
    seed = int(train_config.get("seed", 42))
    set_seed(seed)
    device = get_device(str(train_config.get("device", "cpu")))
    data = train_config["data"]
    datasets = {split: MTGDataset(dataset_config["mtg"], split) for split in ("train", "validation", "test")}
    loaders = {
        split: build_dataloader(
            dataset,
            batch_size=int(data["batch_size"]),
            shuffle=split == "train",
            num_workers=int(data["num_workers"]),
            pin_memory=bool(data["pin_memory"]),
        )
        for split, dataset in datasets.items()
    }
    model = build_model(train_config, device)
    criterion = build_tag_loss(train_config)
    optimizer = build_optimizer(model, train_config)
    output_dir, resume_path, checkpoint = prepare_run(ROOT / "outputs/tag/mtg", RUN, seed)
    with logged_output(output_dir / "train.log"):
        print(time.strftime("[RUN] %Y-%m-%d %H:%M:%S"))
        if resume_path:
            print(f"[RUN] resume={resume_path}")
        run_experiment(
            task=MTG_TASK,
            options=RUN,
            output_dir=output_dir,
            resume_checkpoint=checkpoint,
            dataset_config=dataset_config,
            train_config=train_config,
            run_config={"dataset": dataset_config, "train": train_config},
            datasets=datasets,
            loaders=loaders,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            setup_lines=("split=official split-0 | labels=87+40+56=183",),
        )


if __name__ == "__main__":
    main()
