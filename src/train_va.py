"""VA 回归入口"""

import sys
import time
from pathlib import Path

from torch.utils.data import ConcatDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import VADataset, build_dataloader, domain_balanced_sampler
from src.training.common import get_device, load_yaml, logged_output, set_seed
from src.training.engine import RunOptions, prepare_run, run_experiment
from src.training.factory import build_model, build_optimizer, build_va_loss
from src.training.tasks import make_domain_balanced_va_metrics, make_va_task


EXPERIMENT = "deam"  # deam | pmemo | joint | deam_to_pmemo | pmemo_to_deam
RUN = RunOptions(smoke_test=False, resume_from=None, init_weights=None, evaluate_test=False)
DATASET_CONFIG = ROOT / "configs/dataset.yaml"
TRAIN_CONFIG = ROOT / "configs/train.yaml"
DOMAINS = ("deam", "pmemo")
SPLITS = ("train", "validation", "test")


def build_experiment(dataset_config: dict, experiment: str, seed: int):
    all_sets = {
        name: {split: VADataset(dataset_config[name], name, split) for split in SPLITS}
        for name in DOMAINS
    }
    if experiment in DOMAINS:
        return all_sets[experiment], None, make_va_task(experiment)
    if "_to_" in experiment:
        source, target = experiment.split("_to_", 1)
        if source not in DOMAINS or target not in DOMAINS or source == target:
            raise ValueError(f"未知跨域实验：{experiment}")
        datasets = {
            "train": all_sets[source]["train"],
            "validation": all_sets[target]["validation"],
            "test": all_sets[target]["test"],
        }
        return datasets, None, make_va_task(experiment)
    if experiment != "joint":
        raise ValueError(f"未知 VA 实验：{experiment}")

    datasets = {
        split: ConcatDataset([all_sets[name][split] for name in DOMAINS])
        for split in SPLITS
    }
    lengths_by_total = {
        sum(len(all_sets[name][split]) for name in DOMAINS): tuple(len(all_sets[name][split]) for name in DOMAINS)
        for split in ("validation", "test")
    }

    def joint_metrics(predictions, targets):
        lengths = lengths_by_total.get(len(predictions))
        if lengths is None:
            raise ValueError(f"无法识别 joint VA 评估样本数：{len(predictions)}")
        return make_domain_balanced_va_metrics(lengths, DOMAINS)(predictions, targets)

    train_lengths = [len(all_sets[name]["train"]) for name in DOMAINS]
    return datasets, domain_balanced_sampler(train_lengths, seed), make_va_task(experiment, joint_metrics)


def main() -> None:
    dataset_config, train_config = load_yaml(DATASET_CONFIG), load_yaml(TRAIN_CONFIG)
    seed = int(train_config.get("seed", 42))
    set_seed(seed)
    device = get_device(str(train_config.get("device", "cpu")))
    datasets, sampler, task = build_experiment(dataset_config, EXPERIMENT, seed)
    data = train_config["data"]
    loaders = {
        split: build_dataloader(
            dataset,
            batch_size=int(data["batch_size"]),
            shuffle=split == "train" and sampler is None,
            sampler=sampler if split == "train" else None,
            num_workers=int(data["num_workers"]),
            pin_memory=bool(data["pin_memory"]),
        )
        for split, dataset in datasets.items()
    }
    model = build_model(train_config, device)
    criterion = build_va_loss(train_config)
    optimizer = build_optimizer(model, train_config)
    output_dir, resume_path, checkpoint = prepare_run(ROOT / "outputs/va" / EXPERIMENT, RUN, seed)
    with logged_output(output_dir / "train.log"):
        print(time.strftime("[RUN] %Y-%m-%d %H:%M:%S"))
        if resume_path:
            print(f"[RUN] resume={resume_path}")
        run_experiment(
            task=task,
            options=RUN,
            output_dir=output_dir,
            resume_checkpoint=checkpoint,
            dataset_config=dataset_config,
            train_config=train_config,
            run_config={"experiment": EXPERIMENT, "dataset": dataset_config, "train": train_config},
            datasets=datasets,
            loaders=loaders,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            setup_lines=("VA: y_norm=(y-5)/4 | joint selection=mean domain CCC",),
        )


if __name__ == "__main__":
    main()
