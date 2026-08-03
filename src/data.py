import csv
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


mtg_files = {
    "train": "autotagging-train.tsv",
    "validation": "autotagging-validation.tsv",
    "test": "autotagging-test.tsv",
}


def segment_index(path: Path) -> int:
    """segment_10.npy -> 10。"""
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"分段文件名错误：{path}") from error


def load_song_features(feature_dir: Path) -> torch.Tensor:
    """读取一首音乐的全部 MERT 分段，返回 [S,12,768]。"""
    files = sorted(feature_dir.glob("segment_*.npy"), key=segment_index)
    if not files:
        raise FileNotFoundError(f"没有找到 MERT 特征：{feature_dir}")

    segments = []
    for path in files:
        feature = np.load(path, allow_pickle=False)
        if feature.shape == (1, 12, 768):
            feature = feature[0]
        if feature.shape != (12, 768):
            raise ValueError(f"MERT 形状错误：{path}，实际为 {feature.shape}")
        segments.append(torch.from_numpy(feature.astype(np.float32, copy=False)))

    return torch.stack(segments)


class MTGDataset(Dataset):
    """MTG-Jamendo 完整 183 标签数据集。"""

    def __init__(self, config: dict, split: str) -> None:
        feature_root = Path(config["feature_dir"])
        split_file = Path(config["split_dir"]) / mtg_files[split]
        tags = np.load(config["tag_vocabulary_file"], allow_pickle=False).reshape(-1).astype(str)
        tag_to_index = {tag: index for index, tag in enumerate(tags)}
        self.samples = []

        with split_file.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file, delimiter="\t")
            next(reader)
            for row in reader:
                feature_dir = feature_root / Path(row[3].replace("\\", "/")).with_suffix("")
                tag_indices = [tag_to_index[tag] for tag in row[5:]]
                self.samples.append((row[0], feature_dir, tag_indices))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        song_id, feature_dir, tag_indices = self.samples[index]
        target = torch.zeros(183, dtype=torch.float32)
        target[tag_indices] = 1.0
        return {"id": song_id, "features": load_song_features(feature_dir), "target": target, "dataset": "mtg"}


class VADataset(Dataset):
    """DEAM 或 PMEmo 静态 VA 数据集。"""

    def __init__(self, config: dict, dataset_name: str, split: str) -> None:
        feature_root = Path(config["feature_dir"])
        split_file = Path(config["split_dir"]) / f"{split}.txt"
        annotation_file = Path(config["static_annotation_file"])
        annotations = {}

        with annotation_file.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                song_id = str(int(float(row["song_id"])))
                annotations[song_id] = (float(row["valence_mean"]), float(row["arousal_mean"]))

        song_ids = [str(int(float(line))) for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.dataset_name = dataset_name
        self.samples = []

        for song_id in song_ids:
            valence, arousal = annotations[song_id]
            target = ((valence - 5.0) / 4.0, (arousal - 5.0) / 4.0)
            self.samples.append((song_id, feature_root / song_id, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        song_id, feature_dir, target = self.samples[index]
        return {
            "id": song_id,
            "features": load_song_features(feature_dir),
            "target": torch.tensor(target, dtype=torch.float32),
            "dataset": self.dataset_name,
        }


def collate_music_batch(samples: list[dict]) -> dict:
    """把变长歌曲填充为 [B,S_max,12,768]。"""
    feature_list = [sample["features"] for sample in samples]
    lengths = torch.tensor([feature.shape[0] for feature in feature_list], dtype=torch.long)
    features = pad_sequence(feature_list, batch_first=True, padding_value=0.0)
    segment_mask = torch.arange(features.shape[1])[None, :] < lengths[:, None]

    return {
        "ids": [sample["id"] for sample in samples],
        "features": features,
        "segment_mask": segment_mask,
        "lengths": lengths,
        "targets": torch.stack([sample["target"] for sample in samples]),
        "dataset": samples[0]["dataset"],
    }


def build_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = False, num_workers: int = 0, pin_memory: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_music_batch)