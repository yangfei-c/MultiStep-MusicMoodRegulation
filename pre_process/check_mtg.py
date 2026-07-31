import csv
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "configs" / "dataset.yaml"

SPLIT_FILES = {
    "train": "autotagging-train.tsv",
    "validation": "autotagging-validation.tsv",
    "test": "autotagging-test.tsv",
}

EXPECTED_TRACKS = {
    "train": 32859,
    "validation": 11101,
    "test": 11565,
}

EXPECTED_TAGS = {
    "genre": 87,
    "instrument": 40,
    "mood/theme": 56,
    "total": 183,
}

EXPECTED_UNION = 55525


def check(name: str, condition: bool, value=None) -> None:
    """检查条件；失败时立即停止。"""
    if not condition:
        raise RuntimeError(
            f"[FAIL] {name}" if value is None
            else f"[FAIL] {name}：{value}"
        )

    print(
        f"[PASS] {name}" if value is None
        else f"[PASS] {name}：{value}"
    )


def load_mtg_config() -> dict:
    """读取 dataset.yaml 中的 mtg 配置。"""
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or "mtg" not in config:
        raise ValueError("dataset.yaml 中缺少 mtg 配置。")

    return config["mtg"]


def read_tsv(path: Path) -> list[list[str]]:
    """读取 MTG 官方 TSV。"""
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader, None)  # 跳过表头
        rows = list(reader)

    if any(len(row) < 6 for row in rows):
        raise ValueError(f"{path.name} 中存在字段不足的记录。")

    return rows


def main() -> None:
    config = load_mtg_config()

    audio_dir = Path(config["audio_dir"])
    split_dir = Path(config["split_dir"])
    feature_dir = Path(config["feature_dir"])
    tag_file = Path(config["tag_vocabulary_file"])

    print(f"项目根目录：{ROOT}")
    print(f"配置文件：{CONFIG_FILE}")

    print("\n========== 1. 路径检查 ==========")

    for name, path, valid in [
        ("audio_dir", audio_dir, audio_dir.is_dir()),
        ("split_dir", split_dir, split_dir.is_dir()),
        ("feature_dir", feature_dir, feature_dir.is_dir()),
        ("tag_vocabulary_file", tag_file, tag_file.is_file()),
    ]:
        check(name, valid, path)

    print("\n========== 2. 划分数量 ==========")

    splits = {}

    for split, filename in SPLIT_FILES.items():
        path = split_dir / filename
        check(f"{filename} 是否存在", path.is_file())

        splits[split] = read_tsv(path)
        actual = len(splits[split])

        check(
            f"{split} 样本数是否为 {EXPECTED_TRACKS[split]}",
            actual == EXPECTED_TRACKS[split],
            actual,
        )

    print("\n========== 3. 标签检查 ==========")

    official_tags = {
        tag
        for rows in splits.values()
        for row in rows
        for tag in row[5:]
    }

    tag_counts = {
        "genre": sum(
            tag.startswith("genre---")
            for tag in official_tags
        ),
        "instrument": sum(
            tag.startswith("instrument---")
            for tag in official_tags
        ),
        "mood/theme": sum(
            tag.startswith("mood/theme---")
            for tag in official_tags
        ),
        "total": len(official_tags),
    }

    for category, expected in EXPECTED_TAGS.items():
        check(
            f"{category} 标签数是否为 {expected}",
            tag_counts[category] == expected,
            tag_counts[category],
        )

    print("\n========== 4. 标签词表检查 ==========")

    tag_list = np.load(
        tag_file,
        allow_pickle=False,
    ).reshape(-1).astype(str)

    local_tags = set(tag_list)

    check(
        "标签词表数量是否为 183",
        len(tag_list) == EXPECTED_TAGS["total"],
        len(tag_list),
    )
    check(
        "标签词表是否无重复",
        len(local_tags) == len(tag_list),
    )
    check(
        "标签词表与 TSV 标签是否一致",
        local_tags == official_tags,
    )

    print("\n========== 5. 划分泄漏检查 ==========")

    # TSV 前三列依次为 track_id、artist_id、album_id
    id_fields = {
        "track_id": 0,
        "artist_id": 1,
        "album_id": 2,
    }

    split_pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    for first, second in split_pairs:
        print(f"\n{first} vs {second}")

        for field, column in id_fields.items():
            first_ids = {row[column] for row in splits[first]}
            second_ids = {row[column] for row in splits[second]}
            overlap = len(first_ids & second_ids)

            check(
                f"{field} overlap",
                overlap == 0,
                overlap,
            )

    print("\n========== 6. 联集检查 ==========")

    all_track_ids = {
        row[0]
        for rows in splits.values()
        for row in rows
    }

    check(
        f"官方联集是否为 {EXPECTED_UNION}",
        len(all_track_ids) == EXPECTED_UNION,
        len(all_track_ids),
    )

    print("\n========== 最终结果 ==========")
    print("[PASS] MTG-Jamendo 基础数据检查全部通过。")


if __name__ == "__main__":
    main()