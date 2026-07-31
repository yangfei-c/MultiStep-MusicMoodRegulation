import csv
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "configs" / "dataset.yaml"

SPLITS = ("train", "validation", "test")

EXPECTED = {
    "deam": {
        "train": 1261,
        "validation": 271,
        "test": 270,
        "total": 1802,
    },
    "pmemo": {
        "train": 536,
        "validation": 116,
        "test": 115,
        "total": 767,
    },
}

REQUIRED_COLUMNS = {
    "song_id",
    "valence_mean",
    "valence_std",
    "arousal_mean",
    "arousal_std",
}


def check(name: str, condition: bool, value=None) -> None:
    """输出检查结果，失败时停止运行。"""
    message = name if value is None else f"{name}：{value}"

    if not condition:
        raise RuntimeError(f"[FAIL] {message}")

    print(f"[PASS] {message}")


def load_config() -> dict:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_split(path: Path) -> set[int]:
    """读取每行一个 song_id 的划分文件。"""
    ids = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    check(f"{path.name} 是否无重复 ID", len(ids) == len(set(ids)))
    return set(ids)


def read_annotations(path: Path) -> set[int]:
    """读取静态 VA 标注并检查字段和数值。"""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])

        check(
            "静态标注字段是否完整",
            REQUIRED_COLUMNS <= columns,
            sorted(columns),
        )

        rows = list(reader)

    ids = []

    for row_number, row in enumerate(rows, start=2):
        try:
            song_id = int(float(row["song_id"]))

            float(row["valence_mean"])
            float(row["valence_std"])
            float(row["arousal_mean"])
            float(row["arousal_std"])

        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path.name} 第 {row_number} 行存在无效数据。"
            ) from error

        ids.append(song_id)

    check("静态标注 song_id 是否无重复", len(ids) == len(set(ids)))
    return set(ids)


def get_audio_ids(audio_dir: Path) -> set[int]:
    """提取数字命名的 MP3 文件 ID。"""
    return {
        int(path.stem)
        for path in audio_dir.glob("*.mp3")
        if path.stem.isdigit()
    }


def get_feature_ids(feature_dir: Path) -> set[int]:
    """提取数字命名的 MERT 特征目录 ID。"""
    return {
        int(path.name)
        for path in feature_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    }


def check_dataset(name: str, config: dict) -> None:
    print(f"\n{'=' * 16} {name.upper()} {'=' * 16}")

    audio_dir = Path(config["audio_dir"])
    split_dir = Path(config["split_dir"])
    feature_dir = Path(config["feature_dir"])
    annotation_file = Path(config["static_annotation_file"])

    print("\n========== 1. 路径检查 ==========")

    check("audio_dir", audio_dir.is_dir(), audio_dir)
    check("split_dir", split_dir.is_dir(), split_dir)
    check("feature_dir", feature_dir.is_dir(), feature_dir)
    check(
        "static_annotation_file",
        annotation_file.is_file(),
        annotation_file,
    )

    print("\n========== 2. 静态标注检查 ==========")

    annotation_ids = read_annotations(annotation_file)
    expected_total = EXPECTED[name]["total"]

    check(
        f"静态标注数量是否为 {expected_total}",
        len(annotation_ids) == expected_total,
        len(annotation_ids),
    )

    print("\n========== 3. 划分检查 ==========")

    split_ids = {}

    for split in SPLITS:
        split_file = split_dir / f"{split}.txt"

        check(f"{split}.txt 是否存在", split_file.is_file())

        split_ids[split] = read_split(split_file)
        expected_count = EXPECTED[name][split]

        check(
            f"{split} 数量是否为 {expected_count}",
            len(split_ids[split]) == expected_count,
            len(split_ids[split]),
        )

    print("\n========== 4. 重叠与并集 ==========")

    for first, second in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = split_ids[first] & split_ids[second]

        check(
            f"{first} 与 {second} overlap",
            len(overlap) == 0,
            len(overlap),
        )

    union_ids = set().union(*split_ids.values())

    check(
        f"划分并集是否为 {expected_total}",
        len(union_ids) == expected_total,
        len(union_ids),
    )
    check(
        "划分 ID 与静态标注 ID 是否完全一致",
        union_ids == annotation_ids,
    )

    print("\n========== 5. 音频和特征覆盖 ==========")

    audio_ids = get_audio_ids(audio_dir)
    feature_ids = get_feature_ids(feature_dir)

    missing_audio = union_ids - audio_ids
    missing_features = union_ids - feature_ids

    check(
        "所有划分样本是否都有 MP3",
        len(missing_audio) == 0,
        len(missing_audio),
    )
    check(
        "所有划分样本是否都有特征目录",
        len(missing_features) == 0,
        len(missing_features),
    )

    print(f"本地 MP3 数量：{len(audio_ids)}")
    print(f"本地特征目录数量：{len(feature_ids)}")
    print(f"额外 MP3 数量：{len(audio_ids - union_ids)}")
    print(f"额外特征目录数量：{len(feature_ids - union_ids)}")

    print(f"\n[PASS] {name.upper()} 基础数据检查全部通过。")


def main() -> None:
    config = load_config()

    print(f"项目根目录：{ROOT}")
    print(f"配置文件：{CONFIG_FILE}")

    for name in ("deam", "pmemo"):
        check(
            f"dataset.yaml 是否包含 {name}",
            name in config,
        )
        check_dataset(name, config[name])

    print("\n========== 最终结果 ==========")
    print("[PASS] DEAM 和 PMEmo 基础数据检查全部通过。")


if __name__ == "__main__":
    main()