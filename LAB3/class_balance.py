import yaml
from pathlib import Path
from tqdm import tqdm
from pprint import pp

from pathlib import Path
import yaml


def load_name_to_id(yaml_path: Path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    names = data["names"]

    if isinstance(names, dict):
        id_to_name = {int(k): v for k, v in names.items()}
    else:
        id_to_name = {i: v for i, v in enumerate(names)}

    name_to_id = {v: k for k, v in id_to_name.items()}

    return name_to_id


def build_id_mapping(name_to_id: dict, mapping: dict):
    """
    mapping: {old_name: new_name}
    """
    id_mapping = {}

    for old_name, new_name in mapping.items():
        if old_name not in name_to_id:
            raise ValueError(f"{old_name} not found in YAML names")

        if new_name not in name_to_id:
            raise ValueError(f"{new_name} not found in YAML names")

        old_id = name_to_id[old_name]
        new_id = name_to_id[new_name]

        id_mapping[old_id] = new_id

    return id_mapping


def remap_label_file(src_path: Path, dst_path: Path, id_mapping: dict):
    if not src_path.exists():
        return

    with open(src_path, "r") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        # remap if needed
        class_id = id_mapping.get(class_id, class_id)

        new_line = f"{class_id} {' '.join(parts[1:])}\n"
        new_lines.append(new_line)

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dst_path, "w") as f:
        f.writelines(new_lines)


def remap_dataset(
    labels_dir: Path,
    output_labels_dir: Path,
    yaml_path: Path,
    mapping: dict,
):
    name_to_id = load_name_to_id(yaml_path)
    id_mapping = build_id_mapping(name_to_id, mapping)

    print("ID mapping:", id_mapping)

    for label_path in tqdm(labels_dir.rglob("*.txt")):
        relative_path = label_path.relative_to(labels_dir)
        dst_path = output_labels_dir / relative_path

        remap_label_file(label_path, dst_path, id_mapping)

    print(f"Remapped labels saved to: {output_labels_dir}")


def count_class_ids(labels_dir):
    from collections import Counter
    cnt = Counter()

    for p in labels_dir.rglob("*.txt"):
        with open(p) as f:
            for line in f:
                cls = int(line.split()[0])
                cnt[cls] += 1
    return cnt


if __name__ == "__main__":
    src_dataset_path = Path("data/dm-2026-lab-3-object-detection/YOLO/labels")
    dst_dataset_path = Path("data/dm-2026-lab-3-object-detection/YOLO/labels_mapped")
    yaml_path = Path("data/dm-2026-lab-3-object-detection/YOLO/yolo_dataset.yaml")
    mapping = {
        "trafficLight-GreenLeft": "trafficLight-Green",
        "trafficLight-Yellow": "trafficLight",
        "trafficLight-YellowLeft": "trafficLight",
        "trafficLight-RedLeft": "trafficLight-Red",
    }

    remap_dataset(src_dataset_path, dst_dataset_path, yaml_path, mapping)
    print(f"Classes in dataset before mapping")
    pp(sorted(dict(count_class_ids(src_dataset_path)).items()))
    print(f"Classes in dataset after mapping")
    pp(sorted(dict(count_class_ids(dst_dataset_path)).items()))