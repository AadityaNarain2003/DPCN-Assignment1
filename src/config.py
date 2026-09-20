from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(filename: str) -> dict:
    path = PROJECT_ROOT / "config" / filename
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Resolve configured paths relative to this project.
    for section in ("data", "similarity", "network"):
        for key, value in list(config.get(section, {}).items()):
            if isinstance(value, str) and (
                key.endswith("_input")
                or key.endswith("_output")
                or key.endswith("_dir")
                or key.endswith("_matrix")
            ):
                config[section][key] = str((PROJECT_ROOT / value).resolve())

    return config
