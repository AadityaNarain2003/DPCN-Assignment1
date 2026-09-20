from pathlib import Path
import pandas as pd


def load_responses(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def select_question_columns(df: pd.DataFrame, config: dict) -> list[str]:
    data = config["data"]
    prefixes = data.get("question_prefixes")
    excluded = set(data.get("excluded_columns", []))

    if prefixes:
        prefixes = tuple(str(p).upper() for p in prefixes)
        return [
            col for col in df.columns
            if str(col).upper().startswith(prefixes)
        ]

    return [col for col in df.columns if col not in excluded]


def load_correlation_matrix(path: str) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    return matrix.apply(pd.to_numeric, errors="coerce")


def save_matrix(matrix: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(path)
