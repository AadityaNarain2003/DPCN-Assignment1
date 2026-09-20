import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import load_config
from src.config import load_config
from src.data import load_responses, save_matrix
from src.similarity import participant_similarity_matrix


def main():
    config = load_config("people.yaml")
    df = load_responses(config["data"]["raw_input"])

    excluded = set(config["data"]["excluded_columns"])
    question_columns = [c for c in df.columns if c not in excluded]

    matrix = participant_similarity_matrix(
        df,
        question_columns,
        config["similarity"]["min_common_responses"],
    )

    output = config["similarity"]["output_matrix"]
    save_matrix(matrix, output)

    print(f"Participants: {len(df)}")
    print(f"Questions used: {len(question_columns)}")
    print(f"Saved: {output}")

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        matrix, cmap="coolwarm", vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.2,
        xticklabels=False, yticklabels=False,
        cbar_kws={"label": "Spearman Correlation"},
    )
    plt.title("Person-to-Person Spearman Correlation")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
