import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import load_config
from src.data import load_responses, save_matrix
from src.data import select_question_columns
from src.similarity import spearman_similarity_matrix


def main():
    config = load_config("questions.yaml")
    df = load_responses(config["data"]["raw_input"])
    question_columns = select_question_columns(df, config)

    matrix = spearman_similarity_matrix(
        df,
        question_columns,
        config["similarity"]["min_common_responses"],
    )

    output = config["similarity"]["output_matrix"]
    save_matrix(matrix, output)

    print(f"Questions: {len(question_columns)}")
    print(f"Saved: {output}")

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        matrix, cmap="coolwarm", vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.2,
        xticklabels=False, yticklabels=False,
        cbar_kws={"label": "Spearman Correlation"},
    )
    plt.title("Question-to-Question Spearman Correlation")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
