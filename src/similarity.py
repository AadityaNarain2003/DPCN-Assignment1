import numpy as np
import pandas as pd


def spearman_similarity_matrix(
    df: pd.DataFrame,
    columns: list[str],
    min_common_responses: int,
) -> pd.DataFrame:
    matrix = pd.DataFrame(
        np.nan,
        index=columns,
        columns=columns,
        dtype=float,
    )

    for i, col_i in enumerate(columns):
        for j in range(i, len(columns)):
            col_j = columns[j]

            valid = df[col_i].notna() & df[col_j].notna()
            if valid.sum() < min_common_responses:
                continue

            rho = df.loc[valid, col_i].corr(
                df.loc[valid, col_j],
                method="spearman",
            )
            matrix.iloc[i, j] = rho
            matrix.iloc[j, i] = rho

    return matrix


def participant_similarity_matrix(
    df: pd.DataFrame,
    question_columns: list[str],
    min_common_responses: int,
) -> pd.DataFrame:
    labels = [f"P{i + 1}" for i in range(len(df))]
    responses = df[question_columns].copy()
    responses.index = labels

    matrix = pd.DataFrame(np.nan, index=labels, columns=labels)

    for i, person_i in enumerate(labels):
        for j in range(i, len(labels)):
            person_j = labels[j]

            x = responses.loc[person_i]
            y = responses.loc[person_j]
            valid = x.notna() & y.notna()

            if valid.sum() < min_common_responses:
                continue

            rho = x[valid].corr(y[valid], method="spearman")
            matrix.iloc[i, j] = rho
            matrix.iloc[j, i] = rho

    return matrix
