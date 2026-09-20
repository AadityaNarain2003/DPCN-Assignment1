import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "./../Data/Survey_Results_UC.csv"
OUTPUT_FILE = "./../Data/Data_Clean.csv"

ID_COL = "id. Response ID"

LIKERT_MAP = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5
}


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(filepath):
    return pd.read_csv(filepath)


# ============================================================
# 2. IDENTIFY QUESTION COLUMNS
# ============================================================

def get_question_columns(df, id_col):
    return [col for col in df.columns if col != id_col]


# ============================================================
# 3. CONVERT LIKERT RESPONSES TO NUMBERS
# ============================================================

def convert_likert(df, question_cols):
    for col in question_cols:
        df[col] = df[col].map(LIKERT_MAP)

    return df


# ============================================================
# 4. REMOVE COMPLETELY EMPTY ROWS
# ============================================================

def remove_empty_rows(df):
    before = len(df)

    df = df.dropna(how="all")

    removed = before - len(df)

    print(f"Completely empty rows removed: {removed}")

    return df


# ============================================================
# 5. ANALYZE MISSING RESPONSES
# ============================================================

def analyze_missing(df, question_cols):

    df["missing_count"] = df[question_cols].isna().sum(axis=1)

    df["answered_count"] = df[question_cols].notna().sum(axis=1)

    df["missing_percent"] = (
        df["missing_count"] / len(question_cols) * 100
    )

    print("\n--- Missing Data Summary ---")

    print(
        f"Participants with at least one missing response: "
        f"{(df['missing_count'] > 0).sum()}"
    )

    print(
        f"Participants with no missing responses: "
        f"{(df['missing_count'] == 0).sum()}"
    )

    print("\nMissing responses per participant:")
    print(
        df["missing_count"]
        .value_counts()
        .sort_index()
    )

    print("\nMissing percentage:")
    print(df["missing_percent"].describe())

    return df


# ============================================================
# 6. REMOVE PARTICIPANTS BASED ON MISSINGNESS
# ============================================================

def remove_high_missing(df, max_missing, question_cols):

    before = len(df)

    df = df[df["missing_count"] <= max_missing].copy()

    removed = before - len(df)

    print(
        f"\nParticipants removed with more than "
        f"{max_missing} missing responses: {removed}"
    )

    return df


# ============================================================
# 7. REMOVE MISSINGNESS INFORMATION COLUMNS
# ============================================================

def remove_missing_info_columns(df):

    columns_to_remove = [
        "missing_count",
        "answered_count",
        "missing_percent"
    ]

    df = df.drop(
        columns=[
            col for col in columns_to_remove
            if col in df.columns
        ]
    )

    return df


# ============================================================
# 8. PRINT DATASET INFORMATION
# ============================================================

def print_dataset_info(df, question_cols):

    print("\n================================")
    print("FINAL DATASET INFORMATION")
    print("================================")

    print(f"Participants: {len(df)}")
    print(f"Questions: {len(question_cols)}")
    print(f"Total cells: {len(df) * len(question_cols)}")

    print("\nMissing values:")
    print(df[question_cols].isna().sum().sum())

    print("\nDataset shape:")
    print(df.shape)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    # Load
    df = load_data(INPUT_FILE)

    # Identify questions
    question_cols = get_question_columns(df, ID_COL)

    print("Original dataset:")
    print(f"Participants: {len(df)}")
    print(f"Questions: {len(question_cols)}")

    # Convert Likert responses
    df = convert_likert(df, question_cols)

    # Remove completely empty rows
    df = remove_empty_rows(df)

    # Analyze missing data
    df = analyze_missing(df, question_cols)

    # --------------------------------------------------------
    # OPTIONAL:
    # Remove participants based on missing responses.
    #
    # Example:
    max_missing = 11
    df = remove_high_missing(df, max_missing, question_cols)
    #
    # Leave commented until you decide the cutoff.
    # --------------------------------------------------------

    # Remove analysis-only missingness columns
    # Uncomment if you DON'T want these columns in final CSV.
    #
    df = remove_missing_info_columns(df)

    # Print final information
    print_dataset_info(df, question_cols)

    # Save
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved cleaned data to: {OUTPUT_FILE}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()