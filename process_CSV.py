import pandas as pd

import clean_dataset
import csv_cleaner


def process(path: str) -> pd.DataFrame:
    df = clean_dataset.fill_nul(path)
    df = clean_dataset.outliers_clean(df)
    df = csv_cleaner.clean_csv(df)
    return df


def process_and_save(path: str, save_to: str) -> None:
    processed = process(path)
    processed.to_csv(save_to, index=False)


if __name__ == "__main__":
    process_and_save("plane_ticket_price_original.csv", "plane_ticket_price_cleaned.csv")
