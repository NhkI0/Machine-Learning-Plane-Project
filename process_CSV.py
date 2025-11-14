import pandas as pd

import clean_dataset
import csv_cleaner


def process(path: str) -> pd.DataFrame:
    """
    Complete data processing pipeline:
    1. Fill null values
    2. Clean and transform data
    3. Remove outliers (on transformed data)
    4. Remove duplicates (on transformed data)
    """
    print("=== Starting data processing pipeline ===")

    # Step 1: Fill null values
    print("\n[1/4] Filling null values...")
    df = clean_dataset.fill_nul(path)

    # Step 2: Clean and transform data
    print("\n[2/4] Cleaning and transforming data...")
    df = csv_cleaner.clean_csv(df)

    # Step 3: Remove outliers (on transformed data)
    print("\n[3/4] Removing outliers...")
    df = clean_dataset.remove_outliers(df)

    # Step 4: Remove duplicates (on transformed data)
    print("\n[4/4] Removing duplicates...")
    df = csv_cleaner.remove_duplicates(df)

    print("\n=== Data processing complete ===")
    return df


def process_and_save(path: str, save_to: str) -> None:
    """
    Process data and save to CSV file.
    """
    processed = process(path)
    processed.to_csv(save_to, index=False)
    print(f"\nProcessed data saved to: {save_to}")


if __name__ == "__main__":
    process_and_save("plane_ticket_price_original.csv", "plane_ticket_price_cleaned.csv")
