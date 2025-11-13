import pandas as pd


def outliers_clean(df: pd.DataFrame) -> pd.DataFrame:
    """This script aim to clean ouliers from the dataset"""

    # Counting outliers values changed
    counter = 0

    for col in df.select_dtypes(include=['float64', 'int64']):
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        borne_min = q1 - 1.5 * iqr
        borne_max = q3 + 1.5 * iqr
        outliers = df[(df[col] < borne_min) | (df[col] > borne_max)]
        counter += 1
        print(f"{col} : {len(outliers)} valeurs aberrantes")
        print(f"{counter} outliers changed")

    return df


def fill_nul(path: str) -> pd.DataFrame:
    """This script aims to fill empty values in the dataset"""

    # Loading dataset
    df = pd.read_csv(path)

    # Filling empty number values
    for col in df.select_dtypes(include=['float64', 'int64']):
        mediane = df[col].median()
        df[col].fillna(mediane)
        print(f"Add value {mediane} at {col}")

    # Filling empty category values
    for col in df.select_dtypes(include=['object', 'category']):
        mode = df[col].mode()[0]
        df[col].fillna(mode)
        print(f"Add value {mode} at {col}")

    return df


if __name__ == "__main__":
    outliers_clean(fill_nul("./plane_ticket_price_original.csv"))
