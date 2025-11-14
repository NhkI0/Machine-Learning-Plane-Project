import pandas as pd


def outliers_clean(df: pd.DataFrame) -> pd.DataFrame:
    """This script aim to clean outliers from the dataset"""

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


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers from the dataset using IQR method.
    This function removes rows that contain outliers in any numerical column.

    :param df: Input DataFrame
    :return: DataFrame with outliers removed
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)

    for col in df_clean.select_dtypes(include=['float64', 'int64']):
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        borne_min = q1 - 1.5 * iqr
        borne_max = q3 + 1.5 * iqr

        # Keep only rows within bounds for this column
        df_clean = df_clean[(df_clean[col] >= borne_min) & (df_clean[col] <= borne_max)]
        print(f"{col}: Removed {initial_rows - len(df_clean)} outlier rows")
        initial_rows = len(df_clean)

    print(f"Total rows after outlier removal: {len(df_clean)}")
    return df_clean


def fill_nul(path: str) -> pd.DataFrame:
    """This script aims to fill empty values in the dataset"""

    # Loading dataset
    df = pd.read_csv(path)

    # Filling empty number values
    for col in df.select_dtypes(include=['float64', 'int64']):
        mediane = df[col].median()
        df[col] = df[col].fillna(mediane)
        print(f"Add value {mediane} at {col}")

    # Filling empty category values
    for col in df.select_dtypes(include=['object', 'category']):
        if len(df[col].mode()) > 0:
            mode = df[col].mode()[0]
            df[col] = df[col].fillna(mode)
            print(f"Add value {mode} at {col}")

    return df


if __name__ == "__main__":
    outliers_clean(fill_nul("./plane_ticket_price_original.csv"))
