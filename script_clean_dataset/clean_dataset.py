
import pandas as pd

def outliers_clean():
    "This script aim to clean ouliers from the dataset"

    # Loading dataset
    df = pd.read_csv("/Users/alexfougeroux/Documents/ESAIP/ING4/Machine_learning/Machine-Learning-Plane-Project/script_clean_dataset/plane_ticket_price_modified.csv")

    # Counting outliers values changed
    counter = 0

    for col in df.select_dtypes(include=['float64', 'int64']):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        borne_min = Q1 - 1.5 * IQR
        borne_max = Q3 + 1.5 * IQR
        outliers = df[(df[col] < borne_min) | (df[col] > borne_max)]
        counter += 1
        print(f"{col} : {len(outliers)} valeurs aberrantes")
        print(f"{counter} outliers changed")

def fill_nul():
    "This script aims to fill empty values in the dataset"

    # Loading dataset
    df = pd.read_csv("/Users/alexfougeroux/Documents/ESAIP/ING4/Machine_learning/Machine-Learning-Plane-Project/script_clean_dataset/plane_ticket_price_modified.csv")

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

if __name__ == "__main__":
    fill_nul()
    outliers_clean()
