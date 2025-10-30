import pandas as pd


def normalize_duration(x: str) -> str | None:
    if pd.isna(x):
        return None

    x = str(x).strip()
    hours, minutes = 0, 0

    if "h" in x:
        hours = int(x.split("h")[0].strip())

    if "m" in x:
        minutes = int(x.split("h")[-1].replace("m", "").strip()) if "m" in x else 0

    return f"{hours:02d}:{minutes:02d}"


def clean_csv(csv: pd.DataFrame) -> pd.DataFrame:
    # Keep only rows where Total_Stops is "non-stop" or starts with a digit
    csv = csv[csv["Total_Stops"].astype(str).str.match(r"^(?:\d|non-stop)", na=False)].copy()

    csv["Total_Stops"] = csv["Total_Stops"].apply(
        lambda x: 0 if x == "non-stop" else int(str(x)[0])
    )

    csv["Duration"] = csv["Duration"].apply(normalize_duration)

    csv["Arrival_Time"] = csv["Arrival_Time"].apply(
        lambda x: x.split(" ")[0] if " " in x else x
    )
    return csv


def clean_and_save_csv(csv: pd.DataFrame, csv_path: str) -> None:
    clean_csv(csv).to_csv(csv_path, index=False)


if __name__ == "__main__":
    df = pd.read_csv('plane ticket price.csv')
    clean_and_save_csv(df, "clean.csv")
