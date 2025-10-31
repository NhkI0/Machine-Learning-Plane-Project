import pandas as pd


def normalize_duration(x: str) -> str | None:
    """
    Serialize duration to hh:mm format
    :param x: -> ex: 8h 35m | 15h
    :return:  -> ex: 08:35  | 15:00
    """
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
    """
    Return a cleaned version of the csv.
    Remove the NaN values.
    Serialize Total_Stops to just set a number instead of text.
    Serialize Duration as explained in normalize_duration().
    Serialize Arrival_Time to remove the date of arrival when specified.
    :param csv: -> pd.DataFrame
    :return: -> pd.DataFrame
    """
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
    """
    Invoke and save the cleaned version of the csv.
    :param csv: -> pd.DataFrame
    :param csv_path: -> str
    :return:  -> None
    """
    clean_csv(csv).to_csv(csv_path, index=False)


if __name__ == "__main__":
    df = pd.read_csv('plane ticket price.csv')
    clean_and_save_csv(df, "clean.csv")
