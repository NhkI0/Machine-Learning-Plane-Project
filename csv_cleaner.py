import pandas as pd
import datetime as dt


def normalize_duration(x: str) -> int:
    """
    Serialize duration to hh:mm format
    :param x: -> ex: 8h 35m | 15h
    :return:  -> ex: 515    | 900
    """
    if pd.isna(x):
        return None

    x = str(x).strip()
    hours, minutes = 0, 0

    if "h" in x:
        hours = int(x.split("h")[0].strip())

    if "m" in x:
        minutes = int(x.split("h")[-1].replace("m", "").strip()) if "m" in x else 0

    return hours * 60 + minutes


def clean_csv(csv: pd.DataFrame) -> pd.DataFrame:
    """
    Return a cleaned version of the csv.
    Remove the NaN values.
    Serialize Total_Stops to just set a number instead of text.
    Serialize Duration as explained in normalize_duration().
    Serialize Arrival_Time to remove the date of arrival when specified.
    Split date and time columns into separate components.
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

    csv['Journey_day'] = pd.to_datetime(csv['Date_of_Journey'], dayfirst=True).dt.day
    csv['Journey_month'] = pd.to_datetime(csv['Date_of_Journey'], dayfirst=True).dt.month

    csv['Departure_hour'] = pd.to_datetime(csv['Dep_Time'], format='%H:%M').dt.hour
    csv['Departure_min'] = pd.to_datetime(csv['Dep_Time'], format='%H:%M').dt.minute

    csv['Arrival_hour'] = pd.to_datetime(csv['Arrival_Time'], format='%H:%M').dt.hour
    csv['Arrival_min'] = pd.to_datetime(csv['Arrival_Time'], format='%H:%M').dt.minute

    csv = csv.drop(['Date_of_Journey', 'Dep_Time', 'Arrival_Time'], axis=1)

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
