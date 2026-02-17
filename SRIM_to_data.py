import re

import pandas as pd
from io import StringIO


def SRIM_to_dataframe(file_path: str):
    start = '  --------------  ---------- ---------- ----------  ----------  ----------\n'
    end = "-----------------------------------------------------------\n"

    lines = []

    with open(file_path) as infile:
        copy = False
        for line in infile:
            if line == start:
                copy = True
                continue
            elif line == end:
                copy = False
                continue
            elif copy:
                lines.append(line.strip())

    columns = ["ionE MeV", "elec", "nuc", "range um", "long", "lat"]

    df = pd.DataFrame({'string': lines})  # new dataframe

    df = df['string'].str.split('\s{2,}', expand=True)  # split by 2 or more spaces

    df.columns = columns
    df = df[["ionE MeV", "range um"]]

    return df


def change_ion_units(row):
    value = re.findall("\d+\.\d+", row)[0]
    value = (float(value))
    units = re.findall("[a-zA-Z]+", row)[0].lower()

    if units == "keV".lower():
        value = value / 1_000

    return value


def change_range_units(row):
    value = re.findall("\d+\.*\d+", row)[0]
    value = (float(value))
    units = re.findall("[a-zA-Z]+", row)[0].lower()

    if units == "mm".lower():
        value = value * 1_000

    return value


def transform_dataframe(df):
    df['ionE MeV'] = df['ionE MeV'].apply(change_ion_units)
    df['range um'] = df['range um'].apply(change_range_units)

    return df


def process_file(file_path: str):
    df = SRIM_to_dataframe(file_path)
    df = transform_dataframe(df)

    return df
