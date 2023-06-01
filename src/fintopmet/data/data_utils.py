import enum
import re
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import fintopmet


def day_cols(df: pd.DataFrame) -> list[str]:
    return df.columns[df.columns.str.startswith("d_")].tolist()


def subset_day_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df[day_cols(df)]


def load_evaluation() -> pd.DataFrame:
    return pd.read_csv(fintopmet.fp.DATA / "m5" / "sales_train_evaluation.csv")


def load_calendar() -> pd.DataFrame:
    return pd.read_csv(fintopmet.fp.DATA / "m5" / "calendar.csv", parse_dates=["date"])


def load_sell_prices() -> pd.DataFrame:
    return pd.read_csv(fintopmet.fp.DATA / "m5" / "sell_prices.csv")


def load_validation() -> pd.DataFrame:
    return pd.read_csv(fintopmet.fp.DATA / "m5" / "sales_train_validation.csv")


class TimeSeriesData:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.calendar = Calendar()

    def query(self, query: str) -> pd.DataFrame:
        return self.data.query(query)

    def query_id(self, id: str) -> pd.DataFrame:
        df = self.query(f"id == '{id}'").pipe(subset_day_cols).T
        df.columns = [id]
        df.index = df.index.map(self.calendar.date_map)
        return df


def filter_path_entries(path, reg):
    regex = re.compile(reg)
    return [entry for entry in path.iterdir() if regex.search(entry.name)]


def pd_to_pl(df: pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(df)


def list_files(glob: str) -> list[Path]:
    return list((fintopmet.fp.DATA / "m5" / "proc").glob(glob))


class Data(enum.Enum):
    EVALUATION = "evaluation"
    CALENDAR = "calendar"
    SELL_PRICES = "sell_prices"
    VALIDATION = "validation"


def strip_multiindex(df: pd.DataFrame):
    return df.rename_axis(columns=lambda x: None, index=lambda x: None)


def load(data: str | enum.Enum):
    if isinstance(data, enum.Enum):
        data = data.value
    match data:
        case "evaluation":
            return load_evaluation()
        case "calendar":
            return load_calendar()
        case "sell_prices":
            return load_sell_prices()
        case "validation":
            return load_validation()
        case _:
            raise ValueError


# --------------------- Melt --------------------- #
INDEX_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
ID_COLS = ["id", "d"]


def melt_sales(df):
    return pd.melt(df, id_vars=INDEX_COLUMNS, var_name="d", value_name="sales")


def add_pred_dates(df, end_date: int = 1941, h: int = 28):
    """Assigns dates to the prediction horizon"""
    return pd.concat(
        (
            df[INDEX_COLUMNS].assign(**{"d": f"d_{i}", "sales": np.nan})
            for i in range(end_date + 1, end_date + 1 + h)
        )
    ).reset_index(drop=True)


def id_cols_to_cats(df):
    """Converts index columns to categorical dtype"""
    for col in INDEX_COLUMNS:
        df[col] = df[col].astype("category")
    return df


# ---------------------  --------------------- #


def collapse_events(df, col):
    return (
        pd.get_dummies(df.set_index("date")[col])
        .reset_index()
        .melt(id_vars=["date"], var_name="event", value_name="happened")
    )


def _event_gp(df, infix):
    return pd.concat(
        (
            collapse_events(df, col=f"event_{infix}_1"),
            collapse_events(df, col=f"event_{infix}_2"),
        ),
        axis=0,
    ).groupby(["date", "event"])


def event_counts(df, gran: bool = False) -> pd.DataFrame:
    if gran:
        infix = "name"
    else:
        infix = "type"
    return _event_gp(df, infix=infix).happened.sum().astype(int).unstack(1)


def event_dummies(df, gran: bool = False) -> pd.DataFrame:
    if gran:
        infix = "name"
    else:
        infix = "type"
    return _event_gp(df, infix=infix).happened.any().astype(int).unstack(1)


def calendar_dummies(df):
    return pd.concat(
        (
            pd.get_dummies(df.set_index("date")["wday"], prefix="wday"),
            pd.get_dummies(df.set_index("date")["month"], prefix="month"),
            # We forecast in the same year; so dummies make sense for final year
            pd.get_dummies(df.set_index("date")["year"], prefix="year"),
        ),
        axis=1,
    ).astype(int)


def snap_dummies(df) -> pd.DataFrame:
    return df.set_index("date")[["snap_CA", "snap_TX", "snap_WI"]]


calendar_cols = ["wm_yr_wk", "weekday", "wday", "month", "year"]


class Calendar:
    def __init__(self):
        self.df = load_calendar()

    @cached_property
    def date_map(self):
        return dict(zip(self.df.d, self.df.date))

    def all_dummies(self) -> pd.DataFrame:
        return pd.concat(
            (
                calendar_dummies(self.df),
                event_dummies(self.df, gran=False),
                event_dummies(self.df, gran=True),
                snap_dummies(self.df),
            ),
            axis=1,
        )
