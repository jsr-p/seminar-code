import numpy as np
import pandas as pd
from tqdm import tqdm

import fintopmet

SHIFT_DAY = 28
TARGET = "sales"
LAG_DAYS = [col for col in range(SHIFT_DAY, SHIFT_DAY + 15)]


def assign_lags(grid_df):
    grid_df = grid_df.assign(
        **{
            f"{TARGET}_lag_{lag}": grid_df.groupby(["id"])[TARGET]
            .transform(lambda x: x.shift(lag))
            .astype(np.float32)
            for lag in tqdm(LAG_DAYS, "assigning lags")
        }
    )
    return grid_df


def assign_rolling(grid_df):
    """
    Rollings
    with 28 day shift
    """
    for i in [7, 14, 30, 60, 180]:
        print("Rolling period:", i)
        grid_df["rolling_mean_" + str(i)] = (
            grid_df.groupby(["id"])[TARGET]
            .transform(lambda x: x.shift(SHIFT_DAY).rolling(i).mean())
            .astype(np.float32)
        )
        grid_df["rolling_std_" + str(i)] = (
            grid_df.groupby(["id"])[TARGET]
            .transform(lambda x: x.shift(SHIFT_DAY).rolling(i).std())
            .astype(np.float32)
        )
    # Rollings
    # with sliding shift
    for d_shift in [1, 7, 14]:
        print("Shifting period:", d_shift)
        for d_window in [7, 14, 30, 60]:
            col_name = "rolling_mean_tmp_" + str(d_shift) + "_" + str(d_window)
            grid_df[col_name] = (
                grid_df.groupby(["id"])[TARGET]
                .transform(lambda x: x.shift(d_shift).rolling(d_window).mean())
                .astype(np.float32)
            )
    return grid_df


def mean_encodings(grid_df):
    grid_df["sales"][grid_df["d"] > (1941 - 28)] = np.nan
    base_cols = list(grid_df)
    icols = [
        ["state_id"],
        ["store_id"],
        ["cat_id"],
        ["dept_id"],
        ["state_id", "cat_id"],
        ["state_id", "dept_id"],
        ["store_id", "cat_id"],
        ["store_id", "dept_id"],
        ["item_id"],
        ["item_id", "state_id"],
        ["item_id", "store_id"],
    ]
    for col in icols:
        print("Encoding", col)
        col_name = "_" + "_".join(col) + "_"
        grid_df["enc" + col_name + "mean"] = (
            grid_df.groupby(col)["sales"].transform("mean").astype(np.float32)
        )
        grid_df["enc" + col_name + "std"] = (
            grid_df.groupby(col)["sales"].transform("std").astype(np.float32)
        )

    keep_cols = [col for col in list(grid_df) if col not in base_cols]
    grid_df = grid_df[["id", "d"] + keep_cols]
    return grid_df
