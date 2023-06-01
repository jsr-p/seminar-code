import enum
import itertools as it
import pickle
import random
from collections.abc import Iterable
from multiprocessing import Pool
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import psutil
from tqdm import tqdm

import fintopmet
from fintopmet import data_utils, log_utils

LOGGER = log_utils.get_logger(__name__)
lgb.register_logger(LOGGER)
# --------------------- Utils --------------------- #


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)


SEED = 42
seed_everything(SEED)


class Aggregations(enum.Enum):
    STORE = "STORE"
    STORE_CAT = "STORE_CAT"
    STORE_DEPT = "STORE_DEPT"


# --------------------- Mappings for aggregations --------------------- #

LGB_PARAMS = {
    Aggregations.STORE.value: {
        "boosting_type": "gbdt",
        "objective": "tweedie",
        "tweedie_variance_power": 1.1,
        "metric": "rmse",
        "subsample": 0.5,
        "subsample_freq": 1,
        "learning_rate": 0.015,
        "num_leaves": 2**11 - 1,
        "min_data_in_leaf": 2**12 - 1,
        "feature_fraction": 0.5,
        "max_bin": 100,
        "n_estimators": 3000,
        "boost_from_average": False,
        "verbose": -1,
        "seed": SEED,
        "num_threads": 4,
    },
    Aggregations.STORE_CAT.value: {
        "boosting_type": "gbdt",
        "objective": "tweedie",
        "tweedie_variance_power": 1.1,
        "metric": "rmse",
        "subsample": 0.5,
        "subsample_freq": 1,
        "learning_rate": 0.015,
        "num_leaves": 2**8 - 1,
        "min_data_in_leaf": 2**8 - 1,
        "feature_fraction": 0.5,
        "max_bin": 100,
        "n_estimators": 3000,
        "boost_from_average": False,
        "verbose": -1,
        "seed": SEED,
        "num_threads": 4,
    },
    Aggregations.STORE_DEPT.value: {
        "boosting_type": "gbdt",
        "objective": "tweedie",
        "tweedie_variance_power": 1.1,
        "metric": "rmse",
        "subsample": 0.5,
        "subsample_freq": 1,
        "learning_rate": 0.015,
        "num_leaves": 2**8 - 1,
        "min_data_in_leaf": 2**8 - 1,
        "feature_fraction": 0.5,
        "max_bin": 100,
        "n_estimators": 3000,
        "boost_from_average": False,
        "verbose": -1,
        "num_threads": 4,
    },
}

MEANS_FEATURES = {
    Aggregations.STORE.value: [
        "enc_cat_id_mean",
        "enc_cat_id_std",
        "enc_dept_id_mean",
        "enc_dept_id_std",
        "enc_item_id_mean",
        "enc_item_id_std",
    ],
    Aggregations.STORE_CAT.value: [
        "enc_store_id_dept_id_mean",
        "enc_store_id_dept_id_std",
        "enc_item_id_store_id_mean",
        "enc_item_id_store_id_std",
    ],
    Aggregations.STORE_DEPT.value: [
        "enc_item_id_store_id_mean",
        "enc_item_id_store_id_std",
    ],
}


# --------------------- Globals from winner's notebook --------------------- #
VER = "priv"
KKK = 0
N_CORES = psutil.cpu_count()


# LIMITS and const
TARGET = "sales"
START_TRAIN = 0
END_TRAIN = 1941 - 28 * KKK
P_HORIZON = 28
USE_AUX = False

REMOVE_FEATURES = ["id", "state_id", "store_id", "date", "wm_yr_wk", "d", TARGET] + [
    "item_id",
]
CAT_COLS = ["cat_id", "dept_id"]

ORIGINAL = fintopmet.fp.M5
BASE = fintopmet.fp.PROC / "grid_part1.parquet"
PRICE = fintopmet.fp.PROC / "grid_part2.parquet"
CALENDAR = fintopmet.fp.PROC / "grid_part3.parquet"
LAGS = fintopmet.fp.PROC / "lags_df_28.parquet"
MEAN_ENC = fintopmet.fp.PROC / "mean_encoding_df.parquet"
PROCESSED_DATA_DIR = fintopmet.fp.PROC


# SPLITS for lags creation
SHIFT_DAY = 28
N_LAGS = 15
LAGS_SPLIT = [col for col in range(SHIFT_DAY, SHIFT_DAY + N_LAGS)]


# First entry is shift day; second entry is roll wind
ROLS_SPLIT = []
for i in [1, 7, 14]:
    for j in [7, 14, 30, 60]:
        ROLS_SPLIT.append([i, j])


STORES_IDS = [
    "CA_1",
    "CA_2",
    "CA_3",
    "CA_4",
    "TX_1",
    "TX_2",
    "TX_3",
    "WI_1",
    "WI_2",
    "WI_3",
]

DEPTS = [
    "HOBBIES_1",
    "HOBBIES_2",
    "HOUSEHOLD_1",
    "HOUSEHOLD_2",
    "FOODS_1",
    "FOODS_2",
    "FOODS_3",
]

CATS = ["HOBBIES", "HOUSEHOLD", "FOODS"]


PRICE_COLS = [
    "sell_price",
    "price_max",
    "price_min",
    "price_std",
    "price_mean",
    "price_norm",
    "price_nunique",
    "price_momentum",
    "price_momentum_m",
    "price_momentum_y",
]

DATE_COLS = [
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "tm_d",
    "tm_w",
    "tm_m",
    "tm_y",
    "tm_wm",
    "tm_dw",
    "tm_w_end",
]


def df_parallelize_run(func, t_split):
    num_cores = np.min([N_CORES, len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=1)
    pool.close()
    pool.join()
    return df


def read_base_data():
    df = pd.concat(
        [
            pd.read_parquet(BASE),
            pd.read_parquet(PRICE)[PRICE_COLS],
            pd.read_parquet(CALENDAR)[DATE_COLS],
        ],
        axis=1,
    )
    return df


def load_means():
    return pd.read_parquet(MEAN_ENC)


def load_lags():
    return pd.read_parquet(LAGS).filter(regex="sales_lag|rolling", axis=1)


def load_all(agg: str):
    LOGGER.info(f"Loading data for {agg}")
    return pd.concat(
        (
            read_base_data(),
            load_means()[MEANS_FEATURES[agg]],
            load_lags().filter(regex="sales_lag|rolling", axis=1),
        ),
        axis=1,
    )


class ProcData:
    def __init__(self, agg: str):
        self.df = load_all(agg)
        self._init()
        self._set_query_fn(agg)

    def _init(self):
        self.features = [col for col in self.df.columns if col not in REMOVE_FEATURES]
        self.target = TARGET
        self.cols = ["id", "d", TARGET] + self.features

    def _set_query_fn(self, agg: str):
        match agg:
            case Aggregations.STORE.value:
                query_fn = self.get_store
            case Aggregations.STORE_DEPT.value:
                query_fn = self.get_store_dept
            case Aggregations.STORE_CAT.value:
                query_fn = self.get_store_cat
            case _:
                raise ValueError
        self.query_fn = query_fn

    def get_store(self, store) -> pd.DataFrame:
        return self.df.query(f"d >= {START_TRAIN} and store_id == '{store}'")[self.cols]

    def get_store_dept(self, store, dept):
        return self.df.query(
            f"d >= {START_TRAIN} and store_id == '{store}' " f"and dept_id == '{dept}'"
        )[self.cols]

    def get_store_cat(self, store, cat):
        return self.df.query(
            f"d >= {START_TRAIN} and store_id == '{store}' " f"and cat_id == '{cat}'"
        )[self.cols]


# Recombine Test set after training
def get_base_test():
    base_test = pd.DataFrame()

    for store_id in STORES_IDS:
        temp_df = pd.read_parquet(
            PROCESSED_DATA_DIR / ("test_" + store_id + ".parquet")
        )
        temp_df["store_id"] = store_id
        base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)

    return base_test


# --------------------- Models --------------------- #


def construct_masks(grid_df):
    train_mask = grid_df["d"] <= END_TRAIN
    valid_mask = train_mask & (grid_df["d"] > (END_TRAIN - P_HORIZON))
    preds_mask = (grid_df["d"] > (END_TRAIN - 100)) & (
        grid_df["d"] <= END_TRAIN + P_HORIZON
    )
    return train_mask, valid_mask, preds_mask


def prepare_data(grid_df: pd.DataFrame, features_columns: list[str]):
    train_mask, valid_mask, preds_mask = construct_masks(grid_df)
    train_data = lgb.Dataset(
        grid_df[train_mask][features_columns], label=grid_df[train_mask][TARGET]
    )

    valid_data = lgb.Dataset(
        grid_df[valid_mask][features_columns], label=grid_df[valid_mask][TARGET]
    )

    grid_df = grid_df[preds_mask].reset_index(drop=True)
    keep_cols = [col for col in list(grid_df) if "_tmp_" not in col]
    grid_df = grid_df[keep_cols]
    d_sales = grid_df[["d", "sales"]]
    substitute = d_sales["sales"].values
    substitute[(d_sales["d"] > END_TRAIN)] = np.nan
    grid_df["sales"] = substitute

    return train_data, valid_data, grid_df


def feat_imp(estimator):
    return pd.DataFrame(
        {"name": estimator.feature_name(), "imp": estimator.feature_importance()}
    ).sort_values("imp", ascending=False)


# --------------------- Save utils --------------------- #


def get_save_name(
    agg: str,
    vals,
    ftype: str = "model",
    model: Optional[str] = None,
    with_ver: bool = False,
    file_suffix: str = ".bin",
) -> str:
    if with_ver:
        version = f"_v{VER}"
    else:
        version = ""
    if model:
        model_name = f"_{model}"
    else:
        model_name = ""
    match agg:
        case Aggregations.STORE.value:
            (store_id,) = vals
            fname = f"{ftype}_{store_id}{model_name}{version}{file_suffix}"
        case Aggregations.STORE_DEPT.value:
            store_id, dept = vals
            fname = f"{ftype}_{store_id}_{dept}{model_name}{version}{file_suffix}"
        case Aggregations.STORE_CAT.value:
            store_id, cat = vals
            fname = f"{ftype}_{store_id}_{cat}{model_name}{version}{file_suffix}"
        case _:
            raise ValueError
    return fname


def load_model(agg, model_type, with_vers, *vals):
    model_file = fintopmet.fp.MODELS / get_save_name(
        agg, vals, model=model_type, ftype="lgb-model", with_ver=with_vers
    )
    with open(model_file, "rb") as file:
        LOGGER.debug(
            f"Loading model {model_file} for {agg=}, {model_type=}, {with_vers=}"
        )
        return pickle.load(file)


def load_nonrec_model(agg, model_type, *vals):
    model_file = fintopmet.fp.MODELS / get_save_name(
        agg, vals, model=model_type, ftype="model", with_ver=False
    )
    with open(model_file, "rb") as file:
        LOGGER.debug(f"Loading model {model_file} for {agg=}, {model_type=}")
        return pickle.load(file)


def get_test_file(agg, model_type, with_ver, *vals, fp=fintopmet.fp.PROC):
    file = get_save_name(
        agg,
        vals,
        model=model_type,
        ftype="test",
        with_ver=with_ver,
        file_suffix=".parquet",
    )
    return fp / file


def load_test_data(agg, model_type, *vals, fp=fintopmet.fp.PROC):
    return pd.read_parquet(get_test_file(agg, model_type, *vals, fp=fp))


# --------------------- Training --------------------- #


def main_train(agg: str | Aggregations):
    """
    Recursive model traning for the m5 competition.
    """
    if isinstance(agg, Aggregations):
        agg = agg.value
    LOGGER.info(f"Initializing training loop with {agg=}")
    data = ProcData(agg)
    iterable = get_iterable(agg)
    lgb_params = LGB_PARAMS[agg]
    feature_columns = data.features
    for vals in iterable:
        data_file = PROCESSED_DATA_DIR / get_save_name(
            agg,
            vals,
            model="recursive",
            ftype="test",
            with_ver=False,
            file_suffix=".parquet",
        )
        grid_df = data.query_fn(*vals)
        model_file = fintopmet.fp.MODELS / get_save_name(
            agg, vals, model="recursive", ftype="lgb-model", with_ver=True
        )
        LOGGER.debug(f"Model file: {model_file=}")
        if model_file.exists() and not data_file.exists():
            LOGGER.info(
                f"Model {model_file} already exists ({vals=}) but data does not; processing and skipping to next"
            )
            train_data, valid_data, grid_df = prepare_data(grid_df, feature_columns)
            grid_df.to_parquet(data_file)
            continue
        elif model_file.exists() and data_file.exists():
            LOGGER.info(
                f"Model {model_file} already exists ({vals=}); skipping to next"
            )
            continue
        train_data, valid_data, grid_df = prepare_data(grid_df, feature_columns)
        grid_df.to_parquet(data_file)
        LOGGER.info(f"Training store with {agg=} and {vals=}")
        seed_everything(SEED)
        estimator = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[valid_data],
            callbacks=[lgb.log_evaluation(period=100)],
        )
        with open(model_file, "wb") as file:
            pickle.dump(estimator, file)

        feat_imp(estimator).to_csv(
            fintopmet.fp.DATA
            / "featimp"
            / get_save_name(
                agg,
                vals,
                model="recursive",
                ftype="featimp",
                with_ver=True,
                file_suffix=".parquet",
            ),
            index=False,
        )
        LOGGER.info(f"Saved model to {model_file}")


# --------------------- Non recursive models --------------------- #

FIRST_DAY = 710
REMOVE_FEATURE = [
    "id",
    "state_id",
    "store_id",
    #                   'item_id',
    #                   'dept_id',
    #                   'cat_id',
    "date",
    "wm_yr_wk",
    "d",
    "sales",
]

cat_var = ["item_id", "dept_id", "store_id", "cat_id", "state_id"] + [
    "event_name_1",
    "event_name_2",
    "event_type_1",
    "event_type_2",
]
cat_var = list(set(cat_var) - set(REMOVE_FEATURE))

VALIDATION = {
    "cv1": [1551, 1610],
    "cv2": [1829, 1857],
    "cv3": [1857, 1885],
    "cv4": [1885, 1913],
    "public": [1913, 1941],
    "private": [1941, 1969],
}
CV_PRIVATE = VALIDATION["private"]


cvs = ["private"]


def get_iterable(agg: str) -> Iterable:
    match agg:
        case Aggregations.STORE.value:
            iterable = it.product(STORES_IDS)
        case Aggregations.STORE_DEPT.value:
            iterable = it.product(STORES_IDS, DEPTS)
        case Aggregations.STORE_CAT.value:
            iterable = it.product(STORES_IDS, CATS)
        case _:
            raise ValueError
    return iterable


def get_agg_cols(agg: str) -> tuple[str, ...]:
    """ID columns for given agg type"""
    match agg:
        case Aggregations.STORE.value:
            return ("store_id",)
        case Aggregations.STORE_DEPT.value:
            return ("store_id", "dept_id")
        case Aggregations.STORE_CAT.value:
            return ("store_id", "cat_id")
        case _:
            raise ValueError


def get_agg_pred_mask(agg: str, test_data: pd.DataFrame, vals: Iterable) -> pd.Series:
    match agg:
        case Aggregations.STORE.value:
            (store,) = vals
            return test_data.store_id == store
        case Aggregations.STORE_DEPT.value:
            (store, dept) = vals
            return (test_data.store_id == store) & (test_data.dept_id == dept)
        case Aggregations.STORE_CAT.value:
            (store, cat) = vals
            return (test_data.store_id == store) & (test_data.cat_id == cat)
        case _:
            raise ValueError


def prepare_nonrecur(
    grid_df, cv: str, model_var: list[str], return_masks: bool = False
):
    tr_mask = (grid_df["d"] <= VALIDATION[cv][0]) & (grid_df["d"] >= FIRST_DAY)
    vl_mask = (grid_df["d"] > VALIDATION[cv][0]) & (grid_df["d"] <= VALIDATION[cv][1])

    train_data = lgb.Dataset(
        grid_df[tr_mask][model_var], label=grid_df[tr_mask]["sales"]
    )

    valid_data = lgb.Dataset(
        grid_df[vl_mask][model_var], label=grid_df[vl_mask]["sales"]
    )
    if return_masks:
        return train_data, valid_data, tr_mask, vl_mask
    return train_data, valid_data


# --------------------- Main non-recursive --------------------- #


def main_train_nonrecursive(agg: str | Aggregations):
    """
    Non-recursive model training M5 competition.
    """
    if isinstance(agg, Aggregations):
        agg = agg.value
    rmsse_bycv = dict()
    data = ProcData(agg)
    iterable = get_iterable(agg)
    lgb_params = LGB_PARAMS[agg]
    for cv in cvs:
        print("cv : day", VALIDATION[cv])
        pred_list = []
        for vals in iterable:
            LOGGER.info(f"Training store with {agg=} and {vals=}")
            grid_df = data.query_fn(*vals)
            model_var = grid_df.columns[~grid_df.columns.isin(REMOVE_FEATURE)].tolist()
            LOGGER.debug(f"Columns for non-recursive model {model_var=}")
            train_data, valid_data = prepare_nonrecur(  # type: ignore
                grid_df, cv, model_var, return_masks=False
            )
            m_lgb = lgb.train(
                lgb_params,
                train_data,
                valid_sets=[valid_data, train_data],
                callbacks=[lgb.log_evaluation(period=100)],
            )
            model_file = fintopmet.fp.MODELS / get_save_name(agg, vals)
            with open(model_file, "wb") as file:
                pickle.dump(m_lgb, file)
            LOGGER.info(f"... finished training store with {agg=} and {vals=}")


# --------------------- Predictions --------------------- #


def make_lag(LAG_DAY, base_test):
    lag_df = base_test[["id", "d", TARGET]]
    col_name = "sales_lag_" + str(LAG_DAY)
    lag_df[col_name] = (
        lag_df.groupby(["id"])[TARGET]
        .transform(lambda x: x.shift(LAG_DAY))
        .astype(np.float16)
    )
    return lag_df[[col_name]]


def make_lag_roll_pd(shift_day: int, roll_wind: int, base_test, target: str = TARGET):
    col_name = f"rolling_mean_tmp_{shift_day}_{roll_wind}"
    return (
        base_test[["id", "d", target]]
        .groupby(["id"])[target]
        .transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())
        .to_frame(col_name)
    )


def make_lag_roll(shift_day: int, roll_wind: int, base_test, target: str = TARGET):
    col_name = f"rolling_mean_tmp_{shift_day}_{roll_wind}"
    if not isinstance(base_test, pl.DataFrame):
        base_test = fintopmet.data_utils.pd_to_pl(base_test)
    return (
        base_test.select(["id", "d", target])
        .with_columns(
            pl.col(target)
            .shift(shift_day)
            .rolling_mean(roll_wind)
            .over("id")
            .alias(col_name)
        )
        .select(col_name)
        .to_pandas()
    )


# --------------------- Recursive --------------------- #


def concat_all_base_test(agg: str, verbose: bool = False):
    files = []
    iterable = get_iterable(agg)
    agg_cols = get_agg_cols(agg)
    for vals in iterable:
        file = get_test_file(agg, "recursive", "", *vals)
        if file.exists():
            files.append((file, *vals))
        else:
            if verbose:
                print(f"File {file} does not exist!")
    dfs = []
    for file, *vals in files:
        cols = {col: val for col, val in zip(agg_cols, vals)}
        dfs.append(pd.read_parquet(file).assign(**cols))  # Assign id cols
    return pd.concat(dfs).reset_index(drop=True), files


def validate_categoricals(df):
    # Have to be the dataframe that we pass to the model object
    # and not just numpy array; uses the categorical info in the
    # dataframe
    id_cols = ["id", "dept_id", "cat_id"]
    if not (df[id_cols].dtypes == "category").all():  # Has to be of category dtype
        print("Id cols not category dtype")
        df = df.astype({col: "category" for col in id_cols})
    return df


def lag_roll_df(all_test: pd.DataFrame):
    lagrolls = pd.concat(
        (
            make_lag_roll(shift_day, roll_wind, all_test, target="sales")
            for shift_day, roll_wind in tqdm(ROLS_SPLIT, "lagrolling df...")
        ),
        axis=1,
    )
    if (intersect := lagrolls.columns.intersection(all_test.columns)).shape[0] > 0:
        print("Dropping columns")
        all_test.drop(intersect, axis=1, inplace=True)  # Reassign columns
    all_test = all_test.join(lagrolls)
    return all_test


def forecast_horizon_recursive(all_test, agg: str, h: int, files: list[Iterable]):
    """Compute forecast horizon over specified test files.

    Should be generalized for store, store-cat, store-dept.
    """
    all_test = all_test.copy()
    for predict_day in range(1, h + 1):
        # Transform values including the forecasted values from t -1
        predict_data = lag_roll_df(all_test.copy())  # Copy while we reuse
        for _, *vals in files:
            day_mask = all_test["d"] == (END_TRAIN + predict_day)
            id_mask = get_agg_pred_mask(agg, predict_data, vals)
            mask = day_mask & id_mask  # Predict for given id & day
            print(f"Predicting for {agg=}, {vals=}, {predict_day=}")
            model = load_model(agg, "recursive", True, *vals)
            # Assigning predictions for t = T + h
            all_test.loc[mask, "sales"] = model.predict(
                predict_data.loc[mask, model.feature_name()]
            )
    return all_test


def make_recursive_submission(forecasts: pd.DataFrame, agg: str):
    forecasts_wide = (
        forecasts.query("d >= 1942")[["id", "d", "sales"]]
        .assign(d=lambda df: df.d.map({1941 + i: f"F{i}" for i in range(1, 29)}))
        .pivot(index="id", columns="d", values="sales")
        .pipe(data_utils.strip_multiindex)
        .pipe(sort_forecast_cols)
    )
    sample_submission = pd.read_csv(fintopmet.fp.M5 / "sample_submission_accuracy.csv")
    submissions = sample_submission.set_index("id").filter([]).join(forecasts_wide)
    submissionfile = (
        fintopmet.fp.M5 / f"submission_{'_'.join(get_agg_cols(agg))}.parquet"
    )
    submissions.to_parquet(submissionfile)
    return submissions


def remap_forecasts(
    df: pd.DataFrame, st: int = 1941, horizon: int = 28
) -> pd.DataFrame:
    return df.assign(
        d=lambda df: df.d.map({st + i: f"F{i}" for i in range(1, horizon + 1)})
    )


def sort_forecast_cols(
    df: pd.DataFrame, st: int = 1941, horizon: int = 28
) -> pd.DataFrame:
    return df[[f"F{i}" for i in range(1, horizon + 1)]]


def forecast_horizon_nonrecursive(
    data: ProcData, agg: str | Aggregations, save_individual: bool = False
):
    """
    Non-recursive model training M5 competition.
    """
    if isinstance(agg, Aggregations):
        agg = agg.value
    iterable = get_iterable(agg)
    pred_list = []
    agg_cols_joined = "_".join(get_agg_cols(agg))
    for vals in iterable:
        LOGGER.info(f"Forecasting for {agg=} and {vals=}")
        grid_df = data.query_fn(*vals)
        model_var = grid_df.columns[~grid_df.columns.isin(REMOVE_FEATURE)].tolist()
        LOGGER.debug(f"Columns for non-recursive model {model_var=}")
        _, _, _, vl_mask = prepare_nonrecur(grid_df, cv="private", model_var=model_var, return_masks=True)  # type: ignore
        model = load_nonrec_model(agg, "", *vals)

        predictions = (
            grid_df.loc[vl_mask, ["id", "d"]]
            .copy()
            .assign(y_pred=model.predict(grid_df.loc[vl_mask, model.feature_name()]))
            .pipe(remap_forecasts, st=1941, horizon=28)
            .pivot(index="id", columns="d", values="y_pred")
            .pipe(data_utils.strip_multiindex)
            .pipe(sort_forecast_cols)
        )
        if save_individual:
            pred_data_file = fintopmet.fp.PREDS / get_save_name(
                agg,
                vals=vals,
                ftype=f"preds_nonrec_{agg_cols_joined}",
                file_suffix=".parquet",
            )
            predictions.to_parquet(pred_data_file)
        pred_list.append(predictions)
    filename = fintopmet.fp.PREDS / f"preds_nonrec_{agg_cols_joined}.parquet"
    all = pd.concat(pred_list)
    all.to_parquet(filename)
    return all
