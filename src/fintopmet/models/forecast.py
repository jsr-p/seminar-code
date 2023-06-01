import numpy as np
import pandas as pd
import seaborn as sns

import fintopmet
from fintopmet.data import data_utils, transforms
from fintopmet.data.data_utils import Data
from fintopmet.models import acc

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)


if __name__ == "__main__":
    rec_subs = dict()
    for agg in ["STORE", "STORE_CAT", "STORE_DEPT"]:
        test_data, files = acc.concat_all_base_test(agg)
        test_data = acc.validate_categoricals(test_data)
        forecasts = acc.forecast_horizon_recursive(
            test_data, agg=agg, h=28, files=files
        )
        rec_subs[agg] = acc.make_recursive_submission(forecasts, agg=agg)

    ## Non-recursive

    nonrec_subs = dict()
    for agg in ["STORE", "STORE_CAT", "STORE_DEPT"]:
        data = acc.ProcData(agg)
        nonrec_subs[agg] = acc.forecast_horizon_nonrecursive(data, agg)

    nonrec_subs["STORE"]

    sample_submission = pd.read_csv(fintopmet.fp.M5 / "sample_submission_accuracy.csv")
    sample_submission_evaluation = sample_submission.iloc[
        30490:
    ]  # First rows are for validation

    def _melt(df):
        return df.reset_index(names="id").melt(
            id_vars="id", var_name="h", value_name="forecast_value"
        )

    final = (
        pd.concat(
            (
                pd.concat(_melt(nonrec_subs[k]) for k in nonrec_subs),
                pd.concat(_melt(rec_subs[k]) for k in nonrec_subs),
            )
        )
        .groupby(["id", "h"])
        .forecast_value.mean()
        .unstack(1)
        .pipe(acc.sort_forecast_cols)
        .pipe(fintopmet.data_utils.strip_multiindex)
    )

    # Sort indices to the correct order, assign id col and export
    # final.loc[sample_submission_evaluation.id].reset_index(names="id").to_csv(fintopmet.fp.M5 / "submission_final.csv", index=False)
    sample_submission.set_index("id").filter([]).join(final).reset_index().fillna(
        0).to_csv(fintopmet.fp.M5 / "submission_final.csv", index=False)
