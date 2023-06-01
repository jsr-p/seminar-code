import timeit

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import fintopmet
from fintopmet.models import acc

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)


def load_data():
    test_data, _ = acc.concat_all_base_test("STORE_DEPT")
    test_data = acc.validate_categoricals(test_data)
    return test_data


@click.command()
@click.option(
    "--pre-transform",
    is_flag=True,
    default=False,
    help="Transform the data to polars before applying function",
)
def benchmark_lag_roll(pre_transform):
    print("Running benchmarking experiment...")
    test_data = load_data()
    shift_day = 7
    roll_wind = 7
    if pre_transform:
        # Pre-transform the data to polars dataframe
        polars_test_data = fintopmet.data_utils.pd_to_pl(test_data)
        time_fns = {
            "Pandas": lambda: acc.make_lag_roll_pd(
                shift_day, roll_wind, test_data, target="sales"
            ),
            "Polars": lambda: acc.make_lag_roll(
                shift_day, roll_wind, polars_test_data, target="sales"
            ),
        }
    else:
        time_fns = {
            "Pandas": lambda: acc.make_lag_roll_pd(
                shift_day, roll_wind, test_data, target="sales"
            ),
            "Polars": lambda: acc.make_lag_roll(
                shift_day, roll_wind, test_data, target="sales"
            ),
        }
    num_repeats = 50
    times = []
    for name, fn in time_fns.items():
        print(f"Running {name=}, {fn=} for {num_repeats=} and {pre_transform=}")
        timer = timeit.Timer(fn)
        elapsed = timer.repeat(repeat=num_repeats, number=1)
        times.append({"name": name, "elapsed": elapsed})
    if pre_transform:
        suffix = "pretrans"
    else:
        suffix = ""
    df = pd.concat(pd.DataFrame(data) for data in times)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    sns.kdeplot(df.query("name == 'Pandas'"), x="elapsed", ax=ax1)
    sns.kdeplot(df.query("name == 'Polars'"), x="elapsed", ax=ax2)
    ax1.set(title="Pandas", xlabel="Rolling average computation time")
    ax2.set(title="Polars", xlabel="Rolling average computation time")
    fig.tight_layout()
    fig.savefig(str(fintopmet.fp.FIGS / f"benchmark{suffix}"))
    df.to_csv(fintopmet.fp.DATA / f"benchmark{suffix}.csv", index=False)


if __name__ == "__main__":
    benchmark_lag_roll()
