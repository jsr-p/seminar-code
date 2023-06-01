import seaborn as sns

import fintopmet
from fintopmet import data_utils

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)


def main():
    validation = data_utils.load_validation()
    calendar = data_utils.load_calendar()
    date_map = dict(zip(calendar.d, calendar.date))
    day_cols = data_utils.day_cols(validation)

    # States
    states = (
        validation.groupby("state_id")[day_cols]
        .sum()
        .transpose()
        .assign(Total=lambda df: df.sum(axis=1))
    )
    states.index = states.index.map(date_map)
    ax = states.plot(xlabel="Date", ylabel="Unit sales", alpha=0.9)
    ax.legend()
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(str(fintopmet.fp.FIGS / "aggregated_sales_states.png"))

    # Cats
    day_cols = data_utils.day_cols(validation)
    categories = validation.groupby("cat_id")[day_cols].sum().transpose()
    categories.index = categories.index.map(date_map)
    ax = categories.plot(xlabel="Date", ylabel="Unit sales", alpha=0.9)
    ax.legend()
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(str(fintopmet.fp.FIGS / "aggregated_sales_cats.png"))

    df = states.join(categories).reset_index(names="date")
    df.to_csv(fintopmet.fp.DATA / "agg.csv", index=False)


if __name__ == "__main__":
    main()
