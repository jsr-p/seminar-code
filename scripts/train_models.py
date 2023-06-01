import click

from fintopmet.models import acc


@click.command()
@click.argument("model", type=click.Choice(["recursive", "non-recursive"]))
@click.option(
    "--agg",
    type=click.Choice(
        choices=[v.value for v in acc.Aggregations], case_sensitive=False
    ),
    required=True,
    help="agg number to train model for",
)
def train(model: str, agg: str):
    print(f"Training model for {agg=}")
    match model:
        case "recursive":
            acc.main_train(agg.upper())
        case "non-recursive":
            acc.main_train_nonrecursive(agg.upper())


if __name__ == "__main__":
    train()
