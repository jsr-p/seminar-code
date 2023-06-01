import fintopmet
from fintopmet.models import acc

if __name__ == "__main__":
    tot_models = (
        len(list(acc.get_iterable("STORE")))
        + len(list(acc.get_iterable("STORE_CAT")))
        + len(list(acc.get_iterable("STORE_DEPT")))
    )

    models_rec = set(
        [
            f.name
            for f in fintopmet.fp.MODELS.glob("*")
            if f.name.startswith("lgb-model")
        ]
    )
    models_nonrec = set(
        [f.name for f in fintopmet.fp.MODELS.glob("*") if f.name.startswith("model")]
    )
    assert len(models_rec) == tot_models
    assert len(models_nonrec) == tot_models
    print("All model estimated")
