from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
DATA = PROJ / "data"
M5 = DATA / "m5"
PROC = M5 / "proc"
PREDS = M5 / "preds"
LOG = PROJ / "logs"
FIGS = PROJ / "output" / "figs"
MODELS = PROJ / "models"

for fp in [MODELS, PREDS]:
    Path.mkdir(fp, exist_ok=True)
