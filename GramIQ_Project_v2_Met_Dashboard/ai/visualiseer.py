"""
Maakt een grafiek voor het verslag/de presentatie:
  boven : gemeten gewicht van een bak op één dag (realistische simulator)
  onder : AI-voorspelling 'leeg over X min' vs. de echte waarde en de baseline

Gebruik:  python visualiseer.py [--bak 07] [--dag 2026-10-24]
Uitvoer:  ai/voorbeeld_voorspelling.png
"""
import argparse
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HIER)
from features import pas_marges_toe  # noqa: E402
from weegschaal_model import PRODUCTEN  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bak", default="07")
    ap.add_argument("--dag", default="2026-10-24")
    a = ap.parse_args()

    model = joblib.load(os.path.join(HIER, "model_leeg.joblib"))
    df = pd.read_csv(os.path.join(HIER, "..", "data", "test_kassa.csv"), parse_dates=["tijd"])
    df["bak_id"] = df.bak_id.astype(str).str.zfill(2)
    d = df[(df.bak_id == a.bak) & (df.tijd.dt.strftime("%Y-%m-%d") == a.dag)].copy()
    if d.empty:
        sys.exit("Geen data voor die bak/dag (zondag?).")
    d["pred"] = np.clip(model["hoofd"].predict(d[model["features"]]), 0, model["max_horizon_min"])
    d["lo"], d["hi"] = pas_marges_toe(d["pred"].values, model["marges"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax1.step(d.tijd, d.gewicht / 1000, where="post", color="#2980b9")
    ax1.set_ylabel("gewicht (kg)")
    ax1.set_title(f"Bak {a.bak} ({PRODUCTEN[a.bak]['naam']}) op {a.dag}")
    ax1.grid(alpha=0.3)

    ax2.fill_between(d.tijd, d.lo, d.hi, color="#e67e22", alpha=0.2, label="AI foutmarge (80%)")
    ax2.plot(d.tijd, d.label_min, color="black", lw=2, label="echt: leeg over (min)")
    ax2.plot(d.tijd, d.pred, color="#e67e22", lw=2, label="AI-voorspelling")
    ax2.plot(d.tijd, d.baseline_min.clip(upper=d.label_min.max() * 1.5), color="#95a5a6", ls="--",
             label="baseline (gewicht ÷ verbruik)")
    ax2.set_ylabel("minuten tot leeg\n(als niemand bijvult)")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    uit = os.path.join(HIER, "voorbeeld_voorspelling.png")
    fig.savefig(uit, dpi=110)
    print("Opgeslagen:", uit)


if __name__ == "__main__":
    main()
