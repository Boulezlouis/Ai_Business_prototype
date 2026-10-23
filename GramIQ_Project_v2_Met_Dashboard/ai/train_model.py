"""
Traint de AI die voorspelt binnen hoeveel (open) minuten een bak leeg is.

Model: Gradient Boosting (scikit-learn HistGradientBoostingRegressor)
  * hoofdmodel  -> beste schatting (mediaan)
  * foutmarge   -> op de validatiedata meten we per voorspellingsbereik hoe ver
    de echte waarde afwijkt (10% en 90% percentiel van werkelijk/voorspeld).
    Zo kan het dashboard een marge tonen ("leeg tussen 14:20 en 14:55").

Evaluatie:
  * validatie  = laatste 20% van de synthetische dagen (niet gebruikt om te trainen)
  * test       = de echte kassaperiode (19-31 okt), volledig ongezien
  * vergeleken met een simpele baseline: gewicht / verbruik van het laatste uur

Uitvoer: ai/model_leeg.joblib en ai/model_rapport.json
"""
from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURE_NAMEN, MAX_HORIZON_MIN, pas_marges_toe  # noqa: E402

HIER = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HIER, "..", "data")


def maak_model(loss="absolute_error"):
    kw = dict(loss=loss, max_iter=400, learning_rate=0.06, max_leaf_nodes=48,
              min_samples_leaf=40, l2_regularization=1.0,
              categorical_features=[FEATURE_NAMEN.index("product_code")], random_state=0)
    return HistGradientBoostingRegressor(**kw)


MARGE_GRENZEN = [0, 10, 20, 40, 80, 160, 10_000]   # bins op de voorspelling (min)


def leer_marges(pred, y):
    """Per bereik van de voorspelling: 10%/90% percentiel van (y+1)/(pred+1)."""
    r = (y + 1) / (pred + 1)
    marges = []
    for a, b in zip(MARGE_GRENZEN[:-1], MARGE_GRENZEN[1:]):
        m = (pred >= a) & (pred < b)
        if m.sum() < 30:
            marges.append((0.5, 1.5))
        else:
            marges.append((float(np.quantile(r[m], 0.1)), float(np.quantile(r[m], 0.9))))
    return {"grenzen": MARGE_GRENZEN, "factoren": marges}


def evalueer(naam, df, pred):
    y = df.label_min.values
    base = df.baseline_min.values
    niet_leeg = df.pct_vol.values > 3        # momenten waarop de bak nog niet leeg is
    kort = niet_leeg & (y <= 60)             # het belangrijkste: bijna leeg
    r = {
        "rijen": int(len(df)),
        "MAE_model_min": float(np.mean(np.abs(pred - y))),
        "MAE_baseline_min": float(np.mean(np.abs(base - y))),
        "MAE_model_niet_leeg": float(np.mean(np.abs(pred[niet_leeg] - y[niet_leeg]))),
        "MAE_baseline_niet_leeg": float(np.mean(np.abs(base[niet_leeg] - y[niet_leeg]))),
        "MAE_model_binnen_60min": float(np.mean(np.abs(pred[kort] - y[kort]))),
        "MAE_baseline_binnen_60min": float(np.mean(np.abs(base[kort] - y[kort]))),
    }
    print(f"\n== {naam} ({len(df)} rijen) ==")
    print(f"  Gem. fout model    : {r['MAE_model_min']:6.1f} min   (baseline {r['MAE_baseline_min']:6.1f} min)")
    print(f"  ...bak niet leeg   : {r['MAE_model_niet_leeg']:6.1f} min   (baseline {r['MAE_baseline_niet_leeg']:6.1f} min)")
    print(f"  ...leeg binnen 1u  : {r['MAE_model_binnen_60min']:6.1f} min   (baseline {r['MAE_baseline_binnen_60min']:6.1f} min)")
    return r


def main():
    train_all = pd.read_csv(os.path.join(DATA, "training_synthetisch.csv"), parse_dates=["tijd"])
    test = pd.read_csv(os.path.join(DATA, "test_kassa.csv"), parse_dates=["tijd"])

    dagen = np.sort(train_all.tijd.dt.normalize().unique())
    grens = dagen[int(len(dagen) * 0.8)]
    train = train_all[train_all.tijd < grens]
    val = train_all[train_all.tijd >= grens]

    X, y = train[FEATURE_NAMEN], train.label_min
    print(f"Trainen op {len(train)} rijen ...")
    hoofd = maak_model().fit(X, y)
    marges = leer_marges(np.clip(hoofd.predict(val[FEATURE_NAMEN]), 0, MAX_HORIZON_MIN), val.label_min.values)

    rapport = {"features": FEATURE_NAMEN, "marges": marges}
    for naam, df in (("Validatie (synthetisch, ongezien)", val), ("Test (echte kassaperiode)", test)):
        p = np.clip(hoofd.predict(df[FEATURE_NAMEN]), 0, MAX_HORIZON_MIN)
        r = evalueer(naam, df, p)
        lo, hi = pas_marges_toe(p, marges)
        r["dekking_marge"] = float(np.mean((df.label_min >= lo) & (df.label_min <= hi)))
        print(f"  Echte waarde binnen de foutmarge: {100*r['dekking_marge']:.0f}% (doel ~80%)")
        rapport[naam] = r

    # finaal model op alle synthetische data
    print("\nFinaal model trainen op alle synthetische data ...")
    X, y = train_all[FEATURE_NAMEN], train_all.label_min
    pakket = {
        "features": FEATURE_NAMEN,
        "hoofd": maak_model().fit(X, y),
        "marges": marges,
        "max_horizon_min": MAX_HORIZON_MIN,
    }
    joblib.dump(pakket, os.path.join(HIER, "model_leeg.joblib"), compress=3)
    with open(os.path.join(HIER, "model_rapport.json"), "w") as f:
        json.dump(rapport, f, indent=2)
    print("Opgeslagen: ai/model_leeg.joblib en ai/model_rapport.json")


if __name__ == "__main__":
    main()
