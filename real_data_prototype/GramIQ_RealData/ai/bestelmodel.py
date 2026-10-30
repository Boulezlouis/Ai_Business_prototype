"""
Bestelvoorspelling (#20): hoeveel kg moet er vandaag besteld worden?

Situatie: de groothandel levert 's ochtends vroeg (de inkoop gebeurt rond 3-4u).
Op dat moment kennen we alle verkopen tot en met gisteren en de inkoopprijs
van vandaag. We voorspellen de verkoop van vandaag (in kg) per product.

Model: Gradient Boosting (HistGradientBoostingRegressor), één model voor alle
57 producten met minstens 300 verkoopdagen (meer data = beter patroon).
  * mediaan       -> verwachte verkoop
  * 80%-kwantiel  -> veilige hoeveelheid: op ~80% van de dagen genoeg
Bestelling = 80%-kwantiel / (1 - verliespercentage)   (bederf/afval, annex4)

Evaluatie op de testperiode (jan-jun 2023, nooit gezien tijdens training),
vergeleken met twee eenvoudige regels:
  * "zelfde weekdag vorige week"
  * "gemiddelde van de laatste 7 dagen"

Gebruik:  python bestelmodel.py
Uitvoer:  ai/model_bestelling.joblib + ai/bestel_rapport.json
"""
from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HIER)
from echte_data import (TRAIN_TOT, VAL_TOT, dagverkopen, groothandelsprijzen,  # noqa: E402
                        productinfo)
from weegschaal_model import PRODUCTEN  # noqa: E402

MIN_VERKOOPDAGEN = 300
KWANTIEL = 0.8

BESTEL_FEATURES = [
    "categorie_code", "weekdag", "maand", "dag_van_maand",
    "lag1", "lag2", "lag7", "gem7", "gem14", "gem28", "std7", "zelfde_weekdag_gem4",
    "aangeboden_7", "dagen_sinds_verkoop", "korting_gem7", "verkoopprijs_lag1",
    "inkoopprijs", "inkoopprijs_rel28",
]


def bouw_features() -> pd.DataFrame:
    """Eén rij per product per kalenderdag; features gebruiken enkel info tot gisteren
    (+ de inkoopprijs van vandaag, die 's ochtends gekend is)."""
    dv = dagverkopen()
    info = productinfo()
    n_dagen = dv.groupby("item_code").datum.nunique()
    codes = n_dagen[n_dagen >= MIN_VERKOOPDAGEN].index
    for p in PRODUCTEN.values():          # de bakken altijd meenemen
        codes = codes.union([p["item_code"]])
    dv = dv[dv.item_code.isin(codes)]

    kalender = pd.date_range(dv.datum.min(), dv.datum.max(), freq="D")
    idx = pd.MultiIndex.from_product([codes, kalender], names=["item_code", "datum"])
    df = dv.set_index(["item_code", "datum"]).reindex(idx).reset_index()
    df["aangeboden"] = df.kg.notna() & (df.kg > 0)
    df = df.merge(groothandelsprijzen(), on=["datum", "item_code"], how="left")
    cat = info.set_index("item_code").categorie
    df["categorie_code"] = df.item_code.map(cat).astype("category").cat.codes

    g = df.groupby("item_code", group_keys=False)
    kg_gisteren = g.kg.shift(1)                            # NaN = niet verkocht/niet aangeboden
    df["lag1"] = kg_gisteren
    df["lag2"] = g.kg.shift(2)
    df["lag7"] = g.kg.shift(7)
    gk = kg_gisteren.groupby(df.item_code)
    df["gem7"] = gk.transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["gem14"] = gk.transform(lambda s: s.rolling(14, min_periods=1).mean())
    df["gem28"] = gk.transform(lambda s: s.rolling(28, min_periods=3).mean())
    df["std7"] = gk.transform(lambda s: s.rolling(7, min_periods=2).std())
    df["zelfde_weekdag_gem4"] = (df.groupby(["item_code", df.datum.dt.dayofweek]).kg
                                 .transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean()))
    df["aangeboden_7"] = (df.aangeboden.astype(float).groupby(df.item_code)
                          .transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum()))
    laatst = df.datum.where(df.aangeboden).groupby(df.item_code).transform(lambda s: s.shift(1).ffill())
    df["dagen_sinds_verkoop"] = (df.datum - laatst).dt.days
    df["korting_gem7"] = (df.korting_aandeel.groupby(df.item_code)
                          .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean()))
    df["verkoopprijs_lag1"] = g.verkoopprijs.shift(1)
    inkoop28 = df.inkoopprijs.groupby(df.item_code).transform(lambda s: s.shift(1).rolling(28, min_periods=3).mean())
    df["inkoopprijs_rel28"] = df.inkoopprijs / inkoop28
    df["weekdag"] = df.datum.dt.dayofweek
    df["maand"] = df.datum.dt.month
    df["dag_van_maand"] = df.datum.dt.day
    df = df.merge(info[["item_code", "verlies_pct"]], on="item_code", how="left")
    df["verlies_pct"] = df.verlies_pct.fillna(df.verlies_pct.median())
    return df


def maak_model(loss, **kw):
    return HistGradientBoostingRegressor(
        loss=loss, max_iter=500, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, categorical_features=[BESTEL_FEATURES.index("categorie_code")],
        random_state=0, **kw)


def bestel_kg(q80, verlies_pct):
    return np.asarray(q80) / (1 - np.asarray(verlies_pct) / 100)


def evalueer(df, med, q80):
    y = df.kg.values
    zw = df.lag7.fillna(df.gem7).fillna(0).values
    g7 = df.gem7.fillna(0).values
    bestel = bestel_kg(q80, df.verlies_pct.values)
    return {
        "dagen": int(len(df)),
        "MAE_model_kg": float(np.mean(np.abs(med - y))),
        "MAE_vorige_week_kg": float(np.mean(np.abs(zw - y))),
        "MAE_gem7_kg": float(np.mean(np.abs(g7 - y))),
        "gem_verkoop_kg": float(np.mean(y)),
        # na aftrek van het verlies blijft q80 over: volstaat dat voor de echte verkoop?
        "genoeg_met_bestelling": float(np.mean(q80 >= y)),
        "gem_bestelling_kg": float(np.mean(bestel)),
    }


def main():
    print("Features bouwen ...")
    df = bouw_features()
    # doel: dagen waarop het product verkocht werd en er al wat geschiedenis is
    doel = df[df.aangeboden & df.gem28.notna()].copy()
    train = doel[doel.datum < TRAIN_TOT]
    val = doel[(doel.datum >= TRAIN_TOT) & (doel.datum < VAL_TOT)]
    test = doel[doel.datum >= VAL_TOT]
    print(f"{doel.item_code.nunique()} producten · train {len(train)} · val {len(val)} · test {len(test)} productdagen")

    rapport = {"features": BESTEL_FEATURES, "kwantiel": KWANTIEL}
    med = maak_model("absolute_error").fit(train[BESTEL_FEATURES], train.kg)
    q = maak_model("quantile", quantile=KWANTIEL).fit(train[BESTEL_FEATURES], train.kg)
    bak_codes = {p["item_code"]: (b, p["naam"]) for b, p in PRODUCTEN.items()}
    for naam, d in (("validatie (jul-dec 2022)", val), ("test (jan-jun 2023)", test)):
        pm = np.clip(med.predict(d[BESTEL_FEATURES]), 0, None)
        pq = np.maximum(np.clip(q.predict(d[BESTEL_FEATURES]), 0, None), pm)
        r = {"alle_producten": evalueer(d, pm, pq)}
        for code, (bak, pnaam) in bak_codes.items():
            m = (d.item_code == code).values
            r[f"bak {bak} {pnaam}"] = evalueer(d[m], pm[m], pq[m])
        rapport[naam] = r
        print(f"\n== {naam} ==")
        for k, v in r.items():
            print(f"  {k:28s} gem. fout model {v['MAE_model_kg']:5.2f} kg | vorige week {v['MAE_vorige_week_kg']:5.2f}"
                  f" | gem. 7 dagen {v['MAE_gem7_kg']:5.2f} | gem. verkoop {v['gem_verkoop_kg']:5.1f} kg"
                  f" | bestelling volstaat {100*v['genoeg_met_bestelling']:3.0f}%")

    print("\nFinaal model trainen op train + validatie ...")
    tv = doel[doel.datum < VAL_TOT]
    pakket = {
        "features": BESTEL_FEATURES,
        "mediaan": maak_model("absolute_error").fit(tv[BESTEL_FEATURES], tv.kg),
        "kwantiel": maak_model("quantile", quantile=KWANTIEL).fit(tv[BESTEL_FEATURES], tv.kg),
        "kwantiel_niveau": KWANTIEL,
    }
    joblib.dump(pakket, os.path.join(HIER, "model_bestelling.joblib"), compress=3)
    with open(os.path.join(HIER, "bestel_rapport.json"), "w") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)
    print("Opgeslagen: ai/model_bestelling.joblib en ai/bestel_rapport.json")


if __name__ == "__main__":
    main()
