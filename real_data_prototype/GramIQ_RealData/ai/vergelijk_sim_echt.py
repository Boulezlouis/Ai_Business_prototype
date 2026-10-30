"""
Experiment: is gesimuleerde vraag goed genoeg om de AI te trainen?

Vergelijkt twee modellen voor 'wanneer is de bak leeg', met exact dezelfde
weegschaalsimulatie, features en modelinstellingen:

  A. getraind op ECHTE verkopen (jul 2020 - dec 2022)        = het huidige model
  B. getraind op GESIMULEERDE vraag zoals in prototype v2:
     klanten per uur via een Poisson-proces volgens het gemiddelde
     uurprofiel (per weekdag), met een willekeurige dag- en uurfactor,
     hoeveelheid per klant getrokken uit de echte verdeling.
     Het profiel en de hoeveelheden komen uit dezelfde trainingsperiode.

Beide worden getest op:
  * de echte testperiode (jan - jun 2023)              -> wat telt in de praktijk
  * een gesimuleerde testperiode (ongeziene sim-dagen) -> wat v2 rapporteerde

Gebruik:  python vergelijk_sim_echt.py        (±10 min)
Uitvoer:  ai/vergelijking_sim_echt.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HIER)
from echte_data import TRAIN_TOT, VAL_TOT, verkopen_per_bak  # noqa: E402
from features import FEATURE_NAMEN, MAX_HORIZON_MIN, laad_vraagprofiel  # noqa: E402
from genereer_dataset import maak_samples  # noqa: E402
from train_model import maak_model  # noqa: E402
from weegschaal_model import (OPENING_UUR, PRODUCTEN, SLUITING_UUR, Klant,  # noqa: E402
                              _ts, simuleer_bak)

DATA = os.path.join(HIER, "..", "data")
SIM_DAGEN = 365


def synthetische_klanten_v2(verkopen_train: pd.DataFrame, stuk: float, dagen, rng) -> list[Klant]:
    """Zoals ai/genereer_dataset.py in v2: Poisson per uur volgens het gemiddelde profiel."""
    v = verkopen_train
    n_dagen = v.groupby(v.ts.dt.weekday).ts.apply(lambda s: s.dt.normalize().nunique())
    per_uur = v.groupby([v.ts.dt.weekday, v.ts.dt.hour]).size().unstack(fill_value=0)
    per_uur = per_uur.div(n_dagen, axis=0)
    grammen = v.gram.values
    klanten = []
    for dag in dagen:
        dagfactor = rng.lognormal(0, 0.2)
        for uur in range(OPENING_UUR, SLUITING_UUR):
            rate = float(per_uur.loc[dag.weekday()].get(uur, 0.0))
            n = rng.poisson(rate * dagfactor * rng.lognormal(0, 0.15))
            t0 = _ts(dag.replace(hour=uur))
            for t in np.sort(rng.uniform(t0, t0 + 3600, n)):
                g = float(rng.choice(grammen))
                klanten.append(Klant(tijd=float(t), aantal=max(1, int(round(g / stuk))), gram=g))
    return klanten


def mae(p, y, m=None):
    m = np.ones(len(y), bool) if m is None else m
    return float(np.mean(np.abs(p[m] - y[m])))


def scores(model, df):
    p = np.clip(model.predict(df[FEATURE_NAMEN]), 0, MAX_HORIZON_MIN)
    y = df.label_min.values
    laag = (df.pct_vol.values > 3) & (df.pct_vol.values < 25)
    # kalibratie: AI zegt 30-60 min -> wat is de echte mediaan?
    k = (p >= 30) & (p < 60) & (df.pct_vol.values > 3)
    return {"MAE_min": mae(p, y), "MAE_bak_laag_min": mae(p, y, laag),
            "MAE_baseline_min": mae(df.baseline_min.values, y),
            "MAE_baseline_bak_laag_min": mae(df.baseline_min.values, y, laag),
            "zegt_30_60_echt_mediaan": float(np.median(y[k])) if k.sum() else None,
            "bias_min": float(np.mean(p - y)), "rijen": int(len(df))}


def main():
    rng = np.random.default_rng(123)
    profielen = laad_vraagprofiel()
    verkopen = verkopen_per_bak(PRODUCTEN)
    echt = pd.read_csv(os.path.join(DATA, "samples_echt.csv.gz"), parse_dates=["tijd"])
    echt["bak_id"] = echt.bak_id.astype(str).str.zfill(2)

    # ---- gesimuleerde vraag (v2-methode) voor dezelfde 4 bakken ----
    sim_start = datetime(2024, 1, 1)                  # fictieve periode
    dagen = [sim_start + timedelta(days=i) for i in range(SIM_DAGEN)]
    sim_rijen = []
    for bak, p in PRODUCTEN.items():
        t0 = time.time()
        vt = verkopen[(verkopen.bak_id == bak) & (verkopen.ts < pd.Timestamp(VAL_TOT))]
        klanten = synthetische_klanten_v2(vt, p["stuk_gewicht"], dagen, rng)
        eind = dagen[-1] + timedelta(days=1)
        sim = simuleer_bak(bak, klanten, sim_start, eind + timedelta(days=1), rng)
        rijen = maak_samples(bak, sim, sim_start, eind, rng, {d.date() for d in dagen}, profielen.get(bak))
        sim_rijen += rijen
        print(f"Bak {bak}: {len(klanten)} gesimuleerde klanten, {len(rijen)} samples [{time.time()-t0:.0f}s]")
    sim = pd.DataFrame(sim_rijen)
    grens = sim_start + timedelta(days=int(SIM_DAGEN * 0.8))
    sim_train, sim_test = sim[sim.tijd < grens], sim[sim.tijd >= grens]

    echt_train = echt[echt.periode != "test"]
    echt_test = echt[echt.periode == "test"]
    # zelfde hoeveelheid trainingsdata voor een eerlijke vergelijking
    echt_train_klein = echt_train.sample(len(sim_train), random_state=0)

    print("Trainen ...")
    modellen = {
        "A_echt": maak_model().fit(echt_train_klein[FEATURE_NAMEN], echt_train_klein.label_min),
        "B_gesimuleerd": maak_model().fit(sim_train[FEATURE_NAMEN], sim_train.label_min),
    }
    res = {"trainrijen": int(len(sim_train))}
    for naam, m in modellen.items():
        res[naam] = {"test_echt": scores(m, echt_test), "test_gesimuleerd": scores(m, sim_test)}
        print(f"\n{naam}")
        for t, r in res[naam].items():
            print(f"  {t:17s} fout {r['MAE_min']:5.1f} min | bak<25% {r['MAE_bak_laag_min']:5.1f} min "
                  f"(baseline {r['MAE_baseline_bak_laag_min']:5.1f}) | 'zegt 30-60' echt {r['zegt_30_60_echt_mediaan']} "
                  f"| bias {r['bias_min']:+.1f}")
    # verschil in de labels zelf: hoe snel lopen bakken leeg?
    res["label_verdeling"] = {
        "echt_test_mediaan": float(echt_test.label_min.median()),
        "sim_test_mediaan": float(sim_test.label_min.median()),
        "echt_aandeel_label_600": float((echt_test.label_min >= 600).mean()),
        "sim_aandeel_label_600": float((sim_test.label_min >= 600).mean()),
    }
    print(res["label_verdeling"])
    with open(os.path.join(HIER, "vergelijking_sim_echt.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("Opgeslagen: ai/vergelijking_sim_echt.json")


if __name__ == "__main__":
    main()
