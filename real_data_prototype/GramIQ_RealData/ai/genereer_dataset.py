"""
Genereert trainings-, validatie- en testdata voor de AI die voorspelt wanneer
een bak leeg is, op basis van ECHTE verkopen.

Werkwijze
---------
1. Alle echte verkopen van de 4 producten (juli 2020 - juni 2023, ai/echte_data.py)
   worden klanten aan de bak: elke klant neemt stuk per stuk exact het gewicht
   dat hij aan de kassa betaalde, een paar minuten vóór het afrekenen.
2. Het realistische weegschaalmodel (weegschaal_model.py) zet dat om in
   meetwaarden zoals een echte load cell ze zou sturen (ruis, pieken van een
   hand of iemand die leunt, personeel dat bijvult, ...). Drie jaar aan één stuk.
3. Elke 5 minuten nemen we een 'foto': de features zoals de backend ze live
   berekent + het echte antwoord (label):
       label = aantal open minuten tot de bak leeg is als niemand bijvult.
   Dat kunnen we exact berekenen omdat we de echte toekomstige verkopen kennen.
4. Periodes: train 07/2020-06/2022, validatie 07-12/2022, test 01-06/2023.
   De AI ziet de test-maanden nooit tijdens het trainen.

Uitvoer: data/samples_echt.csv.gz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from echte_data import TRAIN_TOT as _TRAIN_TOT, VAL_TOT as _VAL_TOT, verkopen_per_bak  # noqa: E402
from features import (FEATURE_NAMEN, MAX_HORIZON_MIN, OPEN_MIN_PER_DAG, BakHistoriek,  # noqa: E402
                      laad_vraagprofiel, open_minuten_tussen, standaard_profielpad)
from weegschaal_model import (OPENING_UUR, PRODUCT_CODE, PRODUCTEN, kassa_naar_klanten, meet,  # noqa: E402
                              meet_tijdstippen, naar_datetime, simuleer_bak)

HIER = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HIER, "..", "data")
MEET_INTERVAL_S = 10
SAMPLE_ELKE_MIN = 5
TRAIN_TOT = datetime.fromisoformat(_TRAIN_TOT).date()
VAL_TOT = datetime.fromisoformat(_VAL_TOT).date()


def bouw_vraagprofiel(verkopen: pd.DataFrame, tot) -> dict:
    """Gemiddelde verkoop in gram per open minuut, per bak en weekdag.

    Enkel uit de trainingsperiode (< tot), enkel dagen waarop het product verkocht
    werd. Klanten staan gemiddeld ~8 min vóór het afrekenen aan de bak, dus het
    profiel wordt 8 min vroeger gelegd. Per 10 minuten opgeteld en licht afgevlakt.
    """
    v = verkopen[verkopen.ts < pd.Timestamp(tot)].copy()
    v["minuut"] = ((v.ts.dt.hour - OPENING_UUR) * 60 + v.ts.dt.minute - 8).clip(0, OPEN_MIN_PER_DAG - 1)
    v["blok"] = v.minuut // 10
    v["wd"] = v.ts.dt.weekday
    v["dag"] = v.ts.dt.normalize()
    profiel = {}
    for bak, d in v.groupby("bak_id"):
        n_dagen = d.groupby("wd").dag.nunique()
        som = d.groupby(["wd", "blok"]).gram.sum().unstack(fill_value=0.0)
        som = som.reindex(index=range(7), columns=range(OPEN_MIN_PER_DAG // 10), fill_value=0.0)
        per_min = som.div(n_dagen.reindex(range(7)).fillna(1).clip(lower=1), axis=0) / 10
        per_min = per_min.T.rolling(3, center=True, min_periods=1).mean().T   # afvlakken over 30 min
        profiel[bak] = {
            "naam": PRODUCTEN[bak]["naam"],
            "bron": f"echte verkopen tot {pd.Timestamp(tot):%Y-%m-%d}",
            "gram_per_min": np.repeat(per_min.values, 10, axis=1).round(2).tolist(),
        }
    return profiel


def maak_samples(bak, sim, start, eind, rng, verkoopdagen, profiel=None):
    """Laat metingen door de live feature-pipeline lopen en voeg labels toe."""
    t = meet_tijdstippen(start, eind, MEET_INTERVAL_S)
    tt, metingen = meet(sim, t, rng)
    stuk = PRODUCTEN[bak]["stuk_gewicht"]

    cum_vraag = np.concatenate([[0.0], np.cumsum(sim.vraag_gram)])
    hist = BakHistoriek(bak, product=PRODUCTEN[bak], product_code=PRODUCT_CODE[bak], profiel=profiel)
    rijen = []
    volgende_sample = None
    for ti, g in zip(tt, metingen):
        hist.voeg_toe(float(ti), float(g))
        minuut = int(ti // 60)
        if minuut % SAMPLE_ELKE_MIN != 0 or minuut == volgende_sample:
            continue
        volgende_sample = minuut
        if not BakHistoriek._is_open(ti):
            continue
        dag = naar_datetime(ti).date()
        if dag not in verkoopdagen:
            continue  # product die dag niet in de verkoop (geen enkele transactie)
        f = hist.features()
        if f is None:
            continue
        # ---- label: wanneer is de bak leeg als niemand bijvult? ----
        echt = float(sim.echt_gewicht(np.array([ti]))[0])
        nodig = echt - 0.4 * stuk
        if nodig <= 0:
            label = 0.0
        else:
            i0 = np.searchsorted(sim.vraag_t, ti, side="right")
            doel = cum_vraag[i0] + nodig
            j = np.searchsorted(cum_vraag, doel, side="left")  # cum_vraag[j] >= doel
            if j > len(sim.vraag_t):
                continue  # einde van de simulatie: label onbekend
            t_leeg = sim.vraag_t[j - 1]
            label = open_minuten_tussen(ti, t_leeg)
        f["label_min"] = min(label, MAX_HORIZON_MIN)
        f["echt_gewicht"] = echt
        f["tijd"] = pd.Timestamp(datetime(1970, 1, 1) + timedelta(seconds=float(ti)))
        f["bak_id"] = bak
        f["periode"] = "train" if dag < TRAIN_TOT else ("val" if dag < VAL_TOT else "test")
        rijen.append(f)
    return rijen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bakken", default=",".join(PRODUCTEN), help="bv. 01,03")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    verkopen = verkopen_per_bak(PRODUCTEN)
    start = verkopen.ts.min().normalize().to_pydatetime()
    eind = verkopen.ts.max().normalize().to_pydatetime() + timedelta(days=1)
    print(f"Echte verkopen: {len(verkopen)} transacties van {start:%d-%m-%Y} tot {eind:%d-%m-%Y}")

    # vraagprofiel uit de trainingsperiode -> ook gebruikt door de live backend
    import json
    profiel_json = bouw_vraagprofiel(verkopen, _TRAIN_TOT)
    with open(standaard_profielpad(), "w") as f:
        json.dump(profiel_json, f)
    profielen = laad_vraagprofiel()
    print(f"Vraagprofiel opgeslagen: {standaard_profielpad()}")

    alle = []
    for bak in args.bakken.split(","):
        t0 = time.time()
        p = PRODUCTEN[bak]
        d = verkopen[verkopen.bak_id == bak]
        klanten = kassa_naar_klanten(d.ts, d.gram, p["stuk_gewicht"], rng)
        sim = simuleer_bak(bak, klanten, start, eind + timedelta(days=1), rng)
        verkoopdagen = set(d.ts.dt.date)
        rijen = maak_samples(bak, sim, start, eind, rng, verkoopdagen, profielen.get(bak))
        alle += rijen
        print(f"Bak {bak} ({p['naam']}): {len(klanten)} klanten, {len(verkoopdagen)} verkoopdagen, "
              f"{len(sim.bijvullingen)} keer bijgevuld, {len(rijen)} samples  [{time.time()-t0:.0f}s]")
    df = pd.DataFrame(alle)
    uit = os.path.join(DATA, "samples_echt.csv.gz")
    df.to_csv(uit, index=False)
    print(df.groupby(["periode", "bak_id"]).size().unstack())
    print(f"Opgeslagen: {uit}  ({len(df)} rijen)")
    print("Kolommen:", FEATURE_NAMEN + ["label_min"])


if __name__ == "__main__":
    main()
