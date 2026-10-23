"""
Genereert trainingsdata voor de AI die voorspelt wanneer een bak leeg is.

Werkwijze
---------
1. Uit de kassadata (data/kassa_data_groot.csv) leren we het vraagpatroon:
   transacties per uur per product (weekdag vs. zaterdag) en hoeveel stuks
   een klant meestal neemt.
2. Daarmee genereren we maanden aan synthetische klanten (met dag-tot-dag
   variatie: drukke en rustige dagen).
3. Het realistische weegschaalmodel (weegschaal_model.py) zet die klanten om
   in meetwaarden zoals een echte load cell ze zou sturen (ruis, pieken,
   bijvullen, ...).
4. Elke 5 minuten nemen we een 'foto': de features zoals de backend ze live
   zou berekenen + het echte antwoord (label):
       label = aantal open minuten tot de bak leeg is als niemand bijvult.
   Omdat we in de simulatie de vraag van ALLE klanten kennen (ook van klanten
   die een lege bak aantreffen), kunnen we dat exact berekenen.

Uitvoer: data/training_synthetisch.csv en data/test_kassa.csv
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
from features import (FEATURE_NAMEN, MAX_HORIZON_MIN, BakHistoriek,  # noqa: E402
                      open_minuten_tussen)
from weegschaal_model import (GESLOTEN_WEEKDAGEN, OPENING_UUR, PRODUCTEN,  # noqa: E402
                              SLUITING_UUR, Klant, _ts, kassa_naar_klanten,
                              meet, meet_tijdstippen, simuleer_bak)

HIER = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HIER, "..", "data")
MEET_INTERVAL_S = 10
SAMPLE_ELKE_MIN = 5


def laad_kassa() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA, "kassa_data_groot.csv"), parse_dates=["timestamp"])
    df["bak_id"] = df["bak_id"].astype(str).str.zfill(2)
    return df


def leer_vraagprofiel(df: pd.DataFrame):
    """Gemiddeld aantal transacties per uur, per bak, voor weekdag en zaterdag."""
    df = df.copy()
    df["dag"] = df.timestamp.dt.date
    df["zat"] = df.timestamp.dt.weekday == 5
    df["uur"] = df.timestamp.dt.hour
    profiel = {}
    for bak in PRODUCTEN:
        d = df[df.bak_id == bak]
        for zat in (False, True):
            n_dagen = df[df.zat == zat].dag.nunique()
            per_uur = d[d.zat == zat].groupby("uur").size() / max(n_dagen, 1)
            profiel[(bak, zat)] = {u: float(per_uur.get(u, 0.0)) for u in range(OPENING_UUR, SLUITING_UUR)}
    aantallen = {bak: df[df.bak_id == bak].aantal_verkocht.values for bak in PRODUCTEN}
    return profiel, aantallen


def synthetische_klanten(bak, start, n_dagen, profiel, aantallen, rng):
    klanten = []
    for i in range(n_dagen):
        dag = start + timedelta(days=i)
        if dag.weekday() in GESLOTEN_WEEKDAGEN:
            continue
        zat = dag.weekday() == 5
        dagfactor = rng.lognormal(0, 0.2)          # drukke/rustige dag (weer, promo, ...)
        for uur, rate in profiel[(bak, zat)].items():
            uurfactor = rng.lognormal(0, 0.15)
            n = rng.poisson(rate * dagfactor * uurfactor)
            t0 = _ts(dag.replace(hour=uur))
            for t in np.sort(rng.uniform(t0, t0 + 3600, n)):
                klanten.append(Klant(tijd=float(t), aantal=int(rng.choice(aantallen[bak]))))
    return klanten


def maak_samples(bak, sim, start, eind, rng, bron):
    """Laat metingen door de live feature-pipeline lopen en voeg labels toe."""
    t = meet_tijdstippen(start, eind, MEET_INTERVAL_S)
    tt, metingen = meet(sim, t, rng)
    stuk = PRODUCTEN[bak]["stuk_gewicht"]

    cum_vraag = np.concatenate([[0.0], np.cumsum(sim.vraag_gram)])
    hist = BakHistoriek(bak)
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
        f["bron"] = bron
        rijen.append(f)
    return rijen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dagen", type=int, default=150, help="aantal synthetische dagen")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    kassa = laad_kassa()
    profiel, aantallen = leer_vraagprofiel(kassa)

    # ---------- synthetische trainingsdata ----------
    start = datetime(2026, 3, 2)
    eind = start + timedelta(days=args.dagen)
    alle = []
    for bak in PRODUCTEN:
        t0 = time.time()
        klanten = synthetische_klanten(bak, start, args.dagen, profiel, aantallen, rng)
        sim = simuleer_bak(bak, klanten, start, eind + timedelta(days=3), rng)
        rijen = maak_samples(bak, sim, start, eind, rng, "synthetisch")
        alle += rijen
        print(f"Bak {bak} ({PRODUCTEN[bak]['naam']}): {len(klanten)} klanten, "
              f"{len(sim.bijvullingen)} bijvullingen, {len(rijen)} samples  [{time.time()-t0:.0f}s]")
    pd.DataFrame(alle).to_csv(os.path.join(DATA, "training_synthetisch.csv"), index=False)

    # ---------- testdata: de echte kassaperiode, nooit gezien tijdens training ----------
    kstart = kassa.timestamp.min().to_pydatetime().replace(hour=0, minute=0, second=0)
    keind = kassa.timestamp.max().to_pydatetime() + timedelta(seconds=1)
    test = []
    for bak in PRODUCTEN:
        d = kassa[kassa.bak_id == bak]
        klanten = kassa_naar_klanten(d.timestamp, d.aantal_verkocht, rng)
        sim = simuleer_bak(bak, klanten, kstart, keind, rng)
        test += maak_samples(bak, sim, kstart, keind, rng, "kassa")
    pd.DataFrame(test).to_csv(os.path.join(DATA, "test_kassa.csv"), index=False)
    print(f"Training: {len(alle)} rijen, test (kassaperiode): {len(test)} rijen")
    print("Kolommen:", FEATURE_NAMEN + ["label_min"])


if __name__ == "__main__":
    main()
