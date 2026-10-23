"""
GramIQ weegschaal-simulator (realistische versie).

In plaats van elke kassatransactie als één grote gewichtssprong door te sturen,
simuleert dit script wat een echte load cell onder de bak zou meten:

  * elke 10 s (simulatietijd) het ABSOLUTE nettogewicht van de bak;
  * klanten nemen producten stuk per stuk weg (bananen per tros);
  * hand-/leunpieken, sensorruis, temperatuurdrift, af en toe een gemiste meting;
  * personeel vult de bak stapsgewijs bij nadat hij (bijna) leeg is;
  * de vraag komt uit de echte kassadata (data/kassa_data_groot.csv).

Gebruik:
    python simulator.py                       # 1 echte seconde = 1 gesimuleerde minuut
    python simulator.py --snelheid 10         # trager
    python simulator.py --start "2026-10-24 11:00"
    python simulator.py --geen-mqtt           # enkel printen (test zonder broker)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HIER, "..", "ai"))
from weegschaal_model import (PRODUCTEN, kassa_naar_klanten, meet,  # noqa: E402
                              meet_tijdstippen, naar_datetime, simuleer_bak)

MQTT_TOPIC = "gramiq/weegschaal"


def main():
    ap = argparse.ArgumentParser(description="Realistische GramIQ weegschaal-simulator")
    ap.add_argument("--broker", default=os.environ.get("MQTT_HOST", "localhost"))
    ap.add_argument("--poort", type=int, default=1883)
    ap.add_argument("--snelheid", type=float, default=60.0,
                    help="versnelling t.o.v. echte tijd (60 = 1 s echt is 1 min gesimuleerd)")
    ap.add_argument("--interval", type=float, default=10.0, help="meetinterval weegschaal in seconden")
    ap.add_argument("--start", default=None, help='starttijd, bv. "2026-10-20 07:00"')
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--geen-mqtt", action="store_true", help="niet publiceren, enkel printen")
    ap.add_argument("--stil", action="store_true", help="niet elke meting printen")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    csv = os.path.join(HIER, "..", "data", "kassa_data_groot.csv")
    print(f"Laden van {csv} ...")
    df = pd.read_csv(csv, parse_dates=["timestamp"])
    df["bak_id"] = df["bak_id"].astype(str).str.zfill(2)

    sim_start = df.timestamp.min().to_pydatetime().replace(hour=0, minute=0, second=0)
    sim_eind = df.timestamp.max().to_pydatetime() + timedelta(minutes=30)
    stream_start = datetime.fromisoformat(args.start) if args.start else sim_start

    # ---- per bak de fysieke gebeurtenissen + meetwaarden berekenen ----
    stromen = []
    for bak, p in PRODUCTEN.items():
        d = df[df.bak_id == bak]
        klanten = kassa_naar_klanten(d.timestamp, d.aantal_verkocht, rng)
        sim = simuleer_bak(bak, klanten, sim_start, sim_eind, rng)
        t = meet_tijdstippen(stream_start, sim_eind, args.interval)
        # elke weegschaal meet op een eigen moment (niet allemaal tegelijk)
        t = t + rng.uniform(0, args.interval)
        tt, gewicht = meet(sim, t, rng)
        stromen.append(pd.DataFrame({"t": tt, "bak_id": bak, "gewicht": gewicht}))
        print(f"  Bak {bak} {p['naam']:8s}: {len(klanten)} klanten, {len(sim.bijvullingen)} bijvullingen")
    alle = pd.concat(stromen).sort_values("t").reset_index(drop=True)

    client = None
    if not args.geen_mqtt:
        import paho.mqtt.client as mqtt
        print(f"Verbinden met MQTT broker {args.broker}:{args.poort} ...")
        client = mqtt.Client()
        client.connect(args.broker, args.poort, 60)
        client.loop_start()

    print(f"Start simulatie vanaf {stream_start} (snelheid x{args.snelheid:g}) ...")
    vorige_t = None
    vorige_dag = None
    for rij in alle.itertuples(index=False):
        if vorige_t is not None:
            dt = rij.t - vorige_t
            if dt > 1800:          # nacht/zondag: niet wachten
                dt = 0
            if dt > 0:
                time.sleep(dt / args.snelheid)
        vorige_t = rij.t

        ts = naar_datetime(rij.t)
        if ts.date() != vorige_dag:
            vorige_dag = ts.date()
            print(f"\n===== {ts:%A %d-%m-%Y} =====")
        p = PRODUCTEN[rij.bak_id]
        payload = {
            "bak_id": rij.bak_id,
            "product": p["naam"],
            "gewicht_gram": float(rij.gewicht),
            "capaciteit_gram": p["capaciteit"],
            "stuk_gewicht_gram": p["stuk_gewicht"],
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        }
        topic = f"{MQTT_TOPIC}/{rij.bak_id}"
        if client:
            client.publish(topic, json.dumps(payload))
        if not args.stil:
            print(f"{payload['timestamp']}  {topic}  {payload['gewicht_gram']:8.0f} g")

    if client:
        client.loop_stop()
    print("Simulatie klaar.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGestopt.")
