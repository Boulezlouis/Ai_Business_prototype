"""
GramIQ weegschaal-simulator op ECHTE verkoopdata.

In plaats van elke kassatransactie als één grote gewichtssprong door te sturen,
simuleert dit script wat een echte load cell onder de bak zou meten:

  * elke 10 s (simulatietijd) het ABSOLUTE nettogewicht van de bak;
  * klanten nemen producten stuk per stuk weg (bananen per tros);
  * hand-/leunpieken, sensorruis, temperatuurdrift, af en toe een gemiste meting;
  * personeel vult de bak stapsgewijs bij nadat hij (bijna) leeg is;
  * de vraag komt uit echte verkopen van een supermarkt (data/annex2.csv):
    elke klant neemt exact het gewicht dat hij echt kocht, op het echte tijdstip.

Standaard wordt een week uit de testperiode afgespeeld (juni 2023), die de AI
nooit gezien heeft. Data beschikbaar van 1 juli 2020 tot 30 juni 2023.

Gebruik:
    python simulator.py                       # 1 echte seconde = 1 gesimuleerde minuut
    python simulator.py --snelheid 10         # trager
    python simulator.py --start "2023-03-11 08:30" --dagen 3
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
from echte_data import verkopen_per_bak  # noqa: E402
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
    ap.add_argument("--start", default="2023-06-01 08:30", help='starttijd, bv. "2023-03-11 08:30"')
    ap.add_argument("--dagen", type=float, default=7, help="hoeveel dagen afspelen")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--geen-mqtt", action="store_true", help="niet publiceren, enkel printen")
    ap.add_argument("--stil", action="store_true", help="niet elke meting printen")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    stream_start = datetime.fromisoformat(args.start)
    sim_start = stream_start.replace(hour=0, minute=0, second=0, microsecond=0)
    sim_eind = stream_start + timedelta(days=args.dagen)
    print(f"Echte verkopen laden ({sim_start:%d-%m-%Y} - {sim_eind:%d-%m-%Y}) ...")
    df = verkopen_per_bak(PRODUCTEN, sim_start, sim_eind + timedelta(hours=1))
    if df.empty:
        sys.exit("Geen verkopen in die periode. Kies een start tussen 2020-07-01 en 2023-06-30.")

    # ---- per bak de fysieke gebeurtenissen + meetwaarden berekenen ----
    stromen = []
    for bak, p in PRODUCTEN.items():
        d = df[df.bak_id == bak]
        klanten = kassa_naar_klanten(d.ts, d.gram, p["stuk_gewicht"], rng)
        sim = simuleer_bak(bak, klanten, sim_start, sim_eind, rng)
        t = meet_tijdstippen(stream_start, sim_eind, args.interval)
        # elke weegschaal meet op een eigen moment (niet allemaal tegelijk)
        t = t + rng.uniform(0, args.interval)
        tt, gewicht = meet(sim, t, rng)
        stromen.append(pd.DataFrame({"t": tt, "bak_id": bak, "gewicht": gewicht}))
        print(f"  Bak {bak} {p['naam']:15s}: {len(klanten):5d} echte klanten ({d.gram.sum()/1000:6.1f} kg), "
              f"{len(sim.bijvullingen)} keer bijgevuld")
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
