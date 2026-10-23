from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import paho.mqtt.client as mqtt
import json
import os
import sys
import threading
from collections import deque
from datetime import datetime

# gedeelde code (ruisfilter + features + model) staat in ../ai
HIER = os.path.dirname(os.path.abspath(__file__))
AI_MAP = os.path.join(HIER, "..", "ai")
sys.path.insert(0, AI_MAP)
from features import BakHistoriek, MAX_HORIZON_MIN, pas_marges_toe, tel_open_minuten_op  # noqa: E402
from weegschaal_model import PRODUCTEN, _ts  # noqa: E402

app = FastAPI()

# CORS toegevoegd zodat onze React frontend mag communiceren met deze backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------- AI-model
MODEL = None
RAPPORT = None
try:
    import joblib
    import pandas as pd
    MODEL = joblib.load(os.path.join(AI_MAP, "model_leeg.joblib"))
    with open(os.path.join(AI_MAP, "model_rapport.json")) as f:
        RAPPORT = json.load(f)
    print("AI-model geladen (ai/model_leeg.joblib)")
except Exception as e:  # model ontbreekt of andere sklearn-versie
    print(f"!! AI-model niet geladen ({e}). Gebruik baseline. "
          f"Train opnieuw met: cd ai && python genereer_dataset.py && python train_model.py")

lock = threading.Lock()
historiek = {}        # bak_id -> BakHistoriek
laatste_metingen = {}  # bak_id -> dict voor het dashboard
grafiek = {}          # bak_id -> deque van punten per minuut
laatste_voorspel_minuut = {}


def voorspel(f: dict) -> dict:
    if MODEL is not None:
        X = pd.DataFrame([f])[MODEL["features"]]
        m = min(max(float(MODEL["hoofd"].predict(X)[0]), 0.0), MAX_HORIZON_MIN)
        lo, hi = pas_marges_toe([m], MODEL["marges"])
        lo, hi = float(lo[0]), float(hi[0])
        methode = "AI (gradient boosting)"
    else:
        m = lo = hi = f["baseline_min"]
        methode = "baseline (gewicht / verbruik)"
    m = min(max(m, 0.0), MAX_HORIZON_MIN)
    lo = min(max(min(lo, m), 0.0), MAX_HORIZON_MIN)
    hi = min(max(hi, m), MAX_HORIZON_MIN)
    return {"minuten": round(m, 1), "minuten_min": round(lo, 1), "minuten_max": round(hi, 1), "methode": methode}


def bepaal_status(pct: float, minuten):
    if pct < 3:
        return "leeg"
    if (minuten is not None and minuten <= 15) or pct < 10:
        return "kritiek"
    if (minuten is not None and minuten <= 45) or pct < 25:
        return "laag"
    return "ok"


def verwerk(data: dict):
    bak_id = str(data.get("bak_id", "")).zfill(2)
    if bak_id not in PRODUCTEN or "gewicht_gram" not in data:
        return
    ts = datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S")
    t = _ts(ts)
    with lock:
        h = historiek.get(bak_id)
        if h is None or (h.laatste_t is not None and t < h.laatste_t - 60):
            # nieuwe bak of simulator herstart -> opnieuw beginnen
            h = historiek[bak_id] = BakHistoriek(bak_id)
            grafiek[bak_id] = deque(maxlen=240)
            laatste_voorspel_minuut.pop(bak_id, None)
        h.voeg_toe(t, float(data["gewicht_gram"]))

        p = PRODUCTEN[bak_id]
        g = max(h.gefilterd, 0.0)
        pct = 100 * g / p["capaciteit"]
        info = laatste_metingen.get(bak_id, {})
        info.update({
            "bak_id": bak_id,
            "product": p["naam"],
            "gewicht_ruw_gram": data["gewicht_gram"],
            "gewicht_gram": round(g),
            "pct_vol": round(pct, 1),
            "stuks": int(round(g / p["stuk_gewicht"])),
            "capaciteit_gram": p["capaciteit"],
            "timestamp": data["timestamp"],
        })

        # voorspelling 1x per gesimuleerde minuut
        minuut = int(t // 60)
        if laatste_voorspel_minuut.get(bak_id) != minuut:
            laatste_voorspel_minuut[bak_id] = minuut
            f = h.features()
            if f is None:
                info["voorspelling"] = None
            else:
                v = voorspel(f)
                v["leeg_om"] = tel_open_minuten_op(t, v["minuten"]).strftime("%Y-%m-%d %H:%M")
                v["leeg_vroegst"] = tel_open_minuten_op(t, v["minuten_min"]).strftime("%H:%M")
                v["leeg_laatst"] = tel_open_minuten_op(t, v["minuten_max"]).strftime("%H:%M")
                v["verbruik_g_per_min"] = round(f["verbruik_30"], 1)
                info["voorspelling"] = v
            vm = info.get("voorspelling")
            grafiek[bak_id].append({
                "tijd": ts.strftime("%H:%M"),
                "gewicht": round(g),
                "voorspelling_min": vm["minuten"] if vm else None,
            })
        vm = info.get("voorspelling")
        info["status"] = bepaal_status(pct, vm["minuten"] if vm else None)
        laatste_metingen[bak_id] = info


def on_message(client, userdata, msg):
    try:
        verwerk(json.loads(msg.payload.decode()))
    except Exception as e:
        print(f"Fout bij verwerken bericht: {e}")


def start_mqtt():
    mqtt_client = mqtt.Client()
    mqtt_client.on_message = on_message
    try:
        mqtt_client.connect(os.environ.get("MQTT_HOST", "localhost"), 1883, 60)
        mqtt_client.subscribe("gramiq/weegschaal/#")
        mqtt_client.loop_forever()
    except Exception as e:
        print(f"Kan niet verbinden met MQTT: {e}")


threading.Thread(target=start_mqtt, daemon=True).start()


@app.get("/")
def read_root():
    return {"status": "GramIQ Backend is online", "ai_model": MODEL is not None}


@app.get("/api/bakken")
def get_bakken():
    with lock:
        return dict(sorted(laatste_metingen.items()))


@app.get("/api/historiek/{bak_id}")
def get_historiek(bak_id: str):
    bak_id = bak_id.zfill(2)
    with lock:
        if bak_id not in grafiek:
            raise HTTPException(404, "Onbekende bak")
        return list(grafiek[bak_id])


@app.get("/api/model")
def get_model():
    return {"geladen": MODEL is not None, "rapport": RAPPORT}
