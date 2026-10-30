from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import paho.mqtt.client as mqtt
import atexit
import json
import os
import sys
import threading
from datetime import datetime, timedelta

# gedeelde code (ruisfilter + features + model) staat in ../ai
HIER = os.path.dirname(os.path.abspath(__file__))
AI_MAP = os.path.join(HIER, "..", "ai")
sys.path.insert(0, AI_MAP)
from features import BakHistoriek, MAX_HORIZON_MIN, pas_marges_toe, tel_open_minuten_op  # noqa: E402
from weegschaal_model import PRODUCT_CODE, PRODUCTEN, _ts  # noqa: E402
from database import Database  # noqa: E402

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

# ---------------------------------------------------------------- database
db = Database()
if db.vul_standaard(PRODUCTEN, PRODUCT_CODE):
    print("Nieuwe database aangemaakt met de standaard producten en weegschalen.")
print(f"Database: {db.pad}")
atexit.register(db.sluit)

lock = threading.Lock()
config = {}           # weegschaal_id -> product + tarra (uit de database)
historiek = {}        # bak_id -> BakHistoriek
laatste_metingen = {}  # bak_id -> dict voor het dashboard
laatste_voorspel_minuut = {}

TIJD_FORMAAT = "%Y-%m-%d %H:%M:%S"


def herlaad_config():
    """Leest weegschalen + gekoppelde producten opnieuw uit de database."""
    global config
    config = db.weegschaal_config()


def product_van(c: dict) -> dict:
    """Database-rij -> product-dict zoals BakHistoriek het verwacht."""
    return {"naam": c["naam"], "stuk_gewicht": c["stuk_gewicht_gram"],
            "capaciteit": c["capaciteit_gram"], "stuk_sd": c["stuk_sd_gram"], "tros": bool(c["per_tros"])}


def nieuwe_historiek(bak_id: str) -> BakHistoriek:
    c = config[bak_id]
    return BakHistoriek(bak_id, product=product_van(c), product_code=c["ai_code"])


def voorspel(f: dict) -> dict:
    if MODEL is not None and f["product_code"] is not None:
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


def dashboard_info(bak_id: str, h: BakHistoriek, ruw: float, timestamp: str) -> dict:
    c = config[bak_id]
    g = max(h.gefilterd, 0.0)
    return {
        "bak_id": bak_id,
        "product": c["naam"],
        "gewicht_ruw_gram": ruw,
        "gewicht_gram": round(g),
        "pct_vol": round(100 * g / c["capaciteit_gram"], 1),
        "stuks": int(round(g / c["stuk_gewicht_gram"])),
        "capaciteit_gram": c["capaciteit_gram"],
        "timestamp": timestamp,
    }


def verwerk(data: dict):
    bak_id = str(data.get("bak_id", "")).strip().zfill(2)
    if not str(data.get("bak_id", "")).strip() or "gewicht_gram" not in data:
        return
    ts = datetime.strptime(data["timestamp"], TIJD_FORMAAT)
    t = _ts(ts)
    ruw = float(data["gewicht_gram"])
    with lock:
        if bak_id not in config:
            herlaad_config()            # misschien net in de database toegevoegd
        c = config.get(bak_id)
        if c is None:
            db.registreer_onbekende_weegschaal(bak_id, data["timestamp"])
            herlaad_config()
            print(f"Nieuwe weegschaal {bak_id} geregistreerd; koppel er nog een product aan.")
            return
        if not c["actief"] or c["product_id"] is None:
            return                      # nog geen product gekoppeld -> niets te berekenen

        h = historiek.get(bak_id)
        if h is None or (h.laatste_t is not None and t < h.laatste_t - 60):
            # nieuwe bak of simulator herstart -> opnieuw beginnen
            h = historiek[bak_id] = nieuwe_historiek(bak_id)
            laatste_voorspel_minuut.pop(bak_id, None)
        h.voeg_toe(t, ruw - c["tarra_gram"])

        info = laatste_metingen.get(bak_id, {})
        info.update(dashboard_info(bak_id, h, data["gewicht_gram"], data["timestamp"]))
        db.bewaar_meting(bak_id, c["product_id"], data["timestamp"], ruw,
                         round(max(h.gefilterd, 0.0), 1), info["pct_vol"], info["stuks"])

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
                db.bewaar_voorspelling(bak_id, ts.strftime("%Y-%m-%d %H:%M"), v)
        vm = info.get("voorspelling")
        info["status"] = bepaal_status(info["pct_vol"], vm["minuten"] if vm else None)
        laatste_metingen[bak_id] = info


def herstel_uit_database():
    """Na een herstart: filter, verbruik en dashboard terugzetten uit de opgeslagen metingen."""
    herlaad_config()
    for bak_id, c in config.items():
        if not c["actief"] or c["product_id"] is None:
            continue
        rijen = db.laatste_dag_metingen(bak_id)
        if not rijen:
            continue
        h = historiek[bak_id] = nieuwe_historiek(bak_id)
        for r in rijen:
            h.voeg_toe(_ts(datetime.strptime(r["tijdstip"], TIJD_FORMAAT)), r["gewicht_ruw_gram"] - c["tarra_gram"])
        laatste = rijen[-1]
        info = dashboard_info(bak_id, h, laatste["gewicht_ruw_gram"], laatste["tijdstip"])
        v = db.laatste_voorspelling(bak_id)
        if v and v["tijdstip"] == laatste["tijdstip"][:16]:
            t = _ts(datetime.strptime(v["tijdstip"], "%Y-%m-%d %H:%M"))
            info["voorspelling"] = {
                "minuten": v["minuten"], "minuten_min": v["minuten_min"], "minuten_max": v["minuten_max"],
                "methode": v["methode"], "leeg_om": v["leeg_om"],
                "leeg_vroegst": tel_open_minuten_op(t, v["minuten_min"]).strftime("%H:%M"),
                "leeg_laatst": tel_open_minuten_op(t, v["minuten_max"]).strftime("%H:%M"),
                "verbruik_g_per_min": v["verbruik_g_per_min"],
            }
            laatste_voorspel_minuut[bak_id] = int(t // 60)
        else:
            info["voorspelling"] = None
        vm = info["voorspelling"]
        info["status"] = bepaal_status(info["pct_vol"], vm["minuten"] if vm else None)
        laatste_metingen[bak_id] = info
        print(f"  Bak {bak_id} hersteld uit database ({len(rijen)} metingen van {laatste['tijdstip'][:10]})")


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


with lock:
    herstel_uit_database()
if os.environ.get("GRAMIQ_GEEN_MQTT") != "1":
    threading.Thread(target=start_mqtt, daemon=True).start()


@app.on_event("shutdown")
def afsluiten():
    db.sluit()


@app.get("/")
def read_root():
    return {"status": "GramIQ Backend is online", "ai_model": MODEL is not None, "database": db.pad}


@app.get("/api/bakken")
def get_bakken():
    with lock:
        return dict(sorted(laatste_metingen.items()))


@app.get("/api/historiek/{bak_id}")
def get_historiek(bak_id: str, uren: float = Query(4, gt=0, le=24 * 31)):
    """Gefilterd gewicht + voorspelling per minuut, de laatste `uren` tot de laatste meting."""
    bak_id = bak_id.zfill(2)
    with lock:
        info = laatste_metingen.get(bak_id)
    if info is None:
        raise HTTPException(404, "Onbekende bak of nog geen metingen")
    tot = datetime.strptime(info["timestamp"], TIJD_FORMAAT)
    van = tot - timedelta(hours=uren)
    return db.historiek_per_minuut(bak_id, van.strftime(TIJD_FORMAAT), tot.strftime(TIJD_FORMAAT))


@app.get("/api/metingen/{bak_id}")
def get_metingen(bak_id: str, van: str | None = None, tot: str | None = None,
                 limiet: int = Query(1000, gt=0, le=100000)):
    """Ruwe opgeslagen metingen, bv. ?van=2026-10-24 08:00:00&tot=2026-10-24 12:00:00"""
    return db.metingen(bak_id.zfill(2), van, tot, limiet)


@app.get("/api/producten")
def get_producten():
    return db.producten()


@app.get("/api/weegschalen")
def get_weegschalen():
    return db.weegschalen()


@app.get("/api/database")
def get_database():
    return db.statistiek()


@app.get("/api/model")
def get_model():
    return {"geladen": MODEL is not None, "rapport": RAPPORT}
