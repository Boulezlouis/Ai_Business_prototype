from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import paho.mqtt.client as mqtt
import json
import threading

app = FastAPI()

# CORS toegevoegd zodat onze React frontend mag communiceren met deze backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

laatste_metingen = {}

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        bak_id = data.get("bak_id")
        if bak_id:
            laatste_metingen[bak_id] = data
            print(f"Backend ontving data voor bak {bak_id}: {data}")
    except Exception as e:
        print(f"Fout bij verwerken bericht: {e}")

def start_mqtt():
    mqtt_client = mqtt.Client()
    mqtt_client.on_message = on_message
    try:
        mqtt_client.connect("localhost", 1883, 60)
        mqtt_client.subscribe("gramiq/weegschaal/#")
        mqtt_client.loop_forever()
    except Exception as e:
        print(f"Kan niet verbinden met MQTT: {e}")

threading.Thread(target=start_mqtt, daemon=True).start()

@app.get("/")
def read_root():
    return {"status": "GramIQ Backend is online"}

@app.get("/api/bakken")
def get_bakken():
    return laatste_metingen
