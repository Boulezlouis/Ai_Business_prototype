import time
import pandas as pd
import paho.mqtt.client as mqtt
import json

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "gramiq/weegschaal"

print("Verbinden met MQTT broker...")
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)

print("Laden van data/kassa_data_groot.csv...")
try:
    df = pd.read_csv("../data/kassa_data_groot.csv")
    print("Start simulatie (1 seconde per transactie)...")
    for index, row in df.iterrows():
        payload = {
            "bak_id": str(row["bak_id"]).zfill(2),
            "product": row["product_naam"],
            "gewicht_afname_gram": row["geschat_gewicht_gram"],
            "timestamp": row["timestamp"]
        }
        topic = f"{MQTT_TOPIC}/{payload['bak_id']}"
        client.publish(topic, json.dumps(payload))
        print(f"Gepubliceerd naar {topic}: {payload}")
        time.sleep(1) 
except FileNotFoundError:
    print("Fout: kassa_data_groot.csv niet gevonden. Zet deze in de /data map.")
except Exception as e:
    print(f"Fout: {e}")
