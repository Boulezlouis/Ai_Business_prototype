# GramIQ Project (Inclusief Frontend + AI)

Dit project bevat het volledige pakket: Infrastructuur (Docker), realistische Simulator,
Backend (FastAPI) met voorspellende AI, en Frontend (React).

## Mappen

| Map          | Inhoud |
|--------------|--------|
| `ai/`        | Weegschaalmodel, feature-berekening, dataset-generator, training van de AI |
| `simulator/` | Stuurt realistische weegschaalmetingen over MQTT |
| `backend/`   | Ontvangt metingen, filtert ruis, voorspelt wanneer elke bak leeg is |
| `frontend/`  | Live dashboard |
| `data/`      | Kassadata + gegenereerde trainings- en testdata |

## Opstarten

1. ZET DE DATA KLAAR:
   Plaats het `kassa_data_groot.csv` bestand in de `/data` map.

2. TRAIN DE AI (eenmalig, ±1 minuut):
   `cd ai`
   `pip install -r requirements.txt`
   `python genereer_dataset.py`   (maakt data/training_synthetisch.csv en data/test_kassa.csv)
   `python train_model.py`        (maakt ai/model_leeg.joblib + ai/model_rapport.json)
   `python visualiseer.py`        (optioneel: grafiek voor het verslag)
   Er staat al een getraind model in de map. Geeft de backend toch
   "AI-model niet geladen" (bv. door een andere scikit-learn-versie), voer dan
   deze stap opnieuw uit.

3. START DE INFRASTRUCTUUR (Terminal 1):
   `docker-compose up -d`

4. START DE BACKEND (Terminal 2):
   `cd backend`
   `pip install -r requirements.txt`
   `uvicorn main:app --reload`
   (De backend draait nu op poort 8000)

5. START DE REACT FRONTEND (Terminal 3):
   `cd frontend`
   `npm install`
   `npm run dev`
   (Ga in je browser naar http://localhost:3000)

6. START DE SIMULATOR (Terminal 4):
   `cd simulator`
   `pip install -r requirements.txt`
   `python simulator.py`
   Opties: `--snelheid 60` (standaard: 1 echte seconde = 1 gesimuleerde minuut),
   `--start "2026-10-24 11:00"`, `--stil`, `--geen-mqtt` (test zonder broker).

## Wat doet de simulator nu (realistisch)?

Vroeger stuurde de simulator elke kassatransactie als één sprong door
(bv. "13 bananen = -2602 g"). Een echte weegschaal werkt anders. De nieuwe
simulator (`ai/weegschaal_model.py`) bootst na wat een load cell onder de bak meet:

* **Absoluut nettogewicht** van de bak elke 10 s (na tarra), geen "afname per verkoop".
* **Stuk per stuk**: een klant neemt tomaten/appels één voor één (1,5–4 s ertussen),
  bananen per tros. Elk stuk weegt net iets anders (bv. tomaat 130 g ± 14 g).
* De klant staat 2–20 minuten **vóór** het afrekenen aan de bak (vraag = echte kassadata).
* **Hand in de bak** (+30–300 g kort), **iemand die leunt** (+0,8–3 kg, 3–15 s).
* **Sensorruis** (± 1,5 g), **temperatuurdrift** (± 4 g over de dag), afronding op 1 g,
  1% gemiste metingen (WiFi).
* Klant legt soms een stuk **terug** en neemt een ander.
* **Bijvullen** door personeel: pas wanneer de bak onder 12–30% zakt, met een
  reactietijd van meestal 5–25 min (soms veel langer). Bijvullen gebeurt in
  scheppen/kratten van 1,8–4,5 kg, soms eerst rotte stuks eruit.
  Is het personeel te laat, dan is de bak écht **leeg** en vinden klanten niets.
* Elke ochtend om 7u30 worden alle bakken gevuld. Open 8u–20u, zondag gesloten.

De backend filtert de korte pieken weg met een mediaanfilter (laatste 7 metingen).

## Voorspellende AI: wanneer is de bak leeg?

* **Vraag**: binnen hoeveel minuten (openingstijd) is de bak leeg als niemand bijvult?
* **Trainingsdata**: 150 gesimuleerde dagen. Het vraagpatroon (klanten per uur,
  weekdag vs. zaterdag, stuks per klant) is geleerd uit de kassadata, met
  variatie in drukte per dag en per uur. Elke 5 minuten wordt een sample genomen.
* **Features** (zelfde code live en bij training, `ai/features.py`): gefilterd gewicht,
  % vol, geschat aantal stuks, verbruik (g/min) over 10/30/60/120 min en vandaag,
  uur, weekdag, tijd tot sluiting, tijd sinds bijvulling, product.
* **Model**: Gradient Boosting (scikit-learn `HistGradientBoostingRegressor`),
  plus een foutmarge die het 80%-interval geeft.
* **Resultaat** op de echte kassaperiode (19–31 okt, nooit gezien tijdens training):

  | | Gem. fout AI | Gem. fout baseline (gewicht ÷ verbruik) |
  |---|---|---|
  | Alle momenten | 14,9 min | 45,6 min |
  | Bak leeg binnen 1 uur | 10,8 min | 31,4 min |
  | Echte waarde binnen de foutmarge | 81% | – |

* **Backend-API**:
  * `GET /api/bakken` → gewicht, % vol, stuks, status en `voorspelling`
    (`minuten`, `leeg_om`, `leeg_vroegst`, `leeg_laatst`, `verbruik_g_per_min`)
  * `GET /api/historiek/{bak_id}` → gewicht + voorspelling per minuut (laatste 4 u)
  * `GET /api/model` → evaluatierapport van het model
