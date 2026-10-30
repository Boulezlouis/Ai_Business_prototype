# GramIQ – prototype op echte verkoopdata

Zelfde opzet als `GramIQ_Project_v2_Met_Dashboard` (simulator → MQTT → backend met
AI en database → React-dashboard), maar de klanten komen nu uit **878.503 echte
verkooptransacties** van een supermarkt, in plaats van uit verzonnen data.

## De dataset

**Supermarket Sales Data**, Kaggle: <https://www.kaggle.com/datasets/yapwh1208/supermarket-sales-data>
(groenten van één supermarkt, 1 juli 2020 – 30 juni 2023, 251 producten in 6 categorieën).

| Bestand | Inhoud |
|---|---|
| `data/annex1.csv` | productcode, naam, categorie |
| `data/annex2.csv` | **elke verkoop**: datum, tijdstip (tot op de ms), product, **kg**, prijs/kg, verkoop of retour, korting |
| `data/annex3.csv` | groothandelsprijs per product per dag |
| `data/annex4.csv` | verliespercentage (bederf) per product |

`annex2.csv` is 52 MB en staat daarom niet in git (zie `.gitignore`). Download de dataset
en zet de vier CSV-bestanden in `data/`.

Wat we in de data zien:
* open elke dag van ±9u tot 22u; drukst rond 10–11u en 16–18u, in het weekend ±40% meer;
* een klant koopt meestal 0,3–0,7 kg (mediaan 0,44 kg); 461 retours, 47.000 verkopen met korting;
* 10 dagen zonder verkopen (o.a. Chinees Nieuwjaar 2021).

## De vier bakken

| Bak | Product (naam in de data) | Verkoopdagen | Gem. per dag | Verlies |
|---|---|---|---|---|
| 01 | Broccoli | 1.076 | 25,6 kg | 9,3% |
| 02 | Lotuswortel (*Net Lotus Root (1)*) | 1.057 | 25,7 kg | 5,5% |
| 03 | Aubergine (*Eggplant (2)*) | 1.022 | 13,3 kg | 6,1% |
| 04 | Groene paprika (*Wuhu Green Pepper (1)*) | 858 (vanaf 2021) | 32,8 kg | 5,7% |

Losse groenten die per kg verkocht worden, bijna elke dag in de verkoop. Stukgewicht
(bv. broccoli 400 g) en capaciteit van de bakken (10–15 kg) zijn **aannames**: die
staan niet in de data. Aanpassen kan in `ai/weegschaal_model.py` → `PRODUCTEN`.

## Wat is echt en wat is gesimuleerd?

| Echt (uit de data) | Gesimuleerd (`ai/weegschaal_model.py`) |
|---|---|
| tijdstip van elke aankoop | klant staat 2–15 min vóór het afrekenen aan de bak |
| gewicht van elke aankoop (kg) | dat gewicht wordt stuk per stuk uit de bak genomen |
| welke dagen een product verkocht werd | sensorruis, hand in de bak, iemand die leunt, wifi-uitval |
| groothandelsprijs en verlies | personeel dat bijvult (ochtend + als de bak onder 12–30% zakt) |

## Opstarten

Python 3.10+ en Node.js nodig.

1. **DATA**: zet `annex1.csv` t/m `annex4.csv` in `data/`.

2. **AI TRAINEN** (getrainde modellen zitten al in `ai/`; enkel nodig als de backend
   "niet geladen" meldt, bv. door een andere scikit-learn-versie; getraind met 1.9.1):
   ```
   cd ai
   pip install -r requirements.txt
   python genereer_dataset.py   # ±9 min: 3 jaar weegschaalmetingen -> data/samples_echt.csv.gz + data/vraagprofiel.json
   python train_model.py        # ±1 min: AI 'wanneer is de bak leeg'  -> ai/model_leeg.joblib
   python bestelmodel.py        # ±7 min: AI 'hoeveel bestellen'       -> ai/model_bestelling.joblib
   python visualiseer.py        # optioneel: grafiek voor het verslag  -> ai/voorbeeld_voorspelling.png
   ```
   Heb je `data/samples_echt.csv.gz` al, dan kan je `genereer_dataset.py` overslaan
   wanneer je enkel opnieuw wil trainen.

3. **MQTT-BROKER** (terminal 1): `docker-compose up -d`

4. **BACKEND** (terminal 2):
   ```
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```

5. **DASHBOARD** (terminal 3):
   ```
   cd frontend
   npm install
   npm run dev          # http://localhost:3000
   ```

6. **SIMULATOR** (terminal 4):
   ```
   cd simulator
   pip install -r requirements.txt
   python simulator.py
   ```
   Standaard: een week vanaf 1 juni 2023 (testperiode, nooit gezien door de AI),
   1 echte seconde = 1 gesimuleerde minuut. Opties: `--start "2023-03-11 08:30"`,
   `--dagen 3`, `--snelheid 300`, `--stil`, `--geen-mqtt`.
   Elke datum tussen 1 juli 2020 en 30 juni 2023 kan.

## AI 1 – wanneer is de bak leeg?

Zelfde aanpak als in v2 (`features.py`, Gradient Boosting met een foutmarge van 80%),
maar getraind op **echte vraag** en met een extra **vraagprofiel**: hoeveel er normaal
per minuut verkocht wordt, per product en weekdag (`data/vraagprofiel.json`, enkel uit
de trainingsperiode). Daarmee weet de AI dat er om 17u een piek komt, en of het vandaag
drukker is dan normaal (`drukte`).

* **Train**: jul 2020 – jun 2022 · **validatie**: jul – dec 2022 · **test**: jan – jun 2023 (ongezien)
* Label: aantal open minuten tot de bak leeg is als niemand bijvult (berekend uit de echte toekomstige verkopen).

| Testperiode jan–jun 2023 | AI | Baseline (gewicht ÷ verbruik) |
|---|---|---|
| Gemiddelde fout, alle momenten | 80 min | 123 min |
| Gemiddelde fout als de bak < 25% vol is | **50 min** | 81 min |
| Echte waarde binnen de foutmarge | 84% | – |

| Als de AI zegt… | …is de bak echt leeg na (mediaan) | helft van de gevallen tussen |
|---|---|---|
| 0–30 min | 22 min | 11 – 40 min |
| 30–60 min | 51 min | 33 – 80 min |
| 60–120 min | 102 min | 70 – 148 min |

De fout is groter dan in v2 (15 min): echte klanten komen in golven en soms koopt één
klant 5 kg in één keer. Dat is niet te voorspellen, maar wel eerlijk gemeten. De AI is
wel goed gekalibreerd: de voorspellingen kloppen gemiddeld en de marge is betrouwbaar.

## AI 2 – hoeveel moet er besteld worden? (#20)

`ai/bestelmodel.py`. De groothandel levert 's ochtends vroeg; op dat moment voorspelt
het model per product de verkoop van vandaag uit de verkopen tot en met gisteren
(vorige dagen, zelfde weekdag vorige weken, kortingen, prijs, inkoopprijs van vandaag).
Eén model voor de 57 producten met minstens 300 verkoopdagen.

* **verwacht** = mediaan · **veilig** = 80%-kwantiel (volstaat op ±80% van de dagen)
* **bestellen** = veilig ÷ (1 − verlies%) · **kost** = bestellen × inkoopprijs van vandaag

| Testperiode jan–jun 2023 | Gem. fout AI | Zelfde dag vorige week | Gem. 7 dagen | Bestelling volstaat |
|---|---|---|---|---|
| Alle 57 producten | **4,7 kg** | 6,6 kg | 5,5 kg | 78% van de dagen |
| Broccoli | 5,0 kg | 7,8 kg | 6,4 kg | 83% |
| Lotuswortel | 4,8 kg | 6,7 kg | 5,9 kg | 81% |
| Aubergine | 4,5 kg | 6,3 kg | 5,7 kg | 76% |
| Groene paprika | 6,7 kg | 10,9 kg | 9,0 kg | 84% |

Het dashboard toont het bestelvoorstel van de gesimuleerde dag, met wat er tot nu toe
echt verkocht is.

## Backend-API

Zelfde als v2 (met de SQLite-database van issue #6/#7, `data/gramiq.db`), plus:

* `GET /api/bestelling` → bestelvoorstel per bak voor de huidige (gesimuleerde) dag:
  `verwacht_kg`, `veilig_kg`, `verlies_pct`, `bestel_kg`, `inkoopprijs_per_kg`, `kost`,
  `verkocht_tot_nu_kg`
* `GET /api/bakken`, `/api/historiek/{bak}?uren=4`, `/api/metingen/{bak}`, `/api/producten`,
  `/api/weegschalen`, `/api/database`, `/api/model`

## Mappen

| Map | Inhoud |
|---|---|
| `ai/` | `echte_data.py` (inlezen), weegschaalmodel, features + vraagprofiel, datasetgenerator, beide modellen |
| `simulator/` | speelt echte verkopen af als weegschaalmetingen over MQTT |
| `backend/` | FastAPI + SQLite: metingen ontvangen, filteren, bewaren, voorspellen |
| `frontend/` | React-dashboard (bakken, grafiek, bestelvoorstel) |
| `data/` | dataset + gegenereerde bestanden |
