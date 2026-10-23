# GramIQ Project (Inclusief Frontend)

Dit project bevat nu het volledige pakket: Infrastructuur (Docker), Simulator, Backend (FastAPI), en Frontend (React).

1. ZET DE DATA KLAAR:
   Plaats het `kassa_data_groot.csv` bestand in de `/data` map.

2. START DE INFRASTRUCTUUR (Terminal 1):
   `docker-compose up -d`

3. START DE BACKEND (Terminal 2):
   `cd backend`
   `pip install -r requirements.txt`
   `uvicorn main:app --reload`
   (De backend draait nu op poort 8000)

4. START DE REACT FRONTEND (Terminal 3):
   (Zorg dat je Node.js geïnstalleerd hebt op je PC)
   `cd frontend`
   `npm install`
   `npm run dev`
   (Ga in je browser naar http://localhost:3000)

5. START DE SIMULATOR (Terminal 4):
   `cd simulator`
   `pip install -r requirements.txt`
   `python simulator.py`

Zodra je de simulator start (stap 5), zul je de grafieken op je React-dashboard (stap 4) live zien bewegen!
