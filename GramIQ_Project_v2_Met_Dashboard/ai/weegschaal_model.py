"""
Realistisch model van een slimme weegschaal onder een groente/fruitbak.

Waarom dit bestaat
------------------
De oude simulator stuurde elke kassatransactie door als één sprong
(bv. "13 bananen = -2602 g in één keer"). Zo gedraagt een echte weegschaal
zich niet. Een echte weegschaal:

  * stuurt elke X seconden het ABSOLUTE nettogewicht van de bak (na tarra);
  * ziet een klant stuk per stuk producten wegnemen (tomaat per tomaat,
    bananen per tros), met een paar seconden tussen elke greep;
  * meet tijdens het grijpen tijdelijk iets méér (hand die op de bak drukt);
  * heeft sensorruis (+/- 1-2 g) en een kleine temperatuurdrift;
  * ziet soms een klant die een stuk teruglegt of iemand die even op de bak leunt;
  * wordt door personeel bijgevuld in een paar scheppen/kratten (stapsgewijs),
    meestal pas een tijdje nadat de bak bijna leeg is;
  * kan echt leeg zijn als het personeel te laat is (klanten vinden dan niets).

Deze module wordt gebruikt door:
  * simulator/simulator.py   -> live data over MQTT (vraag = echte kassadata)
  * ai/genereer_dataset.py   -> trainingsdata voor de voorspellende AI
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

# --------------------------------------------------------------------------
# Configuratie
# --------------------------------------------------------------------------
PRODUCTEN = {
    # stuk_gewicht / stuk_sd in gram, capaciteit = netto gewicht van een volle bak
    "07": {"naam": "Tomaten", "stuk_gewicht": 130.0, "stuk_sd": 14.0, "capaciteit": 20000.0, "tros": False},
    "08": {"naam": "Appels", "stuk_gewicht": 180.0, "stuk_sd": 18.0, "capaciteit": 18000.0, "tros": False},
    "09": {"naam": "Bananen", "stuk_gewicht": 200.0, "stuk_sd": 20.0, "capaciteit": 15000.0, "tros": True},
}
PRODUCT_CODE = {"07": 0, "08": 1, "09": 2}

OPENING_UUR = 8      # winkel open 08:00
SLUITING_UUR = 20    # winkel dicht 20:00
GESLOTEN_WEEKDAGEN = {6}  # zondag
OCHTENDVULLING = (7, 30)  # personeel vult alle bakken voor opening

SENSOR_RUIS_SD = 1.5        # gram
DRIFT_AMPLITUDE = 4.0       # gram, trage temperatuurdrift over de dag


def is_open(ts: datetime) -> bool:
    return ts.weekday() not in GESLOTEN_WEEKDAGEN and OPENING_UUR <= ts.hour < SLUITING_UUR


# --------------------------------------------------------------------------
# Datastructuren
# --------------------------------------------------------------------------
@dataclass
class Klant:
    tijd: float      # seconden sinds 1970 (naïeve lokale tijd) waarop de klant aan de bak staat
    aantal: int      # gewenst aantal stuks


@dataclass
class BakSimulatie:
    """Resultaat van een simulatie van één bak."""
    bak_id: str
    stappen_t: np.ndarray        # tijdstippen van gewichtsveranderingen
    stappen_delta: np.ndarray    # verandering in gram
    start_gewicht: float
    transienten: list            # (start, eind, amplitude, soort)
    vraag_t: np.ndarray          # tijdstip waarop klant iets wil
    vraag_gram: np.ndarray       # gewenste hoeveelheid (ook als bak leeg is)
    bijvullingen: list = field(default_factory=list)  # (start, eind)

    def echt_gewicht(self, t: np.ndarray) -> np.ndarray:
        """Exact (ruisvrij) nettogewicht op tijdstippen t."""
        cum = np.concatenate([[0.0], np.cumsum(self.stappen_delta)])
        idx = np.searchsorted(self.stappen_t, t, side="right")
        return self.start_gewicht + cum[idx]


# --------------------------------------------------------------------------
# Simulatie
# --------------------------------------------------------------------------
_EPOCH = datetime(1970, 1, 1)


def _ts(dt: datetime) -> float:
    """Naïeve datetime -> seconden (tijdzone-onafhankelijk, geen zomertijdproblemen)."""
    return (dt - _EPOCH).total_seconds()


def naar_datetime(t: float) -> datetime:
    return _EPOCH + timedelta(seconds=float(t))


def simuleer_bak(bak_id: str, klanten: list[Klant], start: datetime, eind: datetime,
                 rng: np.random.Generator) -> BakSimulatie:
    """Simuleer de fysieke gebeurtenissen op één bak.

    klanten: lijst van klanten (tijd waarop ze aan de bak staan + gewenst aantal).
    """
    p = PRODUCTEN[bak_id]
    stuk, stuk_sd, cap = p["stuk_gewicht"], p["stuk_sd"], p["capaciteit"]

    stappen: list[tuple[float, float]] = []
    transienten: list[tuple[float, float, float, str]] = []
    vraag_t, vraag_g = [], []
    bijvullingen = []

    gewicht = cap * rng.uniform(0.9, 1.0)
    start_gewicht = gewicht

    def stuk_gewicht():
        return max(stuk * 0.5, rng.normal(stuk, stuk_sd))

    def vul_bij(t0: float, doel_frac: float, soort: str) -> float:
        """Personeel vult de bak stapsgewijs bij. Geeft eindtijd terug."""
        nonlocal gewicht
        t = t0
        # soms eerst rotte/beschadigde stuks verwijderen
        if gewicht > stuk and rng.random() < 0.15:
            weg = min(gewicht, rng.uniform(1, 4) * stuk)
            gewicht -= weg
            stappen.append((t, -weg))
            t += rng.uniform(10, 30)
        doel = cap * doel_frac
        while gewicht < doel - 300:
            schep = min(doel - gewicht, rng.uniform(1800, 4500))  # een krat/schep
            # krat rust even op de rand -> tijdelijk extra gewicht
            transienten.append((t - rng.uniform(3, 8), t + 2, rng.uniform(400, 1500), "krat"))
            gewicht += schep
            stappen.append((t, schep))
            t += rng.uniform(15, 45)
        bijvullingen.append((t0, t))
        return t

    # ochtendvullingen + willekeurige "leun"-momenten per dag
    dag = start.replace(hour=0, minute=0, second=0, microsecond=0)
    dag_events = []  # (tijd, soort)
    while dag < eind:
        if dag.weekday() not in GESLOTEN_WEEKDAGEN:
            ochtend = dag.replace(hour=OCHTENDVULLING[0], minute=OCHTENDVULLING[1])
            dag_events.append((_ts(ochtend) + rng.uniform(0, 900), "ochtend"))
            n_leun = rng.poisson(2.5)
            for _ in range(n_leun):
                tl = _ts(dag.replace(hour=OPENING_UUR)) + rng.uniform(0, (SLUITING_UUR - OPENING_UUR) * 3600)
                dag_events.append((tl, "leun"))
        dag += timedelta(days=1)
    dag_events.sort()

    klanten = sorted(klanten, key=lambda k: k.tijd)
    gepland_bijvullen = None   # tijdstip
    bezig_tot = -1.0           # personeel bezig tot
    drempel = rng.uniform(0.12, 0.30)

    def check_bijvullen(t_nu: float):
        nonlocal gepland_bijvullen, drempel
        if gepland_bijvullen is None and gewicht < cap * drempel and t_nu > bezig_tot:
            # reactietijd personeel: meestal 5-25 min, soms veel later (druk)
            reactie = rng.lognormal(math.log(14 * 60), 0.7)
            reactie = min(reactie, 90 * 60)
            gepland_bijvullen = t_nu + reactie
            drempel = rng.uniform(0.12, 0.30)

    ei, ki = 0, 0
    while ki < len(klanten) or ei < len(dag_events) or gepland_bijvullen is not None:
        t_k = klanten[ki].tijd if ki < len(klanten) else math.inf
        t_e = dag_events[ei][0] if ei < len(dag_events) else math.inf
        t_b = gepland_bijvullen if gepland_bijvullen is not None else math.inf
        t_min = min(t_k, t_e, t_b)
        if t_min == math.inf:
            break

        if t_min == t_b:
            gepland_bijvullen = None
            if is_open(naar_datetime(t_b)):
                bezig_tot = vul_bij(t_b, rng.uniform(0.85, 1.0), "bijvullen")
            # buiten openingsuren: ochtendvulling lost het op
            continue

        if t_min == t_e:
            t, soort = dag_events[ei]
            ei += 1
            if soort == "ochtend":
                gepland_bijvullen = None
                bezig_tot = vul_bij(t, rng.uniform(0.92, 1.0), "ochtend")
            else:  # iemand leunt op de bak of zet er een mand op
                duur = rng.uniform(3, 15)
                transienten.append((t, t + duur, rng.uniform(800, 3000), "leun"))
            continue

        # ---- klant ----
        k = klanten[ki]
        ki += 1
        if k.aantal <= 0:
            continue
        vraag_t.append(k.tijd)
        vraag_g.append(k.aantal * stuk)

        t = k.tijd
        if p["tros"]:
            # bananen: per tros (3-7 stuks) in één keer
            rest = k.aantal
            grepen = []
            while rest > 0:
                n = min(rest, int(rng.integers(3, 8)))
                grepen.append(n)
                rest -= n
        else:
            grepen = [1] * k.aantal

        t_start = t
        for n in grepen:
            t += rng.uniform(1.5, 4.0) if not p["tros"] else rng.uniform(4, 10)
            w = sum(stuk_gewicht() for _ in range(n))
            if gewicht <= stuk * 0.4:
                break  # bak leeg: klant vindt niets meer
            w = min(w, gewicht)
            gewicht -= w
            stappen.append((t, -w))
        # hand die in de bak duwt / producten verschuift
        transienten.append((t_start, t + 1.0, rng.uniform(30, 300), "grijp"))

        # klant legt soms een stuk terug en neemt een ander
        if not p["tros"] and rng.random() < 0.08 and gewicht > stuk:
            terug = stuk_gewicht()
            t += rng.uniform(2, 6)
            stappen.append((t, terug))
            t += rng.uniform(1.5, 4)
            stappen.append((t, -terug))

        check_bijvullen(t)

    stappen.sort()
    st = np.array([s[0] for s in stappen]) if stappen else np.zeros(0)
    sd = np.array([s[1] for s in stappen]) if stappen else np.zeros(0)
    return BakSimulatie(
        bak_id=bak_id, stappen_t=st, stappen_delta=sd, start_gewicht=start_gewicht,
        transienten=transienten, vraag_t=np.array(vraag_t), vraag_gram=np.array(vraag_g),
        bijvullingen=bijvullingen,
    )


def meet(sim: BakSimulatie, t: np.ndarray, rng: np.random.Generator,
         uitval_kans: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Zet het echte gewicht om naar wat de load cell zou meten.

    Geeft (tijdstippen, metingen) terug; sommige metingen vallen weg (WiFi).
    """
    echt = sim.echt_gewicht(t)
    meting = echt.copy()

    # tijdelijke extra krachten (hand, leunen, krat op de rand)
    for (a, b, amp, soort) in sim.transienten:
        i0 = np.searchsorted(t, a, side="left")
        i1 = np.searchsorted(t, b, side="right")
        if i1 <= i0:
            continue
        n = i1 - i0
        if soort == "grijp":
            meting[i0:i1] += amp * rng.uniform(0, 1, n)
        elif soort == "leun":
            meting[i0:i1] += amp * rng.uniform(0.8, 1.1, n)
        else:  # krat
            meting[i0:i1] += amp * rng.uniform(0.3, 1.0, n)

    # sensorruis + trage drift (temperatuur) + kwantisatie op 1 g
    dag_fase = (t % 86400) / 86400
    meting += DRIFT_AMPLITUDE * np.sin(2 * np.pi * dag_fase)
    meting += rng.normal(0, SENSOR_RUIS_SD, len(t))
    meting = np.round(np.maximum(meting, -3))  # lege bak kan net onder 0 zakken

    houd = rng.random(len(t)) >= uitval_kans
    return t[houd], meting[houd]


def meet_tijdstippen(start: datetime, eind: datetime, interval_s: float,
                     van_uur: float = 7.0, tot_uur: float = 20.5) -> np.ndarray:
    """Meettijdstippen per dag tussen van_uur en tot_uur (nachten worden overgeslagen)."""
    stukken = []
    dag = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while dag < eind:
        if dag.weekday() not in GESLOTEN_WEEKDAGEN:
            a = max(_ts(dag) + van_uur * 3600, _ts(start))
            b = min(_ts(dag) + tot_uur * 3600, _ts(eind))
            if b > a:
                stukken.append(np.arange(a, b, interval_s))
        dag += timedelta(days=1)
    return np.concatenate(stukken) if stukken else np.zeros(0)


def kassa_naar_klanten(kassa_tijden, aantallen, rng: np.random.Generator) -> list[Klant]:
    """Een klant staat een paar minuten vóór het afrekenen aan de bak."""
    klanten = []
    for kt, n in zip(kassa_tijden, aantallen):
        t = _ts(kt) - rng.uniform(2, 20) * 60
        d = naar_datetime(t)
        if d.hour < OPENING_UUR:
            t = _ts(d.replace(hour=OPENING_UUR, minute=0, second=0)) + rng.uniform(0, 120)
        klanten.append(Klant(tijd=t, aantal=int(n)))
    return klanten
