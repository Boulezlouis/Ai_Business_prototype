"""
Gedeelde code voor ruisfiltering en feature-berekening.

Wordt gebruikt bij het trainen (ai/genereer_dataset.py) EN live in de backend,
zodat het model in productie exact dezelfde features ziet als tijdens training.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from statistics import median

import numpy as np

from weegschaal_model import (GESLOTEN_WEEKDAGEN, OPENING_UUR, PRODUCT_CODE,
                              PRODUCTEN, SLUITING_UUR, naar_datetime, _ts)

MEDIAAN_VENSTER = 7          # aantal ruwe metingen in de mediaanfilter
MIN_AFNAME_GRAM = 50         # kleinere dalingen per minuut = ruis/drift
MAX_HORIZON_MIN = 600        # voorspellingen worden hierop afgekapt
VENSTERS = (10, 30, 60, 120)  # minuten voor verbruikssnelheid

FEATURE_NAMEN = [
    "product_code", "stuk_gewicht", "gewicht", "pct_vol", "stuks",
    "verbruik_10", "verbruik_30", "verbruik_60", "verbruik_120", "verbruik_vandaag",
    "uur", "weekdag", "min_tot_sluiting", "min_sinds_bijvulling", "baseline_min",
    # vraagprofiel (geleerd uit de echte verkopen van de trainingsperiode)
    "verwacht_60", "verwacht_120", "verwacht_rest_vandaag", "drukte", "profiel_min",
]
PROFIEL_FEATURES = FEATURE_NAMEN[-5:]

OPEN_MIN_PER_DAG = (SLUITING_UUR - OPENING_UUR) * 60


# --------------------------------------------------------------------------
# Foutmarge rond de voorspelling (geleerd in train_model.py)
# --------------------------------------------------------------------------
def pas_marges_toe(pred, marges):
    """Onder- en bovengrens rond een (array van) voorspelling(en)."""
    pred = np.asarray(pred, dtype=float)
    idx = np.clip(np.searchsorted(marges["grenzen"], pred, side="right") - 1, 0, len(marges["factoren"]) - 1)
    f = np.array(marges["factoren"])[idx]
    lo = np.clip((pred + 1) * f[:, 0] - 1, 0, None)
    hi = (pred + 1) * f[:, 1] - 1
    return np.minimum(lo, pred), np.maximum(hi, pred)


# --------------------------------------------------------------------------
# Openingsuren-rekenwerk ("open minuten" = minuten dat de winkel open is)
# --------------------------------------------------------------------------
def _open_minuten_op_dag(d: datetime) -> float:
    """Open minuten verstreken op dag d tot tijdstip d."""
    if d.weekday() in GESLOTEN_WEEKDAGEN:
        return 0.0
    m = d.hour * 60 + d.minute + d.second / 60 - OPENING_UUR * 60
    return min(max(m, 0.0), OPEN_MIN_PER_DAG)


def open_minuten_tussen(a: float, b: float) -> float:
    """Aantal open minuten tussen twee tijdstippen (seconden)."""
    da, db = naar_datetime(a), naar_datetime(b)
    if da.date() == db.date():
        return _open_minuten_op_dag(db) - _open_minuten_op_dag(da)
    tot = OPEN_MIN_PER_DAG * (0 if da.weekday() in GESLOTEN_WEEKDAGEN else 1) - _open_minuten_op_dag(da)
    d = (da + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    while d.date() < db.date():
        if d.weekday() not in GESLOTEN_WEEKDAGEN:
            tot += OPEN_MIN_PER_DAG
        d += timedelta(days=1)
    return tot + _open_minuten_op_dag(db)


def tel_open_minuten_op(t: float, minuten: float) -> datetime:
    """Tijdstip t + `minuten` open minuten (sluitingsuren en zondag overgeslagen)."""
    d = naar_datetime(t)
    rest = minuten
    while True:
        if d.weekday() not in GESLOTEN_WEEKDAGEN:
            sluit = d.replace(hour=SLUITING_UUR, minute=0, second=0, microsecond=0)
            opent = d.replace(hour=OPENING_UUR, minute=0, second=0, microsecond=0)
            if d < opent:
                d = opent
            if d < sluit:
                beschikbaar = (sluit - d).total_seconds() / 60
                if rest <= beschikbaar:
                    return d + timedelta(minutes=rest)
                rest -= beschikbaar
        d = (d + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


# --------------------------------------------------------------------------
# Vraagprofiel: verwachte verkoop (gram per open minuut) per bak en weekdag
# --------------------------------------------------------------------------
def standaard_profielpad() -> str:
    import os
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "vraagprofiel.json")


def laad_vraagprofiel(pad: str | None = None) -> dict:
    """{bak_id: np.array (7 weekdagen x OPEN_MIN_PER_DAG)} of {} als het bestand ontbreekt."""
    import json
    import os
    pad = pad or standaard_profielpad()
    if not os.path.exists(pad):
        return {}
    with open(pad) as f:
        ruw = json.load(f)
    return {bak: np.asarray(v["gram_per_min"], dtype=float) for bak, v in ruw.items()}


class Vraagprofiel:
    """Rekent met het gemiddelde verkooppatroon van één bak."""

    def __init__(self, gram_per_min: np.ndarray):
        self.g = gram_per_min                         # (7, OPEN_MIN_PER_DAG)
        self.cum = np.cumsum(self.g, axis=1)

    def komende(self, d: datetime, horizon: int = MAX_HORIZON_MIN) -> np.ndarray:
        """Verwachte verkoop per open minuut vanaf moment d (over sluitingsuren heen)."""
        m = int(_open_minuten_op_dag(d))
        delen = [self.g[d.weekday(), m:]] if d.weekday() not in GESLOTEN_WEEKDAGEN else []
        lengte = sum(len(x) for x in delen)
        dag = d
        while lengte < horizon:
            dag = dag + timedelta(days=1)
            if dag.weekday() in GESLOTEN_WEEKDAGEN:
                continue
            delen.append(self.g[dag.weekday()])
            lengte += OPEN_MIN_PER_DAG
        return np.concatenate(delen)[:horizon]

    def tot_nu_vandaag(self, d: datetime) -> float:
        m = int(_open_minuten_op_dag(d))
        return float(self.cum[d.weekday(), m - 1]) if m > 0 else 0.0


# --------------------------------------------------------------------------
# Online verwerking van metingen voor één bak
# --------------------------------------------------------------------------
class BakHistoriek:
    """Houdt de recente metingen van één bak bij en berekent features.

    * mediaanfilter over de laatste 7 metingen verwijdert kortstondige pieken
      (hand in de bak, iemand die leunt, een krat op de rand);
    * een reeks per minuut wordt gebruikt om het verbruik te berekenen:
      enkel DALINGEN tellen (bijvullen telt niet als 'negatief verbruik').
    """

    def __init__(self, bak_id: str, product: dict | None = None, product_code: int | None = None,
                 profiel: np.ndarray | None = None):
        """`product` (stuk_gewicht, capaciteit) en `product_code` komen live uit de
        database; zonder argumenten (training) uit de vaste PRODUCTEN-lijst.
        `profiel` = verwacht verkooppatroon (zie laad_vraagprofiel); zonder profiel
        worden de profiel-features leeg (NaN) gelaten."""
        self.profiel = Vraagprofiel(profiel) if profiel is not None else None
        self.bak_id = bak_id
        self.p = product if product is not None else PRODUCTEN[bak_id]
        self.product_code = product_code if product is not None else PRODUCT_CODE.get(bak_id)
        self.ruw = deque(maxlen=MEDIAAN_VENSTER)
        self.gefilterd = None
        self.laatste_t = None
        # per minuut: (minuut_t, gefilterd gewicht, afname in die minuut)
        self.minuten = deque(maxlen=max(VENSTERS) + 5)
        self._huidige_minuut = None
        self._vorige_minuut_gewicht = None
        self.laatste_bijvulling_t = None
        self.dag = None
        self.afname_vandaag = 0.0
        self.start_vandaag = None

    # ---------------------------------------------------------------- input
    def voeg_toe(self, t: float, gewicht: float):
        # nieuwe dag -> geschiedenis van gisteren is niet relevant voor verbruik
        dag = naar_datetime(t).date()
        if self.dag != dag:
            self.dag = dag
            self.afname_vandaag = 0.0
            self.start_vandaag = None
            self.minuten.clear()
            self._vorige_minuut_gewicht = None
            self._huidige_minuut = None
            self.ruw.clear()

        self.ruw.append(gewicht)
        self.gefilterd = median(self.ruw)
        self.laatste_t = t

        minuut = int(t // 60)
        if self._huidige_minuut is None:
            self._huidige_minuut = minuut
            self._vorige_minuut_gewicht = self.gefilterd
            return
        if minuut != self._huidige_minuut:
            # sluit de vorige minuut af (ook over hiaten heen)
            g = self.gefilterd
            verschil = self._vorige_minuut_gewicht - g
            afname = verschil if verschil >= MIN_AFNAME_GRAM else 0.0
            if -verschil >= max(1000.0, 0.1 * self.p["capaciteit"]):
                self.laatste_bijvulling_t = t
            if self._is_open(t):
                if self.start_vandaag is None:
                    self.start_vandaag = t
                self.afname_vandaag += afname
            self.minuten.append((minuut, g, afname))
            self._vorige_minuut_gewicht = g
            self._huidige_minuut = minuut

    @staticmethod
    def _is_open(t: float) -> bool:
        d = naar_datetime(t)
        return d.weekday() not in GESLOTEN_WEEKDAGEN and OPENING_UUR <= d.hour < SLUITING_UUR

    # ------------------------------------------------------------- features
    def verbruik(self, venster_min: int) -> float:
        """Gemiddelde afname in gram per minuut over het venster."""
        if not self.minuten:
            return 0.0
        nu = self.minuten[-1][0]
        som = sum(a for (m, _, a) in self.minuten if m > nu - venster_min)
        beschikbaar = min(venster_min, len(self.minuten))
        return som / max(beschikbaar, 1)

    def klaar(self) -> bool:
        return self.gefilterd is not None and len(self.minuten) >= 10

    def features(self) -> dict | None:
        if not self.klaar():
            return None
        t = self.laatste_t
        d = naar_datetime(t)
        g = max(self.gefilterd, 0.0)
        cap, stuk = self.p["capaciteit"], self.p["stuk_gewicht"]
        v = {w: self.verbruik(w) for w in VENSTERS}
        open_vandaag = open_minuten_tussen(self.start_vandaag, t) if self.start_vandaag else 0.0
        verbruik_vandaag = self.afname_vandaag / open_vandaag if open_vandaag >= 1 else v[60]
        min_tot_sluiting = max(0.0, (SLUITING_UUR * 60) - (d.hour * 60 + d.minute + d.second / 60))
        sinds_bijv = (t - self.laatste_bijvulling_t) / 60 if self.laatste_bijvulling_t else 999.0
        basis_snelheid = v[60] if v[60] > 0 else verbruik_vandaag
        baseline = min(g / basis_snelheid, MAX_HORIZON_MIN) if basis_snelheid > 0 else MAX_HORIZON_MIN
        f = {
            "product_code": self.product_code,
            "stuk_gewicht": stuk,
            "gewicht": g,
            "pct_vol": 100 * g / cap,
            "stuks": g / stuk,
            "verbruik_10": v[10],
            "verbruik_30": v[30],
            "verbruik_60": v[60],
            "verbruik_120": v[120],
            "verbruik_vandaag": verbruik_vandaag,
            "uur": d.hour + d.minute / 60,
            "weekdag": d.weekday(),
            "min_tot_sluiting": min_tot_sluiting,
            "min_sinds_bijvulling": min(sinds_bijv, 999.0),
            "baseline_min": baseline,
        }
        f.update(self._profiel_features(d, g))
        return f

    def _profiel_features(self, d: datetime, g: float) -> dict:
        if self.profiel is None:
            return {k: float("nan") for k in PROFIEL_FEATURES}
        komend = self.profiel.komende(d)
        verwacht_tot_nu = self.profiel.tot_nu_vandaag(d)
        # drukker of rustiger dan normaal vandaag? (500 g demping voor het begin van de dag)
        drukte = float(np.clip((self.afname_vandaag + 500) / (verwacht_tot_nu + 500), 0.2, 5.0))
        cum = np.cumsum(komend * drukte)
        j = int(np.searchsorted(cum, g))
        rest_vandaag = OPEN_MIN_PER_DAG - int(_open_minuten_op_dag(d))
        return {
            "verwacht_60": float(komend[:60].sum()),
            "verwacht_120": float(komend[:120].sum()),
            "verwacht_rest_vandaag": float(komend[:rest_vandaag].sum()),
            "drukte": drukte,
            "profiel_min": float(min(j, MAX_HORIZON_MIN)),
        }
