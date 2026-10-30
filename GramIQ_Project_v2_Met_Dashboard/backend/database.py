"""
GramIQ database (SQLite).

Eén bestand (standaard data/gramiq.db) met vier tabellen:

    producten       wat er in een bak ligt (stukgewicht, capaciteit van een volle bak)
    weegschalen     elke fysieke weegschaal (id = bak_id uit het MQTT-topic),
                    gekoppeld aan één product, met eigen tarra
    metingen        elke ontvangen meting: ruw gewicht + gefilterd gewicht, % vol, stuks
    voorspellingen  één AI-voorspelling per weegschaal per minuut

Waarom SQLite: zit standaard in Python, geen extra server op de winkel-PC,
één bestand dat je kan back-uppen. Met WAL-modus kan de backend tegelijk
schrijven (MQTT-thread) en lezen (API).

Schrijven gebeurt in een aparte thread die metingen bundelt (max. ~1 s
vertraging), zodat de MQTT-verwerking nooit op de schijf moet wachten.
"""
from __future__ import annotations

import os
import queue
import sqlite3
import threading
import time
from contextlib import contextmanager

SCHEMA_VERSIE = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS producten (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    naam               TEXT    NOT NULL UNIQUE,
    stuk_gewicht_gram  REAL    NOT NULL CHECK (stuk_gewicht_gram > 0),
    stuk_sd_gram       REAL    NOT NULL DEFAULT 0,
    capaciteit_gram    REAL    NOT NULL CHECK (capaciteit_gram > 0),
    per_tros           INTEGER NOT NULL DEFAULT 0,
    ai_code            INTEGER          -- productcode in het AI-model (NULL = niet getraind)
);

CREATE TABLE IF NOT EXISTS weegschalen (
    id             TEXT    PRIMARY KEY,          -- bv. '07' (laatste deel van het MQTT-topic)
    naam           TEXT,
    product_id     INTEGER REFERENCES producten(id) ON DELETE SET NULL,
    tarra_gram     REAL    NOT NULL DEFAULT 0,   -- wordt van het ontvangen gewicht afgetrokken
    actief         INTEGER NOT NULL DEFAULT 1,
    aangemaakt_op  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    laatst_gezien  TEXT                          -- tijdstip van de laatste meting
);

CREATE TABLE IF NOT EXISTS metingen (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    weegschaal_id           TEXT    NOT NULL REFERENCES weegschalen(id) ON DELETE CASCADE,
    product_id              INTEGER REFERENCES producten(id) ON DELETE SET NULL,
    tijdstip                TEXT    NOT NULL,    -- 'YYYY-MM-DD HH:MM:SS' volgens de weegschaal
    gewicht_ruw_gram        REAL    NOT NULL,    -- zoals ontvangen
    gewicht_gefilterd_gram  REAL,                -- na tarra + mediaanfilter
    pct_vol                 REAL,
    stuks                   INTEGER,
    ontvangen_op            TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    UNIQUE (weegschaal_id, tijdstip)
);
CREATE INDEX IF NOT EXISTS idx_metingen_weegschaal_tijd ON metingen (weegschaal_id, tijdstip);

CREATE TABLE IF NOT EXISTS voorspellingen (
    weegschaal_id       TEXT NOT NULL REFERENCES weegschalen(id) ON DELETE CASCADE,
    tijdstip            TEXT NOT NULL,           -- minuut: 'YYYY-MM-DD HH:MM'
    minuten             REAL,                    -- leeg over ... open minuten
    minuten_min         REAL,
    minuten_max         REAL,
    leeg_om             TEXT,
    methode             TEXT,
    verbruik_g_per_min  REAL,
    PRIMARY KEY (weegschaal_id, tijdstip)
);

CREATE TABLE IF NOT EXISTS schema_info (
    sleutel TEXT PRIMARY KEY,
    waarde  TEXT
);
"""


def standaard_pad() -> str:
    hier = os.path.dirname(os.path.abspath(__file__))
    return os.environ.get("GRAMIQ_DB", os.path.join(hier, "..", "data", "gramiq.db"))


class Database:
    def __init__(self, pad: str | None = None, flush_interval: float = 1.0):
        self.pad = os.path.abspath(pad or standaard_pad())
        os.makedirs(os.path.dirname(self.pad), exist_ok=True)
        self.flush_interval = flush_interval
        self._wachtrij: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        with self.verbinding() as con:
            con.executescript(SCHEMA)
            con.execute("INSERT OR REPLACE INTO schema_info VALUES ('versie', ?)", (str(SCHEMA_VERSIE),))
        self._schrijver = threading.Thread(target=self._schrijf_lus, name="db-schrijver", daemon=True)
        self._schrijver.start()

    # ------------------------------------------------------------ verbinding
    def _open(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.pad, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode = WAL")
        con.execute("PRAGMA synchronous = NORMAL")
        con.execute("PRAGMA foreign_keys = ON")
        return con

    @contextmanager
    def verbinding(self):
        """Korte verbinding voor lezen of een losse schrijfactie (commit bij succes)."""
        con = self._open()
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # -------------------------------------------------------- startgegevens
    def vul_standaard(self, producten: dict, product_codes: dict):
        """Eerste start: producten en weegschalen uit de oude vaste configuratie overnemen.

        `producten` = weegschaal_model.PRODUCTEN ({bak_id: {...}}). Gebeurt enkel
        als de tabel producten nog leeg is; daarna is de database de bron.
        """
        with self.verbinding() as con:
            if con.execute("SELECT COUNT(*) FROM producten").fetchone()[0]:
                return False
            for bak_id, p in producten.items():
                cur = con.execute(
                    "INSERT INTO producten (naam, stuk_gewicht_gram, stuk_sd_gram, capaciteit_gram, per_tros, ai_code) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (p["naam"], p["stuk_gewicht"], p.get("stuk_sd", 0), p["capaciteit"],
                     int(bool(p.get("tros"))), product_codes.get(bak_id)),
                )
                con.execute(
                    "INSERT OR IGNORE INTO weegschalen (id, naam, product_id) VALUES (?, ?, ?)",
                    (bak_id, f"Bak {bak_id}", cur.lastrowid),
                )
            return True

    # ---------------------------------------------------------- configuratie
    def weegschaal_config(self) -> dict:
        """{weegschaal_id: {product + tarra}} voor alle actieve weegschalen."""
        with self.verbinding() as con:
            rijen = con.execute(
                """SELECT w.id AS weegschaal_id, w.naam AS weegschaal_naam, w.tarra_gram, w.actief,
                          p.id AS product_id, p.naam, p.stuk_gewicht_gram, p.stuk_sd_gram,
                          p.capaciteit_gram, p.per_tros, p.ai_code
                   FROM weegschalen w LEFT JOIN producten p ON p.id = w.product_id"""
            ).fetchall()
        return {r["weegschaal_id"]: dict(r) for r in rijen}

    def registreer_onbekende_weegschaal(self, weegschaal_id: str, tijdstip: str):
        """Een nieuwe weegschaal die zich meldt, komt in de lijst (nog zonder product)."""
        with self.verbinding() as con:
            con.execute(
                "INSERT OR IGNORE INTO weegschalen (id, naam, laatst_gezien) VALUES (?, ?, ?)",
                (weegschaal_id, f"Bak {weegschaal_id}", tijdstip),
            )

    def producten(self) -> list[dict]:
        with self.verbinding() as con:
            return [dict(r) for r in con.execute("SELECT * FROM producten ORDER BY id")]

    def weegschalen(self) -> list[dict]:
        with self.verbinding() as con:
            return [dict(r) for r in con.execute(
                """SELECT w.*, p.naam AS product_naam,
                          (SELECT COUNT(*) FROM metingen m WHERE m.weegschaal_id = w.id) AS aantal_metingen
                   FROM weegschalen w LEFT JOIN producten p ON p.id = w.product_id
                   ORDER BY w.id""")]

    # -------------------------------------------------------------- schrijven
    def bewaar_meting(self, weegschaal_id, product_id, tijdstip, ruw, gefilterd, pct_vol, stuks):
        self._wachtrij.put(("meting", (weegschaal_id, product_id, tijdstip, ruw, gefilterd, pct_vol, stuks)))

    def bewaar_voorspelling(self, weegschaal_id, minuut, v: dict):
        self._wachtrij.put(("voorspelling", (
            weegschaal_id, minuut, v["minuten"], v["minuten_min"], v["minuten_max"],
            v.get("leeg_om"), v.get("methode"), v.get("verbruik_g_per_min"),
        )))

    def _schrijf_lus(self):
        con = self._open()
        try:
            while not self._stop.is_set() or not self._wachtrij.empty():
                try:
                    eerste = self._wachtrij.get(timeout=self.flush_interval)
                except queue.Empty:
                    continue
                items = [eerste]
                deadline = time.monotonic() + self.flush_interval
                while len(items) < 5000 and time.monotonic() < deadline:
                    try:
                        items.append(self._wachtrij.get(timeout=max(0.0, deadline - time.monotonic())))
                    except queue.Empty:
                        break
                self._schrijf(con, items)
                for _ in items:
                    self._wachtrij.task_done()
        finally:
            con.close()

    @staticmethod
    def _schrijf(con, items):
        metingen = [d for (soort, d) in items if soort == "meting"]
        voorsp = [d for (soort, d) in items if soort == "voorspelling"]
        try:
            if metingen:
                con.executemany(
                    """INSERT INTO metingen (weegschaal_id, product_id, tijdstip, gewicht_ruw_gram,
                                             gewicht_gefilterd_gram, pct_vol, stuks)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT (weegschaal_id, tijdstip) DO UPDATE SET
                           product_id = excluded.product_id,
                           gewicht_ruw_gram = excluded.gewicht_ruw_gram,
                           gewicht_gefilterd_gram = excluded.gewicht_gefilterd_gram,
                           pct_vol = excluded.pct_vol,
                           stuks = excluded.stuks,
                           ontvangen_op = datetime('now', 'localtime')""",
                    metingen,
                )
                laatst = {}
                for d in metingen:
                    laatst[d[0]] = max(laatst.get(d[0], ""), d[2])
                con.executemany(
                    "UPDATE weegschalen SET laatst_gezien = ? WHERE id = ?",
                    [(ts, ws) for ws, ts in laatst.items()],
                )
            if voorsp:
                con.executemany(
                    "INSERT OR REPLACE INTO voorspellingen VALUES (?, ?, ?, ?, ?, ?, ?, ?)", voorsp)
            con.commit()
        except sqlite3.Error as e:
            con.rollback()
            print(f"!! Databasefout, {len(items)} records niet bewaard: {e}")

    def wacht_tot_geschreven(self, timeout: float = 10.0) -> bool:
        """Blokkeert tot alles in de wachtrij op schijf staat (handig bij afsluiten en in tests)."""
        einde = time.monotonic() + timeout
        while self._wachtrij.unfinished_tasks:
            if time.monotonic() > einde:
                return False
            time.sleep(0.02)
        return True

    def sluit(self):
        self._stop.set()
        self._schrijver.join(timeout=10)

    # ---------------------------------------------------------------- lezen
    def laatste_dag_metingen(self, weegschaal_id: str) -> list[sqlite3.Row]:
        """Alle metingen van de dag van de laatste meting (om de filter/verbruik te herstellen)."""
        with self.verbinding() as con:
            laatste = con.execute(
                "SELECT MAX(tijdstip) FROM metingen WHERE weegschaal_id = ?", (weegschaal_id,)
            ).fetchone()[0]
            if laatste is None:
                return []
            return con.execute(
                """SELECT tijdstip, gewicht_ruw_gram FROM metingen
                   WHERE weegschaal_id = ? AND tijdstip >= ? AND tijdstip <= ?
                   ORDER BY tijdstip""",
                (weegschaal_id, laatste[:10] + " 00:00:00", laatste),
            ).fetchall()

    def laatste_voorspelling(self, weegschaal_id: str):
        with self.verbinding() as con:
            r = con.execute(
                "SELECT * FROM voorspellingen WHERE weegschaal_id = ? ORDER BY tijdstip DESC LIMIT 1",
                (weegschaal_id,),
            ).fetchone()
        return dict(r) if r else None

    def historiek_per_minuut(self, weegschaal_id: str, van: str, tot: str) -> list[dict]:
        """Laatste gefilterde gewicht per minuut + de voorspelling van die minuut."""
        with self.verbinding() as con:
            rijen = con.execute(
                """WITH per_minuut AS (
                       SELECT substr(tijdstip, 1, 16) AS minuut, MAX(tijdstip) AS laatste
                       FROM metingen
                       WHERE weegschaal_id = ? AND tijdstip BETWEEN ? AND ?
                       GROUP BY minuut)
                   SELECT pm.minuut, m.gewicht_gefilterd_gram AS gewicht, v.minuten AS voorspelling_min
                   FROM per_minuut pm
                   JOIN metingen m ON m.weegschaal_id = ? AND m.tijdstip = pm.laatste
                   LEFT JOIN voorspellingen v ON v.weegschaal_id = ? AND v.tijdstip = pm.minuut
                   ORDER BY pm.minuut""",
                (weegschaal_id, van, tot, weegschaal_id, weegschaal_id),
            ).fetchall()
        return [{"tijd": r["minuut"][11:16], "gewicht": round(r["gewicht"]) if r["gewicht"] is not None else None,
                 "voorspelling_min": r["voorspelling_min"]} for r in rijen]

    def metingen(self, weegschaal_id: str, van: str | None, tot: str | None, limiet: int) -> list[dict]:
        sql = "SELECT * FROM metingen WHERE weegschaal_id = ?"
        args: list = [weegschaal_id]
        if van:
            sql += " AND tijdstip >= ?"
            args.append(van)
        if tot:
            sql += " AND tijdstip <= ?"
            args.append(tot)
        sql += " ORDER BY tijdstip DESC LIMIT ?"
        args.append(limiet)
        with self.verbinding() as con:
            return [dict(r) for r in con.execute(sql, args)][::-1]

    def statistiek(self) -> dict:
        with self.verbinding() as con:
            tel = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                   for t in ("producten", "weegschalen", "metingen", "voorspellingen")}
            eerste, laatste = con.execute("SELECT MIN(tijdstip), MAX(tijdstip) FROM metingen").fetchone()
        grootte = sum(os.path.getsize(self.pad + s) for s in ("", "-wal") if os.path.exists(self.pad + s))
        return {"pad": self.pad, "grootte_mb": round(grootte / 1e6, 2), "aantallen": tel,
                "eerste_meting": eerste, "laatste_meting": laatste,
                "wachtrij": self._wachtrij.qsize()}
