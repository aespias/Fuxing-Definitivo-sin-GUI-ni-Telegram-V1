# ================================================================
# FUXING v3 MAXIMUM — BONOLOTO AI RESEARCH LAB
# ================================================================
# Versión FINAL corregida (revisión cruzada DeepSeek + Kimi K3 + fixes).
#
# Cambios respecto a la versión anterior:
#   1.  parse_fecha_es: soporta "lunes, 4 de marzo de 2024" y variantes.
#   2.  Conformal: misma escala en fit y predict (normalize_rows).
#   3.  scrape_with_cache: fuerza actualización del año en curso.
#   4.  number_features: vectorizado con cum precomputada (pasada como
#       argumento). Ya no reconstruye la matriz 7500 veces.
#   5.  Barra de progreso con ETA estimado.
#   6.  Off-by-one: features incluyen el sorteo más reciente (index+1).
#   7.  Limpieza de código muerto y funciones no usadas.
#   8.  normalize_rows: garantía matemática de suma 6 y valores en [0,1].
#   9.  _compute_components: alineación robusta y cache de resultados.
#  10.  Bonoloto confirmado: 7 sorteos/semana (365/año, 366 bisiesto).
#  11.  _exponential_frequency_from_counts: pesos CORREGIDOS (más peso
#       a sorteos recientes) e INCLUYE el sorteo más reciente.
#  12.  _scrape_year_loteriadehoy: elimina el patrón de fecha del bloque
#       antes de extraer números (no borra números válidos del sorteo).
#
# ADVERTENCIA HONESTA:
# La Bonoloto es un sorteo aleatorio. Ningún programa puede predecir
# la combinación ganadora con probabilidad superior al azar.
# ================================================================

from __future__ import annotations

import os
import re
import csv
import json
import math
import time
import pickle
import random
import sqlite3
import warnings
import threading
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------
# Dependencias opcionales
# ----------------------------------------------------------------
TORCH_AVAILABLE = False
SKLEARN_AVAILABLE = False
XGB_AVAILABLE = False
LGBM_AVAILABLE = False
NUMBA_AVAILABLE = False
TQDM_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    torch = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        ExtraTreesClassifier,
        RandomForestClassifier,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGBClassifier = None

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except Exception:
    lgb = None

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    njit = None

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    tqdm = None


# ================================================================
# CONFIGURACIÓN GLOBAL
# ================================================================

MAX_NUM = 49
NUMBERS_PER_DRAW = 6
START_YEAR = 1988
SEQ_LEN = 120

ULTIMATE_SEED = 20260901

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "bonoloto_data"
MODEL_DIR = BASE_DIR / "bonoloto_models"
LOG_DIR = BASE_DIR / "bonoloto_logs"

for d in (DATA_DIR, MODEL_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

SQLITE_FILE = DATA_DIR / "bonoloto.db"
CACHE_FILE = DATA_DIR / "bonoloto_history.pkl"
CSV_FILE = DATA_DIR / "bonoloto_history.csv"
AUDIT_FILE = DATA_DIR / "dataset_audit.json"
BACKTEST_FILE = DATA_DIR / "backtest_results.json"
FEATURE_AUDIT_FILE = DATA_DIR / "feature_audit.json"

RESULT_FILE = BASE_DIR / "fuxing_resultado.json"
RUN_REPORT_FILE = BASE_DIR / "fuxing_run_report.json"
EXEC_LOG_FILE = LOG_DIR / f"fuxing_{datetime.now():%Y%m%d_%H%M%S}.log"

DEVICE = "cpu"
if TORCH_AVAILABLE and torch.cuda.is_available():
    DEVICE = "cuda"

CPU_COUNT = os.cpu_count() or 2
MAX_CPU_WORKERS = max(1, CPU_COUNT - 1)

NEURAL_ENSEMBLE_SIZE = 5
TREE_ESTIMATORS = 600
XGB_ESTIMATORS = 500
LGBM_ESTIMATORS = 700
MC_ITERATIONS = 1_000_000
MC_ITERATIONS_SEMANAL = 2_000_000

FEATURE_WINDOWS = (5, 10, 20, 40, 80, 160, 320, 640)
FEATURE_HALF_LIFE = 60.0

# Bonoloto se celebra los 7 días de la semana (confirmado por LAE).
# En un año completo hay 365 sorteos (366 en bisiesto). Usamos 365
# como referencia para decidir si un año está "completo".
DRAWS_PER_YEAR_REF = 365

MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5,
    "jun": 6, "jul": 7, "ago": 8, "sep": 9, "oct": 10,
    "nov": 11, "dic": 12,
}


# ================================================================
# LOGGING CON BARRA DE PROGRESO Y ETA
# ================================================================

class FuxingLogger:
    LEVELS = {
        "INFO": "", "DATA": "\033[36m", "NETWORK": "\033[35m",
        "MODEL": "\033[34m", "OPTIM": "\033[33m", "PRED": "\033[32m",
        "AUDIT": "\033[37m", "WARNING": "\033[33m", "ERROR": "\033[31m",
        "OK": "\033[92m",
    }
    RESET = "\033[0m"

    def __init__(self, log_file: Path):
        self.start = datetime.now()
        self.log_file = log_file
        self._lock = threading.Lock()
        self._fh = open(self.log_file, "a", encoding="utf-8")
        self._last_progress_len = 0

    def _ts(self):
        return datetime.now().strftime("%H:%M:%S")

    def _elapsed(self):
        return (datetime.now() - self.start).total_seconds()

    def _color(self, level):
        return self.LEVELS.get(level, "")

    def _emit(self, text):
        with self._lock:
            try:
                print(text, flush=True)
            except Exception:
                pass
            try:
                clean = re.sub(r"\033\[[0-9;]*m", "", text)
                self._fh.write(clean + "\n")
                self._fh.flush()
            except Exception:
                pass

    def log(self, msg, level="INFO"):
        c = self._color(level)
        r = self.RESET
        line = f"[{self._ts()}] {c}[{level:7s}]{r} {msg}  \033[90m(t {self._elapsed():6.1f}s)\033[0m"
        self._emit(line)

    def banner(self, text, char="="):
        line = char * 78
        self._emit("")
        self._emit(line)
        self._emit(f"  {text}")
        self._emit(line)

    def progress(self, pct, msg, width=30):
        """Barra de progreso con ETA estimado."""
        pct = max(0.0, min(100.0, float(pct)))
        filled = int(width * pct / 100.0)
        bar = "#" * filled + "." * (width - filled)
        status = "OK" if pct >= 99.9 else ("RUN" if pct > 0 else "WAIT")

        elapsed = self._elapsed()
        if 0.1 < pct < 99.9:
            eta_sec = elapsed * (100.0 - pct) / pct
            eta_str = f" ETA {int(eta_sec // 60):02d}:{int(eta_sec % 60):02d}"
        elif pct >= 99.9:
            eta_str = " DONE"
        else:
            eta_str = ""

        text = (
            f"[{self._ts()}] \033[96m[{status}]\033[0m "
            f"\033[90m[{bar}]\033[0m {pct:6.2f}%{eta_str}  {msg}"
        )
        with self._lock:
            try:
                clean_len = len(re.sub(r"\033\[[0-9;]*m", "", text))
                pad = " " * max(0, self._last_progress_len - clean_len)
                print(text + pad, flush=True)
                self._last_progress_len = clean_len
            except Exception:
                pass
            try:
                clean = re.sub(r"\033\[[0-9;]*m", "", text)
                self._fh.write(clean + "\n")
                self._fh.flush()
            except Exception:
                pass

    def ok(self, msg):
        self.log(msg, "OK")

    def warn(self, msg):
        self.log(msg, "WARNING")

    def error(self, msg):
        self.log(msg, "ERROR")

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


log = FuxingLogger(EXEC_LOG_FILE)


def print_system_status():
    log.banner("FUXING v3 MAXIMUM — ESTADO DEL SISTEMA")
    log.log(f"Python: {os.sys.version.split()[0]}", "INFO")
    log.log(f"CPU cores: {CPU_COUNT}  |  workers: {MAX_CPU_WORKERS}", "INFO")
    log.log(f"Dispositivo de cómputo: {DEVICE.upper()}", "INFO")
    log.log(f"PyTorch: {'SI' if TORCH_AVAILABLE else 'NO'}  |  "
            f"Scikit-learn: {'SI' if SKLEARN_AVAILABLE else 'NO'}  |  "
            f"XGBoost: {'SI' if XGB_AVAILABLE else 'NO'}  |  "
            f"LightGBM: {'SI' if LGBM_AVAILABLE else 'NO'}", "INFO")
    log.log(f"Numba: {'SI' if NUMBA_AVAILABLE else 'NO'}  |  "
            f"tqdm: {'SI' if TQDM_AVAILABLE else 'NO'}", "INFO")
    log.log(f"Ensemble neuronal: {NEURAL_ENSEMBLE_SIZE} redes Transformer", "INFO")
    log.log(f"Monte Carlo diario: {MC_ITERATIONS:,} iter  |  "
            f"semanal: {MC_ITERATIONS_SEMANAL:,} iter", "INFO")
    log.log(f"Bonoloto: 7 sorteos/semana (lunes a domingo) — {DRAWS_PER_YEAR_REF} sorteos/año", "INFO")


# ================================================================
# UTILIDADES BÁSICAS
# ================================================================

def parse_fecha_es(texto):
    """Parsea fechas en español con soporte para:
       - "lunes, 4 de marzo de 2024"
       - "4 de marzo de 2024"
       - "lunes 4 marzo 2024"
       - "4 marzo 2024"
    """
    if not texto:
        return None
    try:
        txt = texto.lower().strip()
        txt = re.sub(
            r"^(lun(es)?|mar(tes)?|mi[eé]rcoles|jue(ves)?|vie(rnes)?|"
            r"s[aá]b(ado)?|dom(ingo)?)[,.\s]+",
            "", txt,
        )
        txt = re.sub(r"\s+de\s+", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        partes = txt.split()
        if len(partes) != 3:
            return None
        dia = int(re.sub(r"\D", "", partes[0]))
        mes = MESES.get(partes[1].strip("."))
        anio = int(partes[2])
        if mes is None:
            return None
        return datetime(anio, mes, dia)
    except Exception:
        return None


def validate_draw(draw):
    if draw is None or len(draw) != 6:
        return False
    try:
        nums = [int(x) for x in draw]
    except Exception:
        return False
    return len(set(nums)) == 6 and all(1 <= x <= 49 for x in nums)


def normalize_draw(draw):
    return tuple(sorted(int(x) for x in draw))


def one_hot_draw(draw):
    v = np.zeros(MAX_NUM, dtype=np.float32)
    for n in draw:
        if 1 <= n <= MAX_NUM:
            v[n - 1] = 1.0
    return v


def even_count(draw):
    return sum(x % 2 == 0 for x in draw)


def low_count(draw):
    return sum(x <= 24 for x in draw)


def draw_range(draw):
    return max(draw) - min(draw)


def safe_zscore(x):
    x = np.asarray(x, dtype=float)
    std = np.std(x)
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std


def softmax(x, temperature=1.0):
    x = np.asarray(x, dtype=float)
    temperature = max(float(temperature), 1e-6)
    z = (x - np.max(x)) / temperature
    e = np.exp(np.clip(z, -60, 60))
    return e / (e.sum() + 1e-12)


def normalize_mass(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=1.0 / MAX_NUM, posinf=1.0, neginf=1.0)
    p = np.clip(p, 1e-12, None)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full(MAX_NUM, 1.0 / MAX_NUM, dtype=np.float64)
    return p / s


def normalize_rows(a):
    """Proyecta probabilidades marginales sobre el simplex acotado:
    0 <= p_i <= 1 y sum(p_i) = NUMBERS_PER_DRAW.

    La proyección por bisección evita que el recorte final pierda masa,
    incluso con entradas extremas, sesgadas o no finitas.
    """
    arr = np.asarray(a, dtype=float)
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] < NUMBERS_PER_DRAW:
        raise ValueError("La matriz debe tener al menos NUMBERS_PER_DRAW columnas")
    if arr.shape[1] != MAX_NUM:
        raise ValueError(f"Se esperaban {MAX_NUM} columnas y se recibieron {arr.shape[1]}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    target = float(NUMBERS_PER_DRAW)
    out = np.empty_like(arr, dtype=float)

    for i, row in enumerate(arr):
        # La escala de las puntuaciones no debe alterar el resultado de forma
        # arbitraria: la proyección conserva el orden relativo y acota [0, 1].
        row = np.clip(row, 0.0, None)
        if not np.any(row > 0):
            out[i] = target / MAX_NUM
            continue

        lo = float(np.min(row) - 1.0)
        hi = float(np.max(row))
        for _ in range(80):
            mid = (lo + hi) / 2.0
            total = float(np.clip(row - mid, 0.0, 1.0).sum())
            if total > target:
                lo = mid
            else:
                hi = mid
        projected = np.clip(row - (lo + hi) / 2.0, 0.0, 1.0)

        # Corrección de redondeo residual, repartida sólo entre columnas libres.
        residual = target - float(projected.sum())
        if abs(residual) > 1e-10:
            if residual > 0:
                free = np.flatnonzero(projected < 1.0 - 1e-12)
                for j in free:
                    delta = min(residual, 1.0 - projected[j])
                    projected[j] += delta
                    residual -= delta
                    if residual <= 1e-10:
                        break
            else:
                free = np.flatnonzero(projected > 1e-12)
                for j in free:
                    delta = min(-residual, projected[j])
                    projected[j] -= delta
                    residual += delta
                    if residual >= -1e-10:
                        break
        out[i] = np.clip(projected, 0.0, 1.0)

    if was_1d:
        return out[0]
    return out


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def chi_square(freq):
    total = float(np.sum(freq))
    if total <= 0:
        return 0.0
    expected = total / MAX_NUM
    return float(np.sum((freq - expected) ** 2 / expected))

# ================================================================
# PARTE 2 — SCRAPING MULTI-FUENTE
# ================================================================

class DataScraper:
    """Descarga el histórico completo desde 1988 y lo mantiene localmente."""

    ELGORDO = "https://www.elgordo.com/es/resultados/bonoloto--a%C3%B1o-{year}"
    LOTOCRACK = "https://lotocrack.com/historico-de-resultados/bonoloto/resultados-{year}/"
    LOTERIADEHOY = "https://www.loteriadehoy.com/bonoloto/resultados/{year}/"
    RESULTADOSBONOLOTO = "https://www.resultadosbonoloto.com/bonoloto/{year}/"

    def __init__(self, start_year=START_YEAR, end_year=None):
        self.start_year = int(start_year)
        self.end_year = int(end_year or datetime.now().year)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "es-ES,es;q=0.9",
        })
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(SQLITE_FILE) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS draws (
                    fecha TEXT NOT NULL,
                    n1 INTEGER NOT NULL, n2 INTEGER NOT NULL, n3 INTEGER NOT NULL,
                    n4 INTEGER NOT NULL, n5 INTEGER NOT NULL, n6 INTEGER NOT NULL,
                    source TEXT,
                    PRIMARY KEY (fecha, n1, n2, n3, n4, n5, n6)
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_fecha ON draws(fecha)")
            con.commit()

    def _save_sqlite(self, data):
        if not data:
            return
        rows = []
        for fecha, draw in data:
            rows.append((
                fecha.strftime("%Y-%m-%d"),
                int(draw[0]), int(draw[1]), int(draw[2]),
                int(draw[3]), int(draw[4]), int(draw[5]),
                "merged",
            ))
        # La tabla histórica antigua permite varias combinaciones por fecha.
        # Se elimina primero la fecha y se guarda sólo el registro validado más reciente.
        with sqlite3.connect(SQLITE_FILE) as con:
            for row in rows:
                con.execute("DELETE FROM draws WHERE fecha = ?", (row[0],))
                con.execute(
                    "INSERT INTO draws (fecha,n1,n2,n3,n4,n5,n6,source) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    row,
                )
            con.commit()

    def _load_sqlite(self):
        if not SQLITE_FILE.exists():
            return []
        out = []
        try:
            with sqlite3.connect(SQLITE_FILE) as con:
                cur = con.execute(
                    "SELECT fecha,n1,n2,n3,n4,n5,n6 FROM draws ORDER BY fecha, source"
                )
                grouped = defaultdict(Counter)
                date_objects = {}
                for row in cur:
                    try:
                        fecha = datetime.strptime(row[0], "%Y-%m-%d")
                        draw = tuple(sorted(int(x) for x in row[1:7]))
                    except (TypeError, ValueError):
                        continue
                    if validate_draw(draw):
                        grouped[row[0]][draw] += 1
                        date_objects[row[0]] = fecha
                by_date = {}
                for date_key, votes in grouped.items():
                    ranked = votes.most_common()
                    if not ranked:
                        continue
                    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                        log.warn(f"SQLite: conflicto no resuelto para {date_key}; se omite esa fecha.")
                        continue
                    by_date[date_key] = (date_objects[date_key], list(ranked[0][0]))
                out = sorted(by_date.values(), key=lambda x: x[0])
        except Exception as e:
            log.warn(f"SQLite no utilizable: {e}")
            return []
        return out

    def _request(self, url, attempts=4):
        last = None
        for attempt in range(attempts):
            try:
                r = self.session.get(url, timeout=40)
                if r.status_code == 200 and r.text:
                    return r
                last = RuntimeError(f"HTTP {r.status_code}")
            except Exception as e:
                last = e
            time.sleep(1.5 * (attempt + 1))
        raise last or RuntimeError("Error HTTP")

    def _clean_data(self, data):
        """Deduplica por (fecha, draw) con votación por mayoría entre fuentes."""
        by_fecha_meta = defaultdict(list)

        for row in data or []:
            try:
                fecha, draw = row
                if not isinstance(fecha, datetime):
                    fecha = pd.to_datetime(fecha).to_pydatetime()
                draw = tuple(normalize_draw(draw))
                if validate_draw(draw):
                    by_fecha_meta[fecha.date().isoformat()].append((fecha, draw))
            except Exception:
                continue

        result = {}
        for fecha_iso, rows in by_fecha_meta.items():
            vote_count = Counter(draw for _, draw in rows)
            ranked = vote_count.most_common()
            if not ranked:
                continue
            # No escoger arbitrariamente si dos combinaciones distintas tienen
            # exactamente el mismo apoyo. Se informa y se excluye la fecha dudosa.
            if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                log.warn(f"Histórico: conflicto entre fuentes sin mayoría para {fecha_iso}; fecha excluida.")
                continue
            best_draw, _ = ranked[0]
            fecha = rows[0][0]
            result[fecha_iso] = (fecha, list(best_draw))

        return sorted(result.values(), key=lambda x: x[0])

    def _merge(self, *lists):
        combined = []
        for lst in lists:
            combined.extend(lst or [])
        return self._clean_data(combined)

    # ------------------------------------------------------------
    # Parsers por fuente
    # ------------------------------------------------------------
    def _scrape_year_lotocrack(self, year):
        url = self.LOTOCRACK.format(year=int(year))
        r = self._request(url)
        soup = BeautifulSoup(r.text, "html.parser")
        lines = [
            re.sub(r"\s+", " ", x.strip())
            for x in soup.get_text("\n", strip=True).splitlines()
            if x.strip()
        ]
        months = {
            "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
            "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
            "septiembre": 9, "setiembre": 9, "octubre": 10,
            "noviembre": 11, "diciembre": 12,
        }
        date_re = re.compile(
            r"\b(\d{1,2})\s+de\s+([A-Za-zÁÉÍÓÚáéíóúñÑ]+)\s+de\s+(\d{4})\b",
            re.IGNORECASE,
        )
        combo_re = re.compile(
            r"^\s*(\d{1,2}(?:\s+\d{1,2}){5})\s+C-\d{1,2}\s+R",
            re.IGNORECASE,
        )
        rows = []
        for i, line in enumerate(lines):
            dm = date_re.search(line)
            if not dm:
                continue
            try:
                day = int(dm.group(1))
                month = months.get(dm.group(2).lower())
                y = int(dm.group(3))
                if month is None or y != int(year):
                    continue
                fecha = datetime(y, month, day)
            except Exception:
                continue
            for nxt in lines[i + 1:i + 4]:
                cm = combo_re.match(nxt)
                if not cm:
                    continue
                nums = [int(x) for x in re.findall(r"\d{1,2}", cm.group(1))]
                if validate_draw(nums):
                    rows.append((fecha, sorted(nums)))
                break
        return self._clean_data(rows)

    def _scrape_year_elgordo(self, year):
        url = self.ELGORDO.format(year=year)
        r = self._request(url)
        soup = BeautifulSoup(r.text, "html.parser")
        rows = []
        text_lines = [
            re.sub(r"\s+", " ", x.strip())
            for x in soup.get_text("\n", strip=True).splitlines()
            if x.strip()
        ]
        combo_re = re.compile(
            r"^((?:\d{1,2}-){5}\d{1,2})(?:-C\d{1,2})?(?:\s+-?R\d{1,2})?$"
        )
        for i, line in enumerate(text_lines):
            m = combo_re.match(line)
            if not m:
                continue
            nums = [int(x) for x in m.group(1).split("-")]
            if not validate_draw(nums):
                continue
            fecha = None
            for nxt in text_lines[i + 1:i + 6]:
                fecha = parse_fecha_es(nxt)
                if fecha:
                    break
            if fecha and fecha.year == int(year):
                rows.append((fecha, sorted(nums)))
        for fila in soup.select("div.pseudo-table-row"):
            try:
                fe = fila.select_one(".history-results-date")
                ce = fila.select_one(".history-results-combi")
                if not fe or not ce:
                    continue
                fecha = parse_fecha_es(fe.get_text(" ", strip=True))
                if not fecha:
                    continue
                nums = []
                for sp in ce.select("span"):
                    t = sp.get_text(strip=True)
                    if re.fullmatch(r"\d{1,2}", t):
                        n = int(t)
                        if 1 <= n <= 49:
                            nums.append(n)
                if len(nums) < 6:
                    raw = ce.get_text(" ", strip=True)
                    nums = [
                        int(x) for x in re.findall(r"(?<!\d)\d{1,2}(?!\d)", raw)
                        if 1 <= int(x) <= 49
                    ][:6]
                nums = sorted(set(nums))
                if validate_draw(nums):
                    rows.append((fecha, nums))
            except Exception:
                continue
        return self._clean_data(rows)

    def _scrape_year_loteriadehoy(self, year):
        """Parser de loteriadehoy.

        CORRECCIÓN: en lugar de borrar día/mes/año de la lista de números
        (lo que borraría números ganadores legítimos que coincidan),
        eliminamos el PATRÓN de fecha del texto del bloque antes de
        extraer los números.
        """
        url = self.LOTERIADEHOY.format(year=int(year))
        r = self._request(url)
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text("\n", strip=True)
        lines = [re.sub(r"\s+", " ", x.strip()) for x in text.splitlines() if x.strip()]

        rows = []
        for i, line in enumerate(lines):
            fecha = parse_fecha_es(line)
            if not fecha or fecha.year != int(year):
                m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", line)
                if m:
                    try:
                        fecha = datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)))
                    except Exception:
                        fecha = None
            if not fecha or fecha.year != int(year):
                continue
            block = " ".join(lines[i:i + 4])
            # Eliminar patrones de fecha del bloque ANTES de extraer números
            block_no_date = re.sub(r"\b\d{1,2}/\d{1,2}/\d{4}\b", " ", block)
            block_no_date = re.sub(
                r"\b\d{1,2}\s+de\s+[A-Za-zÁÉÍÓÚáéíóúñÑ]+(?:\s+de\s+\d{4})?\b",
                " ", block_no_date, flags=re.IGNORECASE,
            )
            nums = [
                int(x) for x in re.findall(r"(?<!\d)\d{1,2}(?!\d)", block_no_date)
                if 1 <= int(x) <= 49
            ]
            nums = list(dict.fromkeys(nums))[:6]
            if validate_draw(nums):
                rows.append((fecha, sorted(nums)))
        return self._clean_data(rows)

    def _scrape_year_resultadosbonoloto(self, year):
        url = self.RESULTADOSBONOLOTO.format(year=int(year))
        r = self._request(url)
        soup = BeautifulSoup(r.text, "html.parser")
        rows = []
        date_re = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")

        for fila in soup.select("tr, div.row, li"):
            txt = fila.get_text(" ", strip=True)
            dm = date_re.search(txt)
            if not dm:
                continue
            try:
                day = int(dm.group(1))
                month = int(dm.group(2))
                y = int(dm.group(3))
                if y != int(year):
                    continue
                fecha = datetime(y, month, day)
            except Exception:
                continue

            # Elimina sólo la fecha completa; no borres de nuevo el día/mes,
            # porque esos valores también pueden ser números ganadores.
            txt_no_date = txt[:dm.start()] + " " + txt[dm.end():]

            nums = [
                int(x)
                for x in re.findall(r"(?<!\d)\d{1,2}(?!\d)", txt_no_date)
                if 1 <= int(x) <= 49
            ]
            nums = list(dict.fromkeys(nums))[:6]
            if validate_draw(nums):
                rows.append((fecha, sorted(nums)))

        return self._clean_data(rows)

    def scrape_year(self, year):
        year = int(year)
        results = {}

        def try_source(name, fn):
            try:
                rows = fn(year)
                results[name] = rows
                if rows:
                    log.log(f"Año {year} · {name}: {len(rows)} sorteos válidos", "DATA")
            except Exception as e:
                results[name] = []
                log.warn(f"Año {year} · {name}: {type(e).__name__}: {e}")

        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [
                ex.submit(try_source, "lotocrack", self._scrape_year_lotocrack),
                ex.submit(try_source, "elgordo", self._scrape_year_elgordo),
                ex.submit(try_source, "loteriadehoy", self._scrape_year_loteriadehoy),
                ex.submit(try_source, "resultadosbonoloto", self._scrape_year_resultadosbonoloto),
            ]
            for f in as_completed(futs):
                f.result()

        valid_sources = {}
        for name, rows in results.items():
            if len(rows) >= 50:
                valid_sources[name] = rows
            elif rows:
                log.warn(f"Año {year} · {name}: solo {len(rows)} sorteos, se descarta")

        if not valid_sources:
            log.warn(f"Año {year}: ninguna fuente fiable", "WARNING")
            return []

        tagged_rows = []
        for name, rows in valid_sources.items():
            for fecha, draw in rows:
                tagged_rows.append((fecha, draw))
        merged = self._clean_data(tagged_rows)

        log.log(
            f"Año {year}: fusión final {len(merged)} sorteos "
            f"(fuentes usadas: {list(valid_sources.keys())})",
            "DATA",
        )
        return merged

    def scrape_with_cache(self, force_refresh=False):
        old = []
        if not force_refresh:
            old = self._load_sqlite()
            if old:
                log.log(f"Caché SQLite: {len(old):,} sorteos válidos", "DATA")

        by_year = Counter(d.year for d, _ in old)

        years_to_fetch = []
        today = datetime.now().date()
        for y in range(self.start_year, self.end_year + 1):
            have = by_year.get(y, 0)
            is_recent = y >= self.end_year - 1
            year_start = datetime(y, 1, 1).date()
            year_end = datetime(y, 12, 31).date()
            if y == self.start_year:
                # El histórico de Fuxing comienza el 28/02/1988.
                year_start = max(year_start, datetime(1988, 2, 28).date())
            if y == self.end_year:
                year_end = min(year_end, today)
            expected = (year_end - year_start).days + 1 if year_end >= year_start else 0
            # Sólo se toleran hasta tres huecos en años completos; los años recientes
            # se actualizan siempre porque pueden faltar sorteos recientes.
            if force_refresh or have < max(0, expected - 3) or is_recent:
                years_to_fetch.append(y)

        if not years_to_fetch:
            log.log("Histórico completo en caché, no se descarga nada.", "DATA")
            return old

        log.log(
            f"Descargando {len(years_to_fetch)} años "
            f"({years_to_fetch[0]}-{years_to_fetch[-1]})...",
            "NETWORK",
        )

        fresh = []
        total = len(years_to_fetch)
        done = 0
        with ThreadPoolExecutor(max_workers=min(8, total)) as ex:
            futs = {ex.submit(self.scrape_year, y): y for y in years_to_fetch}
            for fut in as_completed(futs):
                y = futs[fut]
                done += 1
                try:
                    rows = fut.result()
                    fresh.extend(rows)
                except Exception as e:
                    log.warn(f"Año {y}: error final {e}")
                pct = 2.0 + 18.0 * (done / total)
                log.progress(pct, f"Histórico: {done}/{total} años · último={y}")

        merged = self._merge(old, fresh)
        if len(merged) < len(old):
            log.warn("La actualización produciría menos datos; se conserva la caché.")
            merged = old

        self._save_all(merged)
        self._audit_coverage(merged)
        return merged

    def scrape_all_years(self):
        return self.scrape_with_cache(force_refresh=True)

    def _save_all(self, data):
        self._save_sqlite(data)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(data, f)
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["fecha", "n1", "n2", "n3", "n4", "n5", "n6"])
            for fecha, draw in data:
                w.writerow([fecha.strftime("%Y-%m-%d"), *draw])

    def _audit_coverage(self, data):
        if not data:
            return
        years = Counter(d.year for d, _ in data)
        audit = {
            "generated": datetime.now().isoformat(),
            "total_draws": len(data),
            "min_date": data[0][0].strftime("%Y-%m-%d"),
            "max_date": data[-1][0].strftime("%Y-%m-%d"),
            "years": dict(sorted(years.items())),
            "expected_start": self.start_year,
            "expected_end": self.end_year,
        }
        with open(AUDIT_FILE, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, ensure_ascii=False)
        log.log(
            f"Histórico final: {len(data):,} sorteos | "
            f"{data[0][0]:%d/%m/%Y} -> {data[-1][0]:%d/%m/%Y}",
            "AUDIT",
        )


# ================================================================
# AUDITORÍA DEL DATASET
# ================================================================

def dataset_audit(data):
    errors = []
    seen = set()
    for fecha, draw in data:
        key = (fecha.date().isoformat(), tuple(draw))
        if not validate_draw(draw):
            errors.append((str(fecha), list(draw), "invalid_draw"))
        if key in seen:
            errors.append((str(fecha), list(draw), "duplicate"))
        seen.add(key)
    dates = [x[0] for x in data]
    report = {
        "records": len(data),
        "errors": len(errors),
        "duplicates_or_invalid": errors[:100],
        "min_date": min(dates).isoformat() if dates else None,
        "max_date": max(dates).isoformat() if dates else None,
    }
    with open(AUDIT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    log.log(f"Auditoría dataset: {len(data):,} registros, {len(errors)} incidencias", "AUDIT")
    return report


def randomness_audit(data):
    freq = np.zeros(MAX_NUM, dtype=float)
    for _, draw in data:
        for n in draw:
            freq[n - 1] += 1
    chi2 = chi_square(freq)
    ent = entropy(freq / (freq.sum() + 1e-12))
    z = safe_zscore(freq)
    outliers = [i + 1 for i, x in enumerate(z) if abs(x) >= 2.5]
    log.log(
        f"Auditoría aleatoriedad: chi2={chi2:.3f}  entropia={ent:.5f}  "
        f"outliers(|z|>=2.5)={len(outliers)}",
        "AUDIT",
    )
    return {
        "frequency": freq.tolist(),
        "chi_square": chi2,
        "entropy": ent,
        "outliers": outliers,
    }
# ================================================================
# PARTE 3 — FEATURE ENGINEERING CORREGIDO
# ================================================================
# FIX aplicado:
#   - _exponential_frequency_from_counts: pesos CORREGIDOS. Antes daba
#     más peso a los sorteos ANTIGUOS y excluía el más reciente. Ahora
#     da más peso a los recientes e INCLUYE el sorteo i.
#   - number_features / build_dataset: cum y draws se precomputan una
#     vez y se pasan como argumentos. Mismo resultado, menos tiempo.
#   - Eliminadas _pair_excess_from_counts y _window_draws (nunca se
#     llamaban y contenían errores conceptuales).
# ================================================================

from math import comb as _math_comb

PROB_PAIR = _math_comb(47, 4) / _math_comb(49, 6)      # ≈ 0.012755
PROB_TRIPLE = _math_comb(46, 3) / _math_comb(49, 6)    # ≈ 0.0010855


def ljung_box(series, lags=20):
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < lags + 5:
        return 0.0, 1.0
    x = x - x.mean()
    denom = np.sum(x * x) + 1e-12
    q = 0.0
    for k in range(1, lags + 1):
        rk = np.sum(x[k:] * x[:-k]) / denom
        q += (rk * rk) / (n - k)
    q *= n * (n + 2)
    try:
        from math import erf
        df = lags
        z = ((q / df) ** (1.0 / 3.0) - (1 - 2.0 / (9 * df))) / math.sqrt(2.0 / (9 * df))
        p = 0.5 * (1 - erf(z / math.sqrt(2)))
    except Exception:
        p = 1.0
    return float(q), float(p)


# ----------------------------------------------------------------
# Feature Engineering
# ----------------------------------------------------------------

class FeatureEngineeringV3:
    """Feature engineering sin fuga de información.

    Todas las features de un índice i usan SOLO sorteos hasta i (incluido).
    El target es el sorteo i+1.
    """

    HALF_LIFE = FEATURE_HALF_LIFE

    # ------------------------------------------------------------
    @staticmethod
    def _build_prefix_counts(draws):
        """Matriz acumulativa (n+1, 49). cum[k, j] = veces que el número
        j+1 apareció en los primeros k sorteos."""
        n = len(draws)
        cum = np.zeros((n + 1, MAX_NUM), dtype=np.float64)
        for i, d in enumerate(draws):
            cum[i + 1] = cum[i]
            for num in d:
                cum[i + 1, num - 1] += 1.0
        return cum

    # ------------------------------------------------------------
    @staticmethod
    def _last_seen_index(draws, end):
        """Para cada número, índice del último sorteo ≤ end en que apareció
        (o -1 si nunca)."""
        last = np.full(MAX_NUM, -1, dtype=np.int64)
        for i in range(end):
            for num in draws[i]:
                last[num - 1] = i
        return last

    # ------------------------------------------------------------
    # FIX CRÍTICO: pesos corregidos + incluye sorteo más reciente
    # ------------------------------------------------------------
    @staticmethod
    def _exponential_frequency_from_counts(cum, index_p, half_life):
        """Frecuencia exponencial hasta el sorteo index_p - 1 (inclusive).

        index_p = número de sorteos a considerar (usualmente index + 1).

        Pesos: w = exp(-decay * back), donde back = 0 para el sorteo
        más reciente y back = N-1 para el más antiguo. Así el peso
        decrece al alejarnos del presente.
        """
        N = index_p
        if N <= 0:
            return np.full(MAX_NUM, 1.0 / MAX_NUM, dtype=np.float64)
        decay = math.log(2.0) / max(half_life, 1e-6)
        # diffs[j] = one-hot del sorteo j (j = 0..N-1)
        diffs = cum[1:N + 1] - cum[:N]
        # backs[j] = 0 para el más reciente, N-1 para el más antiguo
        backs = np.arange(N - 1, -1, -1, dtype=np.float64)
        w = np.exp(-decay * backs)
        weighted = np.sum(diffs * w[:, None], axis=0)
        s = weighted.sum()
        return weighted / s if s > 0 else np.full(MAX_NUM, 1.0 / MAX_NUM)

    @staticmethod
    def _recency_from_last_seen(last, index):
        """Recencia a partir del último índice visto."""
        gaps = (index - last).astype(np.float64)
        gaps[last < 0] = index + 1.0
        return np.log1p(gaps)

    @staticmethod
    def _pair_excess_direct(rows, n_draws):
        if n_draws == 0:
            return np.zeros(MAX_NUM, dtype=np.float64)
        pair = np.zeros((MAX_NUM, MAX_NUM), dtype=np.float64)
        for d in rows:
            for a, b in combinations(d, 2):
                pair[a - 1, b - 1] += 1.0
                pair[b - 1, a - 1] += 1.0
        expected = PROB_PAIR * n_draws
        excess = (pair - expected) / math.sqrt(expected + 1e-9)
        np.fill_diagonal(excess, 0.0)
        return np.sum(excess, axis=1) / (MAX_NUM - 1)

    @staticmethod
    def _triple_excess_direct(rows, n_draws):
        if n_draws == 0:
            return np.zeros(MAX_NUM, dtype=np.float64)
        triple = defaultdict(int)
        for d in rows:
            for tri in combinations(d, 3):
                triple[tri] += 1
        expected = PROB_TRIPLE * n_draws
        score = np.zeros(MAX_NUM, dtype=np.float64)
        for tri, c in triple.items():
            a, b, cc = tri
            excess = (c - expected) / math.sqrt(expected + 1e-9)
            score[a - 1] += excess
            score[b - 1] += excess
            score[cc - 1] += excess
        return score / 1128.0

    @staticmethod
    def structural_features(draws, end, window=80):
        """Estadísticas estructurales de los `window` sorteos hasta `end`."""
        start = max(0, end - window)
        rows = draws[start:end]
        if not rows:
            return np.zeros(14, dtype=np.float64)

        sums = np.array([sum(d) for d in rows], dtype=float)
        ranges = np.array([draw_range(d) for d in rows], dtype=float)
        evens = np.array([even_count(d) for d in rows], dtype=float)
        lows = np.array([low_count(d) for d in rows], dtype=float)
        consec = np.array([
            sum(d[i + 1] == d[i] + 1 for i in range(5)) for d in rows
        ], dtype=float)

        return np.array([
            float(np.mean(sums)), float(np.std(sums)),
            float(np.mean(ranges)), float(np.std(ranges)),
            float(np.mean(evens)), float(np.std(evens)),
            float(np.mean(lows)), float(np.std(lows)),
            float(np.mean(consec)), float(np.std(consec)),
            float(np.min(sums)), float(np.max(sums)),
            float(np.min(ranges)), float(np.max(ranges)),
        ], dtype=np.float64)

    @staticmethod
    def cyclical_features(fecha):
        doy = fecha.timetuple().tm_yday
        dow = fecha.weekday()
        dom = fecha.day
        month = fecha.month
        return np.array([
            math.sin(2 * math.pi * doy / 365.25),
            math.cos(2 * math.pi * doy / 365.25),
            math.sin(2 * math.pi * dow / 7),
            math.cos(2 * math.pi * dow / 7),
            math.sin(2 * math.pi * dom / 31),
            math.cos(2 * math.pi * dom / 31),
            math.sin(2 * math.pi * month / 12),
            math.cos(2 * math.pi * month / 12),
        ], dtype=np.float64)

    @staticmethod
    def number_features(data, index, cum=None, draws=None, last_seen_prefix=None):
        """Construye el vector de features para el índice dado.

        Usa los sorteos HASTA index (incluido). El target asociado
        es data[index + 1].

        cum y draws son opcionales: si se pasan (precomputados), se
        evita reconstruirlos en cada llamada.
        """
        if draws is None:
            draws = [tuple(d) for _, d in data]
        if cum is None:
            cum = FeatureEngineeringV3._build_prefix_counts(draws)

        fecha = data[index][0]

        exp_freq = FeatureEngineeringV3._exponential_frequency_from_counts(
            cum, index + 1, FeatureEngineeringV3.HALF_LIFE
        )

        hard_freqs = []
        for w in FEATURE_WINDOWS:
            start = max(0, index + 1 - w)
            n_rows = index + 1 - start
            if n_rows <= 0:
                hard_freqs.append(np.full(MAX_NUM, 1.0 / MAX_NUM))
            else:
                counts = cum[index + 1] - cum[start]
                hard_freqs.append(counts / n_rows)

        if last_seen_prefix is not None:
            last = last_seen_prefix[index + 1]
        else:
            last = FeatureEngineeringV3._last_seen_index(draws, index + 1)
        rec = FeatureEngineeringV3._recency_from_last_seen(last, index)

        start_pair = max(0, index + 1 - 500)
        pair_rows = draws[start_pair:index + 1]
        pair_ex = FeatureEngineeringV3._pair_excess_direct(pair_rows, len(pair_rows))
        triple_ex = FeatureEngineeringV3._triple_excess_direct(pair_rows, len(pair_rows))

        per_number = np.stack(
            [exp_freq] + hard_freqs + [rec, pair_ex, triple_ex],
            axis=1,
        )
        flat = per_number.flatten().astype(np.float32)

        struct = FeatureEngineeringV3.structural_features(draws, index + 1)
        cyc = FeatureEngineeringV3.cyclical_features(fecha)

        global_freq = cum[index + 1].copy()
        ent = entropy(global_freq / (global_freq.sum() + 1e-12))
        chi2 = chi_square(global_freq)

        extra = np.array([ent, chi2], dtype=np.float64)

        return np.concatenate([flat, struct, cyc, extra]).astype(np.float32)

    @staticmethod
    def feature_dim():
        per_number = MAX_NUM * (1 + len(FEATURE_WINDOWS) + 1 + 1 + 1)
        global_feats = 14 + 8 + 2
        return per_number + global_feats


# ----------------------------------------------------------------
# Construcción del dataset supervisado
# ----------------------------------------------------------------

def build_dataset(data, seq_len=SEQ_LEN):
    if len(data) <= seq_len + 30:
        raise ValueError(
            f"No hay suficientes sorteos para construir el dataset: {len(data)}"
        )

    draws = [tuple(d) for _, d in data]
    # CORRECCIÓN: cum se precomputa UNA VEZ y se pasa a number_features.
    cum = FeatureEngineeringV3._build_prefix_counts(draws)
    # Última aparición por prefijo: evita recorrer todo el histórico para cada muestra.
    last_seen_prefix = np.full((len(draws) + 1, MAX_NUM), -1, dtype=np.int32)
    for i, draw in enumerate(draws):
        last_seen_prefix[i + 1] = last_seen_prefix[i]
        for num in draw:
            last_seen_prefix[i + 1, num - 1] = i

    X_seq, X_feat, Y = [], [], []

    total = len(draws) - seq_len - 1
    log.log(f"Construyendo dataset: {total:,} muestras a procesar...", "MODEL")

    for i in range(seq_len - 1, len(draws) - 1):
        seq = [one_hot_draw(draws[k]) for k in range(i - seq_len + 1, i + 1)]
        feat = FeatureEngineeringV3.number_features(
            data, i, cum=cum, draws=draws, last_seen_prefix=last_seen_prefix,
        )
        X_seq.append(seq)
        X_feat.append(feat)
        Y.append(one_hot_draw(draws[i + 1]))

        done = i - (seq_len - 1) + 1
        if done % 200 == 0 or done == total:
            pct = 20.0 + 12.0 * (done / max(1, total))
            log.progress(pct, f"Dataset: {done:,}/{total:,} muestras")

    X_seq = np.asarray(X_seq, dtype=np.float32)
    X_feat = np.asarray(X_feat, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    log.log(
        f"Dataset construido: X_seq={X_seq.shape}, "
        f"X_feat={X_feat.shape}, Y={Y.shape}",
        "MODEL",
    )
    return X_seq, X_feat, Y


# ----------------------------------------------------------------
# Auditoría de features
# ----------------------------------------------------------------

def feature_audit(X_feat, Y):
    """Tests de aleatoriedad sobre cada feature respecto al target."""
    n_features = X_feat.shape[1]
    results = []
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or len(Y) != len(X_feat):
        raise ValueError(
            f"Dimensiones incompatibles en feature_audit: X_feat={X_feat.shape}, Y={Y.shape}"
        )

    log.log(f"Auditoría de {n_features} features contra el target...", "AUDIT")

    for j in range(n_features):
        col = np.asarray(X_feat[:, j], dtype=np.float64)
        corrs = []
        if np.std(col) >= 1e-12:
            for target_idx in range(Y.shape[1]):
                target_col = Y[:, target_idx]
                if np.std(target_col) < 1e-12:
                    continue
                value = np.corrcoef(col, target_col)[0, 1]
                if np.isfinite(value):
                    corrs.append(float(value))
        corr = float(np.mean(np.abs(corrs))) if corrs else 0.0

        q, p = ljung_box(col, lags=10)

        results.append({
            "index": j,
            "corr_target": corr,
            "ljung_box_q": q,
            "ljung_box_p": p,
        })

        if j % 50 == 0:
            pct = 45.0 + 8.0 * (j / n_features)
            log.progress(pct, f"Auditoría features: {j}/{n_features}")

    results.sort(key=lambda r: abs(r["corr_target"]), reverse=True)
    top = results[:30]
    significant = [r for r in results if r["ljung_box_p"] < 0.05]

    report = {
        "n_features": n_features,
        "significant_ljung_box": len(significant),
        "top_correlations": top,
        "note": "Ljung-Box detecta autocorrelación; no implica capacidad predictiva.",
    }
    with open(FEATURE_AUDIT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.log(
        f"Features con Ljung-Box p<0.05: {len(significant)}/{n_features}",
        "AUDIT",
    )
    if top:
        best = top[0]
        log.log(
            f"Feature más correlacionada con el target: "
            f"idx={best['index']} corr={best['corr_target']:+.4f} "
            f"p={best['ljung_box_p']:.4f}",
            "AUDIT",
        )
    return report

# ================================================================
# PARTE 4 — MODELOS
# ================================================================

class StatisticalEngine:
    """Motor estadístico multiescala con coocurrencia corregida."""

    WINDOWS = (5, 10, 20, 40, 80, 160, 320, 640)

    def __init__(self, data):
        self.data = sorted(data, key=lambda x: x[0])
        self.draws = [tuple(d) for _, d in self.data]

    def _freq(self, window=None):
        rows = self.draws[-window:] if window else self.draws
        f = np.zeros(MAX_NUM, dtype=np.float64)
        for d in rows:
            for n in d:
                f[n - 1] += 1.0
        s = f.sum()
        return f / s if s > 0 else np.full(MAX_NUM, 1.0 / MAX_NUM)

    def _absence(self):
        gaps = np.zeros(MAX_NUM, dtype=np.float64)
        for n in range(1, MAX_NUM + 1):
            gap = 0
            for d in reversed(self.draws):
                if n in d:
                    break
                gap += 1
            gaps[n - 1] = gap
        return gaps

    def _pair_excess(self, window=500):
        rows = self.draws[-window:] if window else self.draws
        n_draws = len(rows)
        if n_draws == 0:
            return np.zeros(MAX_NUM, dtype=np.float64)

        pair = np.zeros((MAX_NUM, MAX_NUM), dtype=np.float64)
        for d in rows:
            for a, b in combinations(d, 2):
                pair[a - 1, b - 1] += 1.0
                pair[b - 1, a - 1] += 1.0

        expected = PROB_PAIR * n_draws
        excess = (pair - expected) / math.sqrt(expected + 1e-9)
        np.fill_diagonal(excess, 0.0)
        return np.sum(excess, axis=1) / (MAX_NUM - 1)

    def score_numbers(self):
        f_full = self._freq()
        f_wins = {w: self._freq(w) for w in self.WINDOWS}
        absence = self._absence()
        absence_z = safe_zscore(absence)
        pair_score = self._pair_excess(500)
        pair_z = safe_zscore(pair_score)

        weights = {5: 0.22, 10: 0.20, 20: 0.18, 40: 0.14,
                   80: 0.10, 160: 0.07, 320: 0.05, 640: 0.04}
        trend = np.zeros(MAX_NUM, dtype=np.float64)
        for w, wt in weights.items():
            trend += wt * safe_zscore(f_wins[w])

        recency_signal = np.tanh(absence_z / 3.0) * 0.04
        pair_signal = pair_z * 0.06

        raw = trend + recency_signal + pair_signal

        return raw, {
            "full": f_full,
            "windows": f_wins,
            "absence": absence,
            "trend": trend,
            "pair_excess": pair_score,
        }

    def probability_vector(self):
        raw, details = self.score_numbers()
        p = softmax(raw, temperature=1.25)
        return p, details

    def summary(self):
        f = self._freq()
        return {
            "total_draws": len(self.draws),
            "min_date": self.data[0][0].isoformat() if self.data else None,
            "max_date": self.data[-1][0].isoformat() if self.data else None,
            "chi_square": chi_square(f * len(self.draws) * NUMBERS_PER_DRAW),
            "entropy": entropy(f),
        }


# ----------------------------------------------------------------
# 4.2 — ENSEMBLE DE ÁRBOLES
# ----------------------------------------------------------------

class TreeEnsemble:
    """Ensemble de clasificadores por número."""

    def __init__(self):
        self.scaler = None
        self.logistic = [None] * MAX_NUM
        self.hist_gb = [None] * MAX_NUM
        self.extra = [None] * MAX_NUM
        self.forest = [None] * MAX_NUM
        self.xgb = [None] * MAX_NUM
        self.lgbm = [None] * MAX_NUM
        self.fitted = False
        self.feature_count = None
        self.training_rows = 0
        self.report = {}

    @staticmethod
    def _safe_proba(model, X):
        if model is None:
            return None
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            return None

    def fit(self, X_train, Y_train, X_val=None, Y_val=None, progress_callback=None):
        if not SKLEARN_AVAILABLE:
            return {"available": False}

        X_train = np.asarray(X_train, dtype=np.float32)
        Y_train = np.asarray(Y_train, dtype=np.float32)
        self.feature_count = X_train.shape[1]
        self.training_rows = len(X_train)

        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X_train)

        for n in range(MAX_NUM):
            if progress_callback:
                pct = 55.0 + 12.0 * (n / MAX_NUM)
                progress_callback(pct, f"Árboles: número {n+1}/{MAX_NUM}")

            y = Y_train[:, n].astype(np.int8)
            if len(np.unique(y)) < 2:
                continue

            try:
                lr = LogisticRegression(
                    C=0.08, max_iter=2500,
                    class_weight="balanced", solver="lbfgs",
                    random_state=ULTIMATE_SEED, n_jobs=None,
                )
                lr.fit(Xs, y)
                self.logistic[n] = lr
            except Exception as e:
                log.warn(f"Logistic n={n+1}: {e}")

            try:
                gb = HistGradientBoostingClassifier(
                    learning_rate=0.035, max_iter=400,
                    max_leaf_nodes=31, max_depth=7,
                    min_samples_leaf=25, l2_regularization=2.0,
                    early_stopping=True, validation_fraction=0.15,
                    n_iter_no_change=30,
                    random_state=ULTIMATE_SEED + n,
                )
                gb.fit(Xs, y)
                self.hist_gb[n] = gb
            except Exception as e:
                log.warn(f"HistGB n={n+1}: {e}")

            try:
                et = ExtraTreesClassifier(
                    n_estimators=TREE_ESTIMATORS,
                    max_features="sqrt", min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    bootstrap=False, n_jobs=MAX_CPU_WORKERS,
                    random_state=ULTIMATE_SEED + 1000 + n,
                )
                et.fit(X_train, y)
                self.extra[n] = et
            except Exception as e:
                log.warn(f"ExtraTrees n={n+1}: {e}")

            try:
                rf = RandomForestClassifier(
                    n_estimators=TREE_ESTIMATORS,
                    max_features="sqrt", min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    bootstrap=True, n_jobs=MAX_CPU_WORKERS,
                    random_state=ULTIMATE_SEED + 2000 + n,
                )
                rf.fit(X_train, y)
                self.forest[n] = rf
            except Exception as e:
                log.warn(f"RandomForest n={n+1}: {e}")

            if XGB_AVAILABLE:
                try:
                    xgb = XGBClassifier(
                        n_estimators=XGB_ESTIMATORS,
                        max_depth=5, learning_rate=0.035,
                        subsample=0.85, colsample_bytree=0.80,
                        min_child_weight=8, reg_alpha=0.05,
                        reg_lambda=2.0, objective="binary:logistic",
                        eval_metric="logloss", tree_method="hist",
                        n_jobs=MAX_CPU_WORKERS,
                        random_state=ULTIMATE_SEED + 3000 + n,
                        verbosity=0,
                    )
                    xgb.fit(X_train, y)
                    self.xgb[n] = xgb
                except Exception as e:
                    log.warn(f"XGB n={n+1}: {e}")

            if LGBM_AVAILABLE:
                try:
                    lg = lgb.LGBMClassifier(
                        n_estimators=LGBM_ESTIMATORS,
                        max_depth=-1, num_leaves=31,
                        learning_rate=0.03,
                        subsample=0.85, colsample_bytree=0.80,
                        min_child_samples=20,
                        reg_alpha=0.05, reg_lambda=2.0,
                        objective="binary", n_jobs=MAX_CPU_WORKERS,
                        random_state=ULTIMATE_SEED + 4000 + n,
                        verbosity=-1,
                    )
                    lg.fit(X_train, y)
                    self.lgbm[n] = lg
                except Exception as e:
                    log.warn(f"LightGBM n={n+1}: {e}")

        self.fitted = True

        metrics = {
            "available": True,
            "rows": len(X_train),
            "features": self.feature_count,
            "xgboost": XGB_AVAILABLE,
            "lightgbm": LGBM_AVAILABLE,
        }
        if X_val is not None and Y_val is not None and len(X_val):
            try:
                pred = self.predict(X_val)
                metrics["val_brier"] = float(np.mean((Y_val - pred) ** 2))
                metrics["val_hits_top12"] = float(self._topk_hits(Y_val, pred, 12))
            except Exception as e:
                log.warn(f"Métricas validación: {e}")

        self.report = metrics
        return metrics

    @staticmethod
    def _topk_hits(y, p, k=12):
        y = np.asarray(y)
        p = np.asarray(p)
        total = 0.0
        for i in range(len(y)):
            pred = set(np.argsort(p[i])[-k:])
            actual = set(np.where(y[i] > 0.5)[0])
            total += len(pred & actual)
        return total / max(1, len(y))

    def predict_components(self, X):
        if not self.fitted or self.scaler is None:
            return None, None

        X = np.asarray(X, dtype=np.float32)
        Xs = self.scaler.transform(X)

        logistic = np.full((len(X), MAX_NUM), 1.0 / MAX_NUM, dtype=float)
        nonlinear = [np.full((len(X), MAX_NUM), 1.0 / MAX_NUM) for _ in range(5)]

        for n in range(MAX_NUM):
            pred = self._safe_proba(self.logistic[n], Xs)
            if pred is not None:
                logistic[:, n] = pred

            for model, idx, use_scaled in (
                (self.hist_gb[n], 0, True),
                (self.extra[n], 1, False),
                (self.forest[n], 2, False),
                (self.xgb[n], 3, False),
                (self.lgbm[n], 4, False),
            ):
                pred = self._safe_proba(model, Xs if use_scaled else X)
                if pred is not None:
                    nonlinear[idx][:, n] = pred

        logistic = normalize_rows(logistic)

        present = []
        if any(x is not None for x in self.hist_gb):
            present.append(nonlinear[0])
        if any(x is not None for x in self.extra):
            present.append(nonlinear[1])
        if any(x is not None for x in self.forest):
            present.append(nonlinear[2])
        if any(x is not None for x in self.xgb):
            present.append(nonlinear[3])
        if any(x is not None for x in self.lgbm):
            present.append(nonlinear[4])

        if present:
            blend = np.mean(present, axis=0)
            blend = normalize_rows(blend)
        else:
            blend = np.full_like(logistic, 1.0 / MAX_NUM)

        return logistic, blend

    def predict(self, X):
        a, b = self.predict_components(X)
        if a is None:
            return None
        return normalize_rows(0.40 * a + 0.60 * b)


# ----------------------------------------------------------------
# 4.3 — BAYESIANO EMPÍRICO
# ----------------------------------------------------------------

class EmpiricalBayes:
    """Intervalos de credibilidad por número con bootstrap vectorizado."""

    def __init__(self, data, n_bootstrap=2000, n_dirichlet=5000):
        self.data = sorted(data, key=lambda x: x[0])
        self.draws = [tuple(d) for _, d in self.data]
        self.n_bootstrap = n_bootstrap
        self.n_dirichlet = n_dirichlet

    def fit(self, progress_callback=None):
        n = len(self.draws)
        if n < 100:
            self.mean = np.full(MAX_NUM, 1.0 / MAX_NUM)
            self.q05 = self.mean.copy()
            self.q95 = self.mean.copy()
            self.p_top6 = self.mean.copy() * 6
            return self

        rng = np.random.default_rng(ULTIMATE_SEED + 500)

        one_hot = np.zeros((n, MAX_NUM), dtype=np.float32)
        for i, d in enumerate(self.draws):
            for num in d:
                one_hot[i, num - 1] = 1.0

        boot_freqs = np.zeros((self.n_bootstrap, MAX_NUM), dtype=np.float64)
        for b in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_freqs[b] = one_hot[idx].sum(axis=0)
            boot_freqs[b] /= (boot_freqs[b].sum() + 1e-12)
            if progress_callback and b % 200 == 0:
                progress_callback(90.0, f"Bayesiano: bootstrap {b}/{self.n_bootstrap}")

        mean = boot_freqs.mean(axis=0)
        var = boot_freqs.var(axis=0)
        s = mean * (1 - mean) / (var + 1e-12) - 1.0
        concentration = float(np.clip(np.median(s), 5.0, 500.0))
        alpha = np.clip(mean * concentration, 1e-3, None)

        samples = rng.dirichlet(alpha, size=self.n_dirichlet)

        self.mean = samples.mean(axis=0)
        self.q05 = np.quantile(samples, 0.05, axis=0)
        self.q95 = np.quantile(samples, 0.95, axis=0)

        top6 = np.argpartition(-samples, 6, axis=1)[:, :6]
        flat = top6.flatten()
        counts = np.bincount(flat, minlength=MAX_NUM).astype(np.float64)
        self.p_top6 = counts / self.n_dirichlet

        log.log(
            f"Bayesiano: concentración={concentration:.1f}, "
            f"p_top6 max={self.p_top6.max():.3f} min={self.p_top6.min():.3f}",
            "MODEL",
        )
        return self


# ----------------------------------------------------------------
# 4.4 — TRANSFORMER + ENSEMBLE
# ----------------------------------------------------------------

if TORCH_AVAILABLE:

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=512):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]


    class FuxingTransformer(nn.Module):
        def __init__(self, feature_size, d_model=256, heads=8, layers=6, dropout=0.15):
            super().__init__()
            self.proj = nn.Linear(MAX_NUM, d_model)
            self.pos = PositionalEncoding(d_model, max_len=SEQ_LEN + 8)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=heads,
                dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.seq_norm = nn.LayerNorm(d_model)

            self.feat = nn.Sequential(
                nn.Linear(feature_size, 512),
                nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.LayerNorm(128), nn.GELU(),
            )

            self.fusion = nn.Sequential(
                nn.Linear(d_model + 128, 512),
                nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout + 0.05),
                nn.Linear(512, 256),
                nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(256, MAX_NUM),
            )

        def forward(self, seq, feat):
            x = self.proj(seq)
            x = self.pos(x)
            x = self.encoder(x)
            x = self.seq_norm(x[:, -1])
            f = self.feat(feat)
            return self.fusion(torch.cat([x, f], dim=1))


class NeuralEnsemble:
    def __init__(self, n_members=NEURAL_ENSEMBLE_SIZE):
        self.n_members = max(1, int(n_members))
        self.models = []
        self.scaler = None
        self.fitted = False
        self.reports = []

    def fit(self, X_seq, X_feat, Y, progress_callback=None):
        if not TORCH_AVAILABLE:
            return {"available": False}

        X_seq = np.asarray(X_seq, dtype=np.float32)
        X_feat = np.asarray(X_feat, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        n = len(X_feat)
        if n < 300:
            log.warn("Red: muestras insuficientes, se omite.")
            return {"available": True, "trained": False}

        split = int(n * 0.70)
        val_cut = int(n * 0.75)
        split = max(200, min(split, n - 100))
        val_cut = max(split + 30, min(val_cut, n))

        self.scaler = StandardScaler()
        self.scaler.fit(X_feat[:split])
        feat_scaled = self.scaler.transform(X_feat).astype(np.float32)

        device = torch.device(DEVICE)

        for member in range(self.n_members):
            seed = ULTIMATE_SEED + 101 * member
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            log.log(f"Red Transformer {member+1}/{self.n_members} (seed={seed})", "MODEL")

            model = FuxingTransformer(
                feature_size=X_feat.shape[1],
                d_model=256, heads=8, layers=6, dropout=0.15,
            ).to(device)

            tx = torch.tensor(X_seq[:split], dtype=torch.float32, device=device)
            tf = torch.tensor(feat_scaled[:split], dtype=torch.float32, device=device)
            ty = torch.tensor(Y[:split], dtype=torch.float32, device=device)
            vx = torch.tensor(X_seq[split:val_cut], dtype=torch.float32, device=device)
            vf = torch.tensor(feat_scaled[split:val_cut], dtype=torch.float32, device=device)
            vy = torch.tensor(Y[split:val_cut], dtype=torch.float32, device=device)

            pos = ty.mean(dim=0)
            neg = 1.0 - pos
            pos_weight = torch.clamp(neg / torch.clamp(pos, min=1e-4), 1.0, 10.0)

            ds = TensorDataset(tx.detach().cpu(), tf.detach().cpu(), ty.detach().cpu())
            batch_size = min(256, max(32, len(ds) // 12))
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                num_workers=0, pin_memory=(device.type == "cuda"))

            optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=3e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            best = float("inf"); best_state = None; bad = 0
            epochs = 80
            for epoch in range(epochs):
                if progress_callback:
                    base = 68.0 + (member / self.n_members) * 20.0
                    span = 20.0 / self.n_members
                    progress_callback(
                        base + ((epoch + 1) / epochs) * span,
                        f"Red {member+1}/{self.n_members}: época {epoch+1}/{epochs}",
                    )
                model.train()
                for xb, fb, yb in loader:
                    xb = xb.to(device, non_blocking=True)
                    fb = fb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(xb, fb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step(epoch + 1)

                model.eval()
                with torch.no_grad():
                    vlogits = model(vx, vf)
                    vloss = criterion(vlogits, vy).item()
                if vloss + 1e-7 < best:
                    best = vloss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                if bad >= 15:
                    break

            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            self.models.append(model)
            self.reports.append({"seed": seed, "epochs": epoch + 1, "val_loss": float(best)})
            log.log(f"Red {member+1}: épocas={epoch+1} val_loss={best:.6f}", "MODEL")

        self.fitted = bool(self.models)
        return {
            "available": True, "trained": self.fitted,
            "ensemble_size": len(self.models),
            "members": self.reports,
        }

    def predict(self, X_seq_last, X_feat_last):
        if not self.fitted or self.scaler is None:
            return None
        device = torch.device(DEVICE)
        feat = self.scaler.transform(np.asarray(X_feat_last, dtype=np.float32)).astype(np.float32)
        xs = torch.tensor(X_seq_last, dtype=torch.float32, device=device)
        xf = torch.tensor(feat, dtype=torch.float32, device=device)
        preds = []
        with torch.no_grad():
            for m in self.models:
                m.eval()
                logits = m(xs, xf)
                preds.append(torch.sigmoid(logits).cpu().numpy())
        return normalize_rows(np.mean(preds, axis=0))
    
# ================================================================
# PARTE 5 — CALIBRACIÓN + CONFORMAL PREDICTION
# ================================================================

class IsotonicCalibrator:
    """Calibra las probabilidades del ensemble por número."""

    def __init__(self, min_samples=200):
        self.calibrators = [None] * MAX_NUM
        self.fitted = False
        self.min_samples = min_samples
        self.report = {}

    def fit(self, P_val, Y_val):
        if not SKLEARN_AVAILABLE:
            return {"available": False}

        P_val = np.asarray(P_val, dtype=float)
        Y_val = np.asarray(Y_val, dtype=float)
        if len(P_val) < self.min_samples:
            log.warn(
                f"Calibración: sólo {len(P_val)} muestras; "
                f"se omite (mínimo {self.min_samples})."
            )
            return {"available": True, "fitted": False}

        fitted_count = 0
        brier_before, brier_after = [], []

        for n in range(MAX_NUM):
            p = P_val[:, n]
            y = Y_val[:, n]
            if len(np.unique(y)) < 2 or np.std(p) < 1e-9:
                continue

            try:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(p, y)
                self.calibrators[n] = iso

                p_cal = iso.predict(p)
                brier_before.append(float(np.mean((p - y) ** 2)))
                brier_after.append(float(np.mean((p_cal - y) ** 2)))
                fitted_count += 1
            except Exception as e:
                log.warn(f"Calibración n={n+1}: {e}")

        self.fitted = fitted_count > 0
        self.report = {
            "fitted_numbers": fitted_count,
            "brier_before": float(np.mean(brier_before)) if brier_before else None,
            "brier_after": float(np.mean(brier_after)) if brier_after else None,
            "brier_evaluation": "in-sample; indicador descriptivo, no estimación independiente",
        }
        if self.report["brier_before"] is not None:
            improvement = (
                (self.report["brier_before"] - self.report["brier_after"])
                / self.report["brier_before"] * 100
            )
            log.log(
                f"Calibración isotónica: {fitted_count}/49 números · "
                f"Brier {self.report['brier_before']:.6f} -> "
                f"{self.report['brier_after']:.6f} "
                f"({improvement:+.2f}%)",
                "MODEL",
            )
        return self.report

    def transform(self, P):
        P = np.asarray(P, dtype=float)
        single = P.ndim == 1
        if single:
            P = P[None, :]
        out = P.copy()
        for n, iso in enumerate(self.calibrators):
            if iso is None:
                continue
            try:
                out[:, n] = iso.predict(P[:, n])
            except Exception:
                pass
        out = normalize_rows(out)
        return out[0] if single else out


# ----------------------------------------------------------------
# 5.2 — CONFORMAL PREDICTION
# ----------------------------------------------------------------

class ConformalPredictor:
    def __init__(self, alpha=0.10):
        self.alpha = alpha
        self.threshold = None
        self.fitted = False
        self.coverage_val = None

    def fit(self, P_cal, Y_cal):
        """P_cal debe estar en la misma escala que predict_set (normalize_rows)."""
        P_cal = np.asarray(P_cal, dtype=float)
        Y_cal = np.asarray(Y_cal, dtype=float)
        n = len(P_cal)
        if n < 100:
            return {"available": True, "fitted": False}

        scores = []
        for i in range(n):
            real_idx = np.where(Y_cal[i] > 0.5)[0]
            if len(real_idx) == 0:
                continue
            s = 1.0 - float(np.min(P_cal[i, real_idx]))
            scores.append(s)

        if not scores:
            return {"available": True, "fitted": False}

        scores = np.asarray(scores)
        q_level = min(1.0, math.ceil((n + 1) * (1 - self.alpha)) / n)
        self.threshold = float(np.quantile(scores, q_level))
        self.fitted = True

        covered = 0
        for i in range(n):
            real_idx = np.where(Y_cal[i] > 0.5)[0]
            if len(real_idx) == 0:
                continue
            mask = (1.0 - P_cal[i]) <= self.threshold
            if all(mask[idx] for idx in real_idx):
                covered += 1
        self.coverage_val = covered / max(1, n)

        log.log(
            f"Conformal alpha={self.alpha:.2f}: umbral={self.threshold:.4f} · "
            f"cobertura empírica={self.coverage_val*100:.2f}%",
            "MODEL",
        )
        return {
            "available": True, "fitted": True,
            "threshold": self.threshold,
            "coverage_val": self.coverage_val,
        }

    def predict_set(self, p):
        if not self.fitted:
            k = max(6, int(round(MAX_NUM * (1 - self.alpha))))
            return set(np.argsort(p)[-k:] + 1)
        p = np.asarray(p, dtype=float)
        mask = (1.0 - p) <= self.threshold
        s = set((np.where(mask)[0] + 1).tolist())
        if len(s) < 6:
            top = np.argsort(p)[-6:] + 1
            s.update(top.tolist())
        return s


# ----------------------------------------------------------------
# 5.3 — OPTIMIZADOR DE PESOS DEL ENSEMBLE
# ----------------------------------------------------------------

class WeightOptimizer:
    """Busca los pesos del ensemble minimizando Brier y maximizando
    hits@12/18 sobre un conjunto de validación temporal."""

    def __init__(self, n_dirichlet=30000, n_refine=10000):
        self.n_dirichlet = n_dirichlet
        self.n_refine = n_refine
        self.weights = None
        self.report = {}

    def optimize(self, y_val, components, progress_callback=None):
        valid = [(i, normalize_rows(p)) for i, p in enumerate(components) if p is not None]
        if not valid or len(y_val) == 0:
            self.weights = np.zeros(len(components))
            if valid:
                self.weights[[i for i, _ in valid]] = 1.0 / len(valid)
            return self.weights

        y_val = np.asarray(y_val, dtype=float)
        n_comp = len(valid)
        stack = np.stack([p for _, p in valid], axis=0)

        rng = np.random.default_rng(ULTIMATE_SEED + 777)

        candidates = []
        for i in range(n_comp):
            w = np.zeros(n_comp); w[i] = 1.0
            candidates.append(w)
        candidates.append(np.ones(n_comp) / n_comp)

        if progress_callback:
            progress_callback(88.0, "Pesos: generando combinaciones Dirichlet")
        alphas = (0.25, 0.5, 1.0, 2.0)
        per = self.n_dirichlet // len(alphas)
        for a in alphas:
            candidates.extend(rng.dirichlet(np.full(n_comp, a), size=per))

        candidates = np.asarray(candidates, dtype=float)
        best_obj = float("inf")
        best_w = candidates[0].copy()
        best_metrics = {}

        batch = 256
        total = len(candidates)
        for start in range(0, total, batch):
            end = min(total, start + batch)
            if progress_callback:
                progress_callback(
                    88.0 + 4.0 * (end / total),
                    f"Pesos: evaluando {end:,}/{total:,}",
                )
            w_batch = candidates[start:end]
            mix = np.einsum("bc,cnk->bnk", w_batch, stack)
            mix = mix / (mix.sum(axis=2, keepdims=True) + 1e-12)

            yb = y_val[None, :, :]
            briers = np.mean((yb - mix) ** 2, axis=(1, 2))

            top12 = np.argpartition(mix, -12, axis=2)[:, :, -12:]
            top18 = np.argpartition(mix, -18, axis=2)[:, :, -18:]
            actual = y_val[None, :, :] > 0.5
            h12 = np.mean(np.sum(np.take_along_axis(actual, top12, axis=2), axis=2), axis=1)
            h18 = np.mean(np.sum(np.take_along_axis(actual, top18, axis=2), axis=2), axis=1)

            objs = briers - 0.010 * h12 - 0.004 * h18
            local = int(np.argmin(objs))
            if objs[local] < best_obj:
                best_obj = float(objs[local])
                best_w = w_batch[local].copy()
                best_metrics = {
                    "brier": float(briers[local]),
                    "hits12": float(h12[local]),
                    "hits18": float(h18[local]),
                }

        if progress_callback:
            progress_callback(92.0, f"Pesos: refinamiento local ({self.n_refine:,} iter)")
        local_w = best_w.copy()
        for radius in (0.10, 0.05, 0.02, 0.01):
            for k in range(self.n_refine // 4):
                trial = np.clip(local_w + rng.normal(0, radius, size=n_comp), 0.0, None)
                if trial.sum() <= 0:
                    continue
                trial /= trial.sum()

                mix = np.einsum("c,cnk->nk", trial, stack)
                mix = mix / (mix.sum(axis=1, keepdims=True) + 1e-12)
                brier = float(np.mean((y_val - mix) ** 2))

                top12 = np.argpartition(mix, -12, axis=1)[:, -12:]
                top18 = np.argpartition(mix, -18, axis=1)[:, -18:]
                actual = y_val > 0.5
                h12 = float(np.mean(np.sum(np.take_along_axis(actual, top12, axis=1), axis=1)))
                h18 = float(np.mean(np.sum(np.take_along_axis(actual, top18, axis=1), axis=1)))
                obj = brier - 0.010 * h12 - 0.004 * h18

                if obj < best_obj:
                    best_obj = obj
                    best_w = trial.copy()
                    local_w = trial.copy()
                    best_metrics = {"brier": brier, "hits12": h12, "hits18": h18}

        full = np.zeros(len(components), dtype=float)
        for w, (idx, _) in zip(best_w, valid):
            full[idx] = w
        self.weights = full
        self.report = {
            "weights": full.tolist(),
            "components_present": [i for i, _ in valid],
            **best_metrics,
        }
        log.log(
            "Pesos óptimos: " + ", ".join(f"{x:.3f}" for x in full),
            "OPTIM",
        )
        log.log(
            f"Validación: Brier={best_metrics.get('brier', 0):.6f} · "
            f"hits@12={best_metrics.get('hits12', 0):.3f} · "
            f"hits@18={best_metrics.get('hits18', 0):.3f}",
            "OPTIM",
        )
        return full
    
# ================================================================
# PARTE 6 — VALIDACIÓN RIGUROSA
# ================================================================

class NestedWalkForward:
    def __init__(self, n_splits=6, purge=SEQ_LEN, embargo=7):
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo
        self.results = []

    def run(self, X_seq, X_feat, Y, train_fn, predict_fn, progress_callback=None):
        n = len(Y)
        if n < self.n_splits * 100:
            log.warn(
                f"Walk-forward: pocas muestras ({n}), se reduce a 3 splits."
            )
            self.n_splits = 3

        block = n // (self.n_splits + 1)
        results = []

        for k in range(self.n_splits):
            test_start = (k + 1) * block
            test_end = min(n, test_start + block)
            train_end = max(0, test_start - self.purge)
            if train_end < 200 or test_end <= test_start:
                continue

            Xtr_seq = X_seq[:train_end]
            Xtr_feat = X_feat[:train_end]
            Ytr = Y[:train_end]
            Xte_seq = X_seq[test_start:test_end]
            Xte_feat = X_feat[test_start:test_end]
            Yte = Y[test_start:test_end]

            if progress_callback:
                progress_callback(
                    92.0 + 2.0 * (k / self.n_splits),
                    f"Walk-forward: split {k+1}/{self.n_splits} "
                    f"(train={train_end}, test={test_end - test_start})",
                )

            try:
                model = train_fn(Xtr_seq, Xtr_feat, Ytr)
                Pte = predict_fn(model, Xte_seq, Xte_feat, test_start)
            except Exception as e:
                log.warn(f"Walk-forward split {k+1}: {e}")
                continue

            if Pte is None:
                continue

            brier = float(np.mean((Yte - Pte) ** 2))
            h12 = self._hits(Pte, Yte, 12)
            h18 = self._hits(Pte, Yte, 18)

            results.append({
                "split": k + 1,
                "train_size": int(train_end),
                "test_size": int(test_end - test_start),
                "brier": brier,
                "hits_top12": h12,
                "hits_top18": h18,
            })
            log.log(
                f"Walk-forward {k+1}/{self.n_splits}: "
                f"Brier={brier:.6f} hits@12={h12:.3f} hits@18={h18:.3f}",
                "AUDIT",
            )

        self.results = results
        if not results:
            return {"tests": 0}

        summary = {
            "tests": len(results),
            "mean_brier": float(np.mean([r["brier"] for r in results])),
            "mean_hits_top12": float(np.mean([r["hits_top12"] for r in results])),
            "mean_hits_top18": float(np.mean([r["hits_top18"] for r in results])),
            "splits": results,
        }
        log.log(
            f"Walk-forward resumen: {summary['tests']} splits · "
            f"Brier={summary['mean_brier']:.6f} · "
            f"hits@12={summary['mean_hits_top12']:.3f} · "
            f"hits@18={summary['mean_hits_top18']:.3f}",
            "AUDIT",
        )
        return summary

    @staticmethod
    def _hits(P, Y, k):
        total = 0.0
        for i in range(len(Y)):
            pred = set(np.argsort(P[i])[-k:])
            actual = set(np.where(Y[i] > 0.5)[0])
            total += len(pred & actual)
        return total / max(1, len(Y))


# ----------------------------------------------------------------
# 6.2 — DIEBOLD-MARIANO CONTRA BASELINE ALEATORIO
# ----------------------------------------------------------------

def diebold_mariano(model_probs, Y, baseline_probs=None, max_lag=None):
    """Comparación pareada de Brier con varianza HAC/Newey-West.

    Es una prueba exploratoria: requiere predicciones verdaderamente fuera de muestra
    y no demuestra que una lotería sea predecible.
    """
    Y = np.asarray(Y, dtype=float)
    model_probs = normalize_rows(np.asarray(model_probs, dtype=float))
    if baseline_probs is None:
        baseline_probs = np.full_like(Y, NUMBERS_PER_DRAW / MAX_NUM)
    else:
        baseline_probs = normalize_rows(np.asarray(baseline_probs, dtype=float))

    if Y.ndim != 2 or model_probs.shape != Y.shape or baseline_probs.shape != Y.shape:
        return {"n": 0, "error": "Dimensiones incompatibles"}
    loss_model = np.mean((Y - model_probs) ** 2, axis=1)
    loss_base = np.mean((Y - baseline_probs) ** 2, axis=1)
    d = np.asarray(loss_base - loss_model, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return {"n": n, "dm_stat": 0.0, "p_value": 1.0,
                "mean_diff": float(d.mean()) if n else 0.0,
                "method": "muestra insuficiente"}

    mean_d = float(d.mean())
    centered = d - mean_d
    lag = int(max_lag if max_lag is not None else min(7, max(1, int(n ** (1 / 3)))))
    lag = max(0, min(lag, n - 1))
    long_var = float(np.dot(centered, centered) / n)
    for k in range(1, lag + 1):
        gamma = float(np.dot(centered[k:], centered[:-k]) / n)
        weight = 1.0 - k / (lag + 1.0)
        long_var += 2.0 * weight * gamma
    long_var = max(long_var, 1e-18)
    dm = mean_d / math.sqrt(long_var / n)
    from math import erf
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(dm) / math.sqrt(2.0))))
    result = {
        "n": n, "mean_diff": mean_d, "dm_stat": float(dm),
        "p_value": float(max(0.0, min(1.0, p))),
        "model_better": bool(mean_d > 0),
        "significant_5pct": bool(p < 0.05 and mean_d > 0),
        "method": "Brier pareado con varianza HAC/Newey-West",
        "hac_max_lag": lag,
    }
    verdict = ("diferencia favorable al modelo, p<0.05" if result["significant_5pct"]
               else "sin evidencia estadística suficiente de mejora")
    log.log(f"DM-HAC: {verdict} (estadístico={dm:+.3f}, p={p:.4f})", "AUDIT")
    return result


# ----------------------------------------------------------------
# 6.3 — BACKTESTING DIARIO
# ----------------------------------------------------------------

class DailyBacktester:
    """Walk-forward: en cada punto, top-K números del motor
    estadístico vs sorteo real. Mide hits."""

    def __init__(self, data, top_k=12, start=800):
        self.data = sorted(data, key=lambda x: x[0])
        self.top_k = top_k
        self.start = start

    def run(self, step=1, progress_callback=None):
        n = len(self.data)
        if n <= self.start + 1:
            return {"tests": 0}

        hits = []
        six_hit = 0
        total = (n - 1 - self.start) // step
        done = 0

        for i in range(self.start, n - 1, step):
            train = self.data[:i]
            actual = set(self.data[i][1])
            engine = StatisticalEngine(train)
            p, _ = engine.probability_vector()
            top = set(np.argsort(p)[-self.top_k:] + 1)
            h = len(actual & top)
            hits.append(h)
            if h == 6:
                six_hit += 1
            done += 1
            if progress_callback and done % 25 == 0:
                progress_callback(
                    30.0 + 10.0 * (done / max(1, total)),
                    f"Backtest diario: {done}/{total}",
                )

        arr = np.asarray(hits, dtype=float)
        expected_random = NUMBERS_PER_DRAW * self.top_k / MAX_NUM
        result = {
            "tests": int(len(arr)),
            "mean_hits_top12": float(arr.mean()) if len(arr) else 0.0,
            "median_hits_top12": float(np.median(arr)) if len(arr) else 0.0,
            "p_ge_3": float(np.mean(arr >= 3)) if len(arr) else 0.0,
            "p_ge_4": float(np.mean(arr >= 4)) if len(arr) else 0.0,
            "p_ge_5": float(np.mean(arr >= 5)) if len(arr) else 0.0,
            "p_6": float(np.mean(arr == 6)) if len(arr) else 0.0,
            "six_hit_count": int(six_hit),
            "random_expected_hits_top12": float(expected_random),
            "lift_over_random": float(arr.mean() / expected_random) if expected_random else 0.0,
        }
        with open(BACKTEST_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log.log(
            f"Backtest diario: {result['tests']} sorteos · "
            f"media={result['mean_hits_top12']:.3f} · "
            f"lift={result['lift_over_random']:.3f}",
            "AUDIT",
        )
        return result
    
# ================================================================
# PARTE 7 — GENERACIÓN DE APUESTAS
# ================================================================

class VectorizedMonteCarlo:
    def __init__(self, n_iterations=1_000_000, seed=ULTIMATE_SEED):
        self.n_iterations = int(n_iterations)
        self.seed = int(seed)

    def sample(self, probs, constraints=True, progress_callback=None, batch_size=50000):
        probs = normalize_mass(probs)
        rng = np.random.default_rng(self.seed)

        freq = np.zeros(MAX_NUM, dtype=np.float64)
        valid = 0
        done = 0
        total = self.n_iterations

        while done < total:
            b = min(batch_size, total - done)
            u = rng.random((b, MAX_NUM))
            logits = np.log(np.clip(probs, 1e-12, None)) - np.log(
                -np.log(np.clip(u, 1e-12, 1 - 1e-12))
            )
            top6 = np.argpartition(-logits, 6, axis=1)[:, :6]

            if constraints:
                sorted_top = np.sort(top6 + 1, axis=1)
                evens = (sorted_top % 2 == 0).sum(axis=1)
                lows = (sorted_top <= 24).sum(axis=1)
                sums = sorted_top.sum(axis=1)
                spread = sorted_top[:, -1] - sorted_top[:, 0]
                diffs = np.diff(sorted_top, axis=1) == 1
                cons = diffs.sum(axis=1)

                mask = (
                    (evens >= 1) & (evens <= 5) &
                    (lows >= 1) & (lows <= 5) &
                    (sums >= 45) & (sums <= 255) &
                    (spread >= 10) &
                    (cons <= 3)
                )
                kept = sorted_top[mask]
            else:
                kept = np.sort(top6 + 1, axis=1)

            if len(kept):
                valid += len(kept)
                flat = kept.flatten() - 1
                np.add.at(freq, flat, 1.0)

            done += b
            if progress_callback and (done % batch_size == 0 or done == total):
                progress_callback(
                    55.0 + 8.0 * (done / total),
                    f"Monte Carlo: {done:,}/{total:,}",
                )

        if valid == 0:
            return probs
        return normalize_mass(freq / valid)

    def top_combinations(self, probs, n=15, n_candidates=None, progress_callback=None):
        if n_candidates is None:
            n_candidates = min(2_000_000, max(200_000, n * 100_000))

        probs = normalize_mass(probs)
        rng = np.random.default_rng(self.seed + 7)

        pool_chunks = []
        total_unique = 0
        done = 0
        batch_size = 50000

        while done < n_candidates and total_unique < n_candidates:
            b = min(batch_size, n_candidates - done)
            u = rng.random((b, MAX_NUM))
            logits = np.log(np.clip(probs, 1e-12, None)) - np.log(
                -np.log(np.clip(u, 1e-12, 1 - 1e-12))
            )
            top6 = np.argpartition(-logits, 6, axis=1)[:, :6]
            top6 = np.sort(top6 + 1, axis=1)

            evens = (top6 % 2 == 0).sum(axis=1)
            lows = (top6 <= 24).sum(axis=1)
            sums = top6.sum(axis=1)
            spread = top6[:, -1] - top6[:, 0]
            cons = (np.diff(top6, axis=1) == 1).sum(axis=1)
            mask = (
                (evens >= 1) & (evens <= 5) &
                (lows >= 1) & (lows <= 5) &
                (sums >= 45) & (sums <= 255) &
                (spread >= 10) &
                (cons <= 3)
            )
            kept = top6[mask]

            if len(kept):
                unique_batch = np.unique(kept, axis=0)
                pool_chunks.append(unique_batch)
                if len(pool_chunks) > 1:
                    merged = np.unique(np.vstack(pool_chunks), axis=0)
                    pool_chunks = [merged]
                total_unique = sum(len(c) for c in pool_chunks)

            done += b
            if progress_callback and (done % batch_size == 0 or done >= n_candidates):
                progress_callback(
                    60.0 + 8.0 * (done / max(1, n_candidates)),
                    f"Candidatos únicos: {total_unique:,} / {done:,} generados",
                )
            if total_unique >= n_candidates:
                break

        if not pool_chunks:
            return []

        final = pool_chunks[0]
        if len(final) > n_candidates:
            idx = rng.permutation(len(final))[:n_candidates]
            final = final[idx]

        return [tuple(int(x) for x in row) for row in final]


# ----------------------------------------------------------------
# 7.2 — MODELO DE POPULARIDAD
# ----------------------------------------------------------------

class PopularityModel:
    def __init__(self):
        self.birthday_penalty = 0.15
        self.sequence_penalty = 0.30
        self.low_cluster_penalty = 0.10
        self.repeated_decade_penalty = 0.08
        self.diagonal_penalty = 0.12
        self.mod5_penalty = 0.06

    def score(self, draw):
        draw = tuple(sorted(draw))
        s = 0.0

        n31 = sum(n <= 31 for n in draw)
        if n31 >= 5:
            s -= self.birthday_penalty
        if n31 == 6:
            s -= self.birthday_penalty * 0.7

        n12 = sum(n <= 12 for n in draw)
        if n12 >= 4:
            s -= self.low_cluster_penalty

        cons = sum(draw[i + 1] == draw[i] + 1 for i in range(5))
        if cons >= 4:
            s -= self.sequence_penalty
        elif cons >= 2:
            s -= self.sequence_penalty * 0.3

        decades = set((n - 1) // 10 for n in draw)
        if len(decades) <= 2:
            s -= self.repeated_decade_penalty

        if len({n % 5 for n in draw}) <= 2:
            s -= self.mod5_penalty

        diag1 = {1, 8, 15, 22, 29, 36, 43}
        diag2 = {7, 14, 21, 28, 35, 42, 49}
        if len(set(draw) & diag1) >= 4 or len(set(draw) & diag2) >= 4:
            s -= self.diagonal_penalty

        return s


# ----------------------------------------------------------------
# 7.3 — PUNTUACIÓN ESTRUCTURAL
# ----------------------------------------------------------------

class StructuralScorer:
    def __init__(self, data):
        self.data = sorted(data, key=lambda x: x[0])
        self.draws = [tuple(d) for _, d in self.data]
        self._build_reference_stats()

    def _build_reference_stats(self):
        recent = self.draws[-500:]
        self.sums = np.array([sum(d) for d in recent], dtype=float)
        self.spreads = np.array([draw_range(d) for d in recent], dtype=float)
        self.parities = Counter(even_count(d) for d in recent)
        self.lows = Counter(low_count(d) for d in recent)
        self.historical = set(tuple(sorted(d)) for d in self.draws)

    def score(self, draw):
        draw = tuple(sorted(draw))
        s = 0.0
        total = sum(draw)
        even = even_count(draw)
        low = low_count(draw)
        cons = sum(draw[i + 1] == draw[i] + 1 for i in range(5))
        spread = draw_range(draw)

        s += math.log1p(self.parities.get(even, 0))
        s += math.log1p(self.lows.get(low, 0))

        if len(self.sums):
            z = abs((total - self.sums.mean()) / (self.sums.std() + 1e-9))
            s -= min(z, 5.0) * 0.12

        if cons > 2:
            s -= 0.25 * (cons - 2)
        if spread < 15:
            s -= 0.20
        if spread > 48:
            s -= 0.10

        return s


# ----------------------------------------------------------------
# 7.4 — GENERADOR DE CARTERA DIVERSIFICADA
# ----------------------------------------------------------------

class BetPortfolio:
    """Selecciona N combinaciones que maximicen score global y
    diversidad (poca superposición entre ellas)."""

    def __init__(self, data, number_probs):
        self.data = sorted(data, key=lambda x: x[0])
        self.number_probs = normalize_mass(number_probs)
        self.structural = StructuralScorer(self.data)
        self.popularity = PopularityModel()

    def _combo_score(self, draw):
        s = 0.0
        for n in draw:
            s += math.log(self.number_probs[n - 1] + 1e-12)
        s += self.structural.score(draw)
        s += self.popularity.score(draw)
        if tuple(sorted(draw)) in self.structural.historical:
            s -= 0.04
        return float(s)

    def select(self, candidates, n=15, max_overlap=4, progress_callback=None):
        if not candidates:
            return []

        scored = []
        total = len(candidates)
        for i, c in enumerate(candidates):
            scored.append((self._combo_score(c), c))
            if progress_callback and i % 5000 == 0:
                progress_callback(
                    70.0 + 6.0 * (i / max(1, total)),
                    f"Cartera: puntuando {i:,}/{total:,}",
                )

        scored.sort(key=lambda x: x[0], reverse=True)

        selected = []
        for sc, draw in scored:
            if all(len(set(draw) & set(prev)) <= max_overlap for prev in selected):
                selected.append(draw)
            if len(selected) >= n:
                break

        return selected
    
# ================================================================
# PARTE 8 — MOTOR SEMANAL ROBUSTO
# ================================================================

class WeeklyEngine:
    WINDOWS = (10, 25, 50, 100, 250, 500, 1000, 2000)
    WINDOW_WEIGHTS = {
        10: 0.10, 25: 0.13, 50: 0.15, 100: 0.15,
        250: 0.14, 500: 0.12, 1000: 0.11, 2000: 0.10,
    }

    def __init__(self, data, progress_callback=None, model_probabilities=None):
        self.data = sorted(data, key=lambda x: x[0])
        self.draws = [tuple(d) for _, d in self.data]
        self.progress = progress_callback or (lambda p, m: None)
        self.model_probs = (
            normalize_mass(model_probabilities)
            if model_probabilities is not None else None
        )
        self._pair_cache = None

    @staticmethod
    def _freq(rows):
        f = np.zeros(MAX_NUM, dtype=np.float64)
        for _, d in rows:
            for n in d:
                f[n - 1] += 1.0
        s = f.sum()
        return f / s if s > 0 else np.full(MAX_NUM, 1.0 / MAX_NUM)

    def _multi_scale(self):
        components = []
        global_p = self._freq(self.data)
        for w, wt in self.WINDOW_WEIGHTS.items():
            rows = self.data[-min(w, len(self.data)):]
            p = self._freq(rows)
            p = 0.85 * p + 0.15 * global_p
            components.append((wt, p))
        total = sum(w for w, _ in components)
        base = sum((w / total) * p for w, p in components)
        return normalize_mass(base), components

    def _stability(self, components):
        matrix = np.vstack([p for _, p in components])
        mean_p = matrix.mean(axis=0)
        var = matrix.std(axis=0)
        stab = mean_p / (1.0 + 8.0 * var)
        return normalize_mass(stab)

    def _analog(self, n_anchors=500):
        n = len(self.draws)
        if n < 1000:
            return np.full(MAX_NUM, 1.0 / MAX_NUM), 0.0, 0

        def signature(end):
            rows = self.data[:end]
            parts = []
            for w in (10, 25, 50, 100, 250):
                parts.append(self._freq(rows[-w:]))
            recent = self.draws[max(0, end - 50):end]
            if recent:
                sums = np.array([sum(d) for d in recent], dtype=float)
                evens = np.array([even_count(d) for d in recent], dtype=float)
                extra = np.array([sums.mean(), sums.std(), evens.mean(), evens.std()])
            else:
                extra = np.zeros(4)
            return np.concatenate(parts + [extra])

        current = signature(n)
        lo, hi = 700, n - 8
        if hi <= lo:
            return np.full(MAX_NUM, 1.0 / MAX_NUM), 0.0, 0
        step = max(1, (hi - lo) // n_anchors)
        candidates = list(range(lo, hi, step))[-n_anchors:]

        scale = np.std(current) + 1e-9
        scored = []
        for anchor in candidates:
            sig = signature(anchor)
            dist = float(np.mean(((sig - current) / scale) ** 2))
            scored.append((dist, anchor))
        scored.sort(key=lambda x: x[0])
        chosen = scored[:min(60, len(scored))]

        out = np.zeros(MAX_NUM, dtype=np.float64)
        total_w = 0.0
        for rank, (dist, anchor) in enumerate(chosen):
            w = math.exp(-min(dist, 25.0) / 4.0) / (1.0 + 0.015 * rank)
            for _, draw in self.data[anchor:anchor + 7]:
                for nmb in draw:
                    out[nmb - 1] += w
            total_w += 7.0 * w
        if total_w <= 0:
            return np.full(MAX_NUM, 1.0 / MAX_NUM), 0.0, 0
        out /= total_w

        best_dist = float(chosen[0][0]) if chosen else 0.0
        reliability = math.exp(-min(best_dist, 20.0) / 5.0)
        return normalize_mass(out), reliability, len(chosen)

    def _pair_triple_scores(self):
        if self._pair_cache is not None:
            return self._pair_cache
        rows = self.data[-1000:]
        pair = Counter()
        triple = Counter()
        for _, draw in rows:
            for p in combinations(draw, 2):
                pair[p] += 1
            for tri in combinations(draw, 3):
                triple[tri] += 1

        pair_score = np.zeros(MAX_NUM, dtype=np.float64)
        for (a, b), c in pair.items():
            pair_score[a - 1] += c
            pair_score[b - 1] += c
        pair_score = normalize_mass(pair_score)

        triple_score = np.zeros(MAX_NUM, dtype=np.float64)
        for tri, c in triple.items():
            for n in tri:
                triple_score[n - 1] += c
        triple_score = normalize_mass(triple_score)

        self._pair_cache = (pair_score, triple_score, pair, triple)
        return self._pair_cache

    def robust_vector(self):
        self.progress(6, "Semanal: calculando multiescala...")
        base, comps = self._multi_scale()

        self.progress(20, "Semanal: evaluando analogías históricas...")
        analog, analog_rel, analog_n = self._analog()

        self.progress(38, "Semanal: midiendo estabilidad entre ventanas...")
        stab = self._stability(comps)

        self.progress(48, "Semanal: coocurrencias de pares y triples...")
        pair_s, triple_s, pair, triple = self._pair_triple_scores()

        global_p = self._freq(self.data)

        wa = 0.22 * analog_rel
        wb = 0.28
        ws = 0.22
        wp = 0.12
        wt = 0.06
        wg = max(0.0, 1.0 - (wa + wb + ws + wp + wt))

        robust = (
            wg * global_p +
            wb * base +
            wa * analog +
            ws * stab +
            wp * pair_s +
            wt * triple_s
        )
        robust = normalize_mass(robust)

        if self.model_probs is not None:
            robust = normalize_mass(0.85 * robust + 0.15 * self.model_probs)
            self.progress(56, "Semanal: integrando señal del ensemble Fuxing...")

        details = {
            "base": base,
            "analog": analog,
            "stability": stab,
            "pair": pair_s,
            "triple": triple_s,
            "analog_reliability": analog_rel,
            "analog_count": analog_n,
            "pair_count": len(pair),
            "triple_count": len(triple),
            "weights": {"global": wg, "base": wb, "analog": wa,
                        "stability": ws, "pair": wp, "triple": wt},
        }
        return robust, details

    def combo_score(self, draw, probs, details):
        pair_s, triple_s, pair, triple = self._pair_triple_scores()
        stab = details["stability"]
        analog = details["analog"]

        s = 0.0
        for n in draw:
            s += math.log(probs[n - 1] + 1e-12)
            s += 0.55 * math.log(stab[n - 1] + 1e-12)
            s += 0.35 * math.log(analog[n - 1] + 1e-12)

        for a, b in combinations(draw, 2):
            s += 0.012 * math.log1p(pair.get((a, b), 0))
        for tri in combinations(draw, 3):
            s += 0.006 * math.log1p(triple.get(tri, 0))

        even = even_count(draw)
        low = low_count(draw)
        total = sum(draw)
        cons = sum(draw[i + 1] == draw[i] + 1 for i in range(5))
        spread = draw_range(draw)

        if even in (0, 6): s -= 0.35
        if low in (0, 6): s -= 0.30
        if total < 75 or total > 225: s -= 0.10
        if cons > 2: s -= 0.18 * (cons - 2)
        if spread < 15: s -= 0.18
        if sum(n <= 31 for n in draw) >= 5: s -= 0.08

        return float(s)

    def deterministic_best(self, probs, details):
        top = sorted((np.argsort(probs)[::-1][:12] + 1).tolist())
        best, best_s = None, -float("inf")
        for combo in combinations(top, 6):
            if not self._valid(combo):
                continue
            sc = self.combo_score(combo, probs, details)
            if sc > best_s:
                best_s, best = sc, combo
        return tuple(best) if best else tuple(sorted(top[:6]))

    @staticmethod
    def _valid(draw):
        even = even_count(draw)
        low = low_count(draw)
        total = sum(draw)
        cons = sum(draw[i + 1] == draw[i] + 1 for i in range(5))
        spread = draw_range(draw)
        return (
            1 <= even <= 5 and
            1 <= low <= 5 and
            45 <= total <= 255 and
            cons <= 3 and
            spread >= 10
        )

    def best_combination(self, probs, details, iterations=MC_ITERATIONS_SEMANAL):
        self.progress(62, f"Semanal: Monte Carlo {iterations:,} iteraciones...")
        probs = normalize_mass(probs)
        rng = np.random.default_rng(ULTIMATE_SEED + 9001)
        pair_s, triple_s, pair_counts, triple_counts = self._pair_triple_scores()
        stab = np.asarray(details["stability"], dtype=float)
        analog = np.asarray(details["analog"], dtype=float)

        pair_lookup = np.zeros((MAX_NUM, MAX_NUM), dtype=np.float32)
        for (a, b), count in pair_counts.items():
            pair_lookup[a - 1, b - 1] = pair_lookup[b - 1, a - 1] = count
        triple_lookup = np.zeros((MAX_NUM, MAX_NUM, MAX_NUM), dtype=np.float32)
        for (a, b, c), count in triple_counts.items():
            triple_lookup[a - 1, b - 1, c - 1] = count

        def score_batch(rows):
            idx = rows - 1
            scores = np.log(probs[idx] + 1e-12).sum(axis=1)
            scores += 0.55 * np.log(stab[idx] + 1e-12).sum(axis=1)
            scores += 0.35 * np.log(analog[idx] + 1e-12).sum(axis=1)
            for i, j in combinations(range(6), 2):
                scores += 0.012 * np.log1p(pair_lookup[idx[:, i], idx[:, j]])
            for i, j, k in combinations(range(6), 3):
                scores += 0.006 * np.log1p(
                    triple_lookup[idx[:, i], idx[:, j], idx[:, k]]
                )

            evens = (rows % 2 == 0).sum(axis=1)
            lows = (rows <= 24).sum(axis=1)
            totals = rows.sum(axis=1)
            consecutive = (np.diff(rows, axis=1) == 1).sum(axis=1)
            spreads = rows[:, -1] - rows[:, 0]
            scores -= np.where((evens == 0) | (evens == 6), 0.35, 0.0)
            scores -= np.where((lows == 0) | (lows == 6), 0.30, 0.0)
            scores -= np.where((totals < 75) | (totals > 225), 0.10, 0.0)
            scores -= 0.18 * np.maximum(0, consecutive - 2)
            scores -= np.where(spreads < 15, 0.18, 0.0)
            scores -= np.where((rows <= 31).sum(axis=1) >= 5, 0.08, 0.0)

            valid = (
                (evens >= 1) & (evens <= 5) &
                (lows >= 1) & (lows <= 5) &
                (totals >= 45) & (totals <= 255) &
                (consecutive <= 3) & (spreads >= 10)
            )
            return scores, valid

        best, best_s = None, -float("inf")
        done = 0
        batch = 50000
        while done < iterations:
            b = min(batch, iterations - done)
            u = rng.random((b, MAX_NUM))
            logits = np.log(np.clip(probs, 1e-12, None)) - np.log(
                -np.log(np.clip(u, 1e-12, 1 - 1e-12))
            )
            top6 = np.argpartition(-logits, 6, axis=1)[:, :6]
            top6 = np.sort(top6 + 1, axis=1)
            scores, valid = score_batch(top6)
            if np.any(valid):
                valid_indices = np.flatnonzero(valid)
                local = valid_indices[int(np.argmax(scores[valid]))]
                if float(scores[local]) > best_s:
                    best_s = float(scores[local])
                    best = tuple(int(x) for x in top6[local])

            done += b
            self.progress(
                62 + 16 * (done / max(1, iterations)),
                f"Semanal: Monte Carlo {done:,}/{iterations:,}",
            )

        det = self.deterministic_best(probs, details)
        s_det = self.combo_score(det, probs, details)
        if s_det > best_s:
            best, best_s = det, s_det
        return best, best_s

    def backtest(self, max_weeks=80):
        n = len(self.draws)
        if n < 1100:
            return {"tests": 0, "reason": "histórico insuficiente"}

        end = n - 8
        start = max(1000, end - max_weeks * 7)
        anchors = list(range(start, end, 7))[-max_weeks:]
        if not anchors:
            return {"tests": 0}

        self.progress(80, f"Semanal: backtesting {len(anchors)} semanas...")
        hits = []
        best_hits = []
        exact = 0
        for pos, anchor in enumerate(anchors):
            train = self.data[:anchor]
            eng = WeeklyEngine(train)
            p, det = eng.robust_vector()
            combo, _ = eng.best_combination(p, det, iterations=200_000)
            future = self.data[anchor:anchor + 7]
            week_hits = [len(set(combo) & set(d)) for _, d in future]
            hits.extend(week_hits)
            best_hits.append(max(week_hits))
            if max(week_hits) == 6:
                exact += 1
            if pos % max(1, len(anchors) // 8) == 0:
                self.progress(
                    80 + 12 * (pos / max(1, len(anchors))),
                    f"Semanal: backtest {pos+1}/{len(anchors)}",
                )

        arr = np.asarray(hits, dtype=float)
        bh = np.asarray(best_hits, dtype=float)
        exp_rand = NUMBERS_PER_DRAW * 6 / MAX_NUM
        result = {
            "tests": int(len(anchors)),
            "draws_evaluated": int(len(arr)),
            "mean_hits_per_draw": float(arr.mean()) if len(arr) else 0.0,
            "median_hits_per_draw": float(np.median(arr)) if len(arr) else 0.0,
            "p_ge_2": float(np.mean(arr >= 2)) if len(arr) else 0.0,
            "p_ge_3": float(np.mean(arr >= 3)) if len(arr) else 0.0,
            "p_ge_4": float(np.mean(arr >= 4)) if len(arr) else 0.0,
            "mean_best_hits_week": float(bh.mean()) if len(bh) else 0.0,
            "p_best_ge_3": float(np.mean(bh >= 3)) if len(bh) else 0.0,
            "p_best_ge_4": float(np.mean(bh >= 4)) if len(bh) else 0.0,
            "six_hit_weeks": int(exact),
            "random_expected_hits_per_draw": float(exp_rand),
            "lift_over_random": float(arr.mean() / exp_rand) if exp_rand else 0.0,
        }
        log.log(
            f"Backtest semanal: {result['tests']} semanas · "
            f"media={result['mean_hits_per_draw']:.3f} · "
            f"mejor_semana={result['mean_best_hits_week']:.3f} · "
            f"lift={result['lift_over_random']:.3f}",
            "AUDIT",
        )
        return result

    def predict(self, mc_iterations=MC_ITERATIONS_SEMANAL, run_backtest=True):
        if len(self.draws) < 1000:
            raise ValueError(
                f"Semanal requiere >=1000 sorteos; tienes {len(self.draws)}."
            )
        self.progress(3, "Semanal: iniciando análisis robusto...")
        probs, details = self.robust_vector()
        self.progress(60, "Semanal: vector robusto consolidado")
        combo, score = self.best_combination(probs, details, iterations=mc_iterations)
        self.progress(90, "Semanal: validando estabilidad de la combinación")
        backtest = self.backtest(max_weeks=60) if run_backtest else {"tests": 0}

        stab_vals = [details["stability"][n - 1] for n in combo]
        stab_score = float(np.mean(stab_vals) / (np.max(details["stability"]) + 1e-12))
        conc = float(np.mean([probs[n - 1] for n in combo]) /
                     (np.mean(np.sort(probs)[-6:]) + 1e-12))

        self.progress(99, "Semanal: análisis terminado")

        return {
            "combination": tuple(sorted(int(x) for x in combo)),
            "score": float(score),
            "probabilities": probs,
            "details_keys": list(details.keys()),
            "stability_score": max(0.0, min(1.0, stab_score)),
            "concentration_score": max(0.0, min(1.0, conc)),
            "analog_reliability": float(details.get("analog_reliability", 0.0)),
            "analog_count": int(details.get("analog_count", 0)),
            "pair_count": int(details.get("pair_count", 0)),
            "triple_count": int(details.get("triple_count", 0)),
            "backtest": backtest,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "period_days": 7,
        }
    
# ================================================================
# PARTE 9 — ORQUESTADOR
# ================================================================

class FuxingPredictor:
    """Predictor principal. Mantiene el estado entre fases."""

    def __init__(self):
        self.data = None
        self.X_seq = None
        self.X_feat = None
        self.Y = None

        self.stat_engine = None
        self.tree_ensemble = None
        self.neural = None
        self.bayes = None

        self.calibrator = None
        self.conformal = None
        self.weights = None

        self.training_report = {}
        self.validation_report = {}
        self.backtest_report = {}

        # FIX: cache de componentes de calibración (evita recalcularlos
        # dos veces sobre el mismo rango en calibrate_and_optimize+validate).
        self._cached_cal_components = None

    def prepare(self, data):
        log.banner("FASE 1 — PREPARACIÓN DE DATOS Y FEATURES", "=")
        self.data = sorted(data, key=lambda x: x[0])
        log.log(f"Sorteos cargados: {len(self.data):,}", "DATA")
        log.log(
            f"Rango: {self.data[0][0]:%d/%m/%Y} -> {self.data[-1][0]:%d/%m/%Y}",
            "DATA",
        )

        dataset_audit(self.data)
        randomness_audit(self.data)

        self.X_seq, self.X_feat, self.Y = build_dataset(self.data, SEQ_LEN)

        feature_audit(self.X_feat, self.Y)

        self.stat_engine = StatisticalEngine(self.data)

        log.ok(
            f"Preparación completa: {len(self.X_seq):,} muestras, "
            f"{self.X_feat.shape[1]} features por muestra"
        )

    def train_models(self):
        log.banner("FASE 2 — ENTRENAMIENTO DE MODELOS", "=")

        n = len(self.X_feat)
        if n < 500:
            raise ValueError(f"Muestras insuficientes para entrenar: {n}")

        split = int(n * 0.70)
        val_end = int(n * 0.75)
        split = max(300, min(split, n - 200))
        val_end = max(split + 30, min(val_end, n))

        Xtr, Xv = self.X_feat[:split], self.X_feat[split:val_end]
        Ytr, Yv = self.Y[:split], self.Y[split:val_end]

        log.log(
            f"Split temporal: train={len(Xtr):,} val={len(Xv):,} "
            f"(sin mezcla aleatoria)",
            "MODEL",
        )

        log.log("Entrenando ensemble de árboles...", "MODEL")
        self.tree_ensemble = TreeEnsemble()
        tree_report = self.tree_ensemble.fit(
            Xtr, Ytr, Xv, Yv,
            progress_callback=lambda p, m: log.progress(p, m),
        )
        log.ok(
            f"Árboles: {tree_report.get('rows', 0):,} filas · "
            f"XGB={tree_report.get('xgboost')} "
            f"LGBM={tree_report.get('lightgbm')}"
        )

        log.log("Entrenando ensemble de Transformers...", "MODEL")
        self.neural = NeuralEnsemble(NEURAL_ENSEMBLE_SIZE)
        nn_report = self.neural.fit(
            self.X_seq, self.X_feat, self.Y,
            progress_callback=lambda p, m: log.progress(p, m),
        )
        if nn_report.get("trained"):
            log.ok(
                f"Red: {nn_report.get('ensemble_size')} modelos · "
                f"val_loss media="
                f"{np.mean([m['val_loss'] for m in nn_report.get('members', [])]):.6f}"
            )
        else:
            log.warn("Red neuronal no entrenada (muestras insuficientes).")

        self.training_report = {
            "tree": tree_report,
            "neural": nn_report,
        }

    def calibrate_and_optimize(self):
        """Optimiza pesos y calibra en bloques cronológicos separados del entrenamiento."""
        log.banner("FASE 3 — CALIBRACIÓN, CONFORMAL Y PESOS", "=")
        n = len(self.X_feat)
        if n < 500:
            self.weights = np.array([0.40, 0.15, 0.30, 0.15], dtype=float)
            log.warn("Muestras insuficientes para optimización temporal; se usan pesos base.")
            return

        opt_start, opt_end = int(n * 0.75), int(n * 0.80)
        cal_start, cal_end = int(n * 0.80), int(n * 0.85)
        conf_start, conf_end = int(n * 0.85), int(n * 0.90)

        log.log(f"Bloques temporales: optimización={opt_start}:{opt_end}, "
                f"calibración={cal_start}:{cal_end}, conformal={conf_start}:{conf_end}",
                "MODEL")

        stat_opt, logistic_opt, tree_opt, nn_opt = self._compute_components(
            range(opt_start, opt_end), "optimización de pesos"
        )
        Y_opt = self.Y[opt_start:opt_end]
        if stat_opt is not None and len(Y_opt):
            optimizer = WeightOptimizer()
            self.weights = optimizer.optimize(
                Y_opt, [stat_opt, logistic_opt, tree_opt, nn_opt],
                progress_callback=lambda p, m: log.progress(p, m),
            )
            self.training_report["weight_optimization"] = optimizer.report
        else:
            self.weights = np.array([0.40, 0.15, 0.30, 0.15], dtype=float)

        log.log("Ajustando calibración en un bloque posterior e independiente...", "MODEL")
        stat_cal, logistic_cal, tree_cal, nn_cal = self._compute_components(
            range(cal_start, cal_end), "calibración"
        )
        Y_cal = self.Y[cal_start:cal_end]
        P_cal = self._combine([stat_cal, logistic_cal, tree_cal, nn_cal], weights=self.weights)
        self.calibrator = IsotonicCalibrator()
        if P_cal is not None and len(Y_cal):
            self.calibrator.fit(P_cal, Y_cal)

        log.log("Ajustando conformal en un bloque distinto de calibración...", "MODEL")
        stat_conf, logistic_conf, tree_conf, nn_conf = self._compute_components(
            range(conf_start, conf_end), "conformal"
        )
        Y_conf = self.Y[conf_start:conf_end]
        P_conf = self._combine(
            [stat_conf, logistic_conf, tree_conf, nn_conf], weights=self.weights
        )
        if P_conf is not None and len(Y_conf):
            if self.calibrator is not None and self.calibrator.fitted:
                P_conf = self.calibrator.transform(P_conf)
            self.conformal = ConformalPredictor(alpha=0.10)
            self.conformal.fit(normalize_rows(P_conf), Y_conf)

        log.ok("Pesos finales: " + ", ".join(f"{x:.3f}" for x in self.weights))

    def _compute_components(self, index_range, label):
        idx = list(index_range)
        if not idx:
            return None, None, None, None

        stat_rows = []
        for j in idx:
            data_index = SEQ_LEN - 1 + j
            if data_index >= len(self.data) - 1:
                break
            train = self.data[:data_index + 1]
            p, _ = StatisticalEngine(train).probability_vector()
            stat_rows.append(normalize_mass(p))
            if len(stat_rows) % 50 == 0:
                log.progress(
                    45.0 + 8.0 * (len(stat_rows) / len(idx)),
                    f"{label}: estadístico {len(stat_rows)}/{len(idx)}",
                )
        stat = np.asarray(stat_rows, dtype=float) if stat_rows else None

        Xf = self.X_feat[idx[:len(stat_rows)]]
        Xs = self.X_seq[idx[:len(stat_rows)]]

        logistic = tree = nn = None
        if self.tree_ensemble is not None and self.tree_ensemble.fitted:
            log_rows = []
            for i in range(len(Xf)):
                a, b = self.tree_ensemble.predict_components(Xf[i:i + 1])
                if a is not None:
                    log_rows.append((a[0], b[0]))
            if log_rows and len(log_rows) == len(Xf):
                logistic = np.array([x[0] for x in log_rows])
                tree = np.array([x[1] for x in log_rows])
            else:
                logistic = np.full((len(Xf), MAX_NUM), 1.0 / MAX_NUM)
                tree = np.full((len(Xf), MAX_NUM), 1.0 / MAX_NUM)
                if log_rows:
                    logistic[:len(log_rows)] = np.array([x[0] for x in log_rows])
                    tree[:len(log_rows)] = np.array([x[1] for x in log_rows])

        if self.neural is not None and self.neural.fitted:
            nn = self.neural.predict(Xs, Xf)

        return stat, logistic, tree, nn

    @staticmethod
    def _combine(components, weights):
        w = np.asarray(weights, dtype=float)
        valid = [(i, normalize_rows(p)) for i, p in enumerate(components) if p is not None]
        if not valid:
            return None
        w_valid = np.array([w[i] for i, _ in valid])
        w_valid = w_valid / (w_valid.sum() + 1e-12)
        out = np.zeros_like(valid[0][1])
        for ww, (_, p) in zip(w_valid, valid):
            out += ww * p
        return normalize_rows(out)

    def validate(self, run_walk_forward=True, run_backtest=True):
        log.banner("FASE 4 — VALIDACIÓN", "=")
        n = len(self.X_feat)
        test_start = int(n * 0.90)
        test_end = int(n * 0.95)
        if test_end <= test_start:
            log.warn("No hay bloque temporal suficiente para la auditoría final.")
        else:
            stat_test, logistic_test, tree_test, nn_test = self._compute_components(
                range(test_start, test_end), "auditoría final no usada para ajuste"
            )
            Y_test = self.Y[test_start:test_end]
            P_test = self._combine(
                [stat_test, logistic_test, tree_test, nn_test],
                weights=self.weights if self.weights is not None
                        else [0.40, 0.15, 0.30, 0.15],
            )
            if P_test is not None and len(Y_test):
                P_eval = self.calibrator.transform(P_test) if (
                    self.calibrator is not None and self.calibrator.fitted
                ) else P_test
                self.validation_report["held_out_test"] = {
                    "start_index": test_start,
                    "end_index": test_end,
                    "samples": int(len(Y_test)),
                    "brier": float(np.mean((Y_test - P_eval) ** 2)),
                    "note": "Bloque cronológico reservado; no usado para entrenar, optimizar pesos ni calibrar.",
                }
                self.validation_report["diebold_mariano"] = diebold_mariano(P_eval, Y_test)

        if run_walk_forward:
            # Este backtest usa exclusivamente el motor estadístico. No se presenta
            # como validación del ensemble de árboles/Transformer.
            def train_fn(Xs, Xf, Y):
                return {"train_size": len(Y), "model": "statistical_only"}

            def predict_fn(model, Xs, Xf, test_start_idx):
                rows = []
                for j in range(len(Xf)):
                    data_index = min(SEQ_LEN - 1 + int(test_start_idx) + j,
                                     len(self.data) - 2)
                    prefix = self.data[:data_index + 1]
                    if len(prefix) < 30:
                        return None
                    p, _ = StatisticalEngine(prefix).probability_vector()
                    rows.append(normalize_mass(p))
                return np.asarray(rows, dtype=float) if rows else None

            wf = NestedWalkForward(n_splits=4, purge=SEQ_LEN, embargo=7)
            wf_report = wf.run(
                self.X_seq, self.X_feat, self.Y, train_fn, predict_fn,
                progress_callback=lambda p, m: log.progress(p, m),
            )
            wf_report["scope"] = "motor estadístico únicamente; no valida el ensemble completo"
            self.validation_report["walk_forward_statistical_only"] = wf_report

        if run_backtest:
            bt = DailyBacktester(self.data, top_k=12, start=800)
            self.backtest_report["daily_statistical_only"] = bt.run(
                progress_callback=lambda p, m: log.progress(p, m)
            )

    def predict(self, mc_iterations=MC_ITERATIONS, n_combinations=15):
        log.banner("FASE 5 — PREDICCIÓN FINAL", "=")

        log.log("Calculando componentes para el próximo sorteo...", "PRED")
        seq_now, feat_now = self._future_sample()

        stat_now = normalize_mass(self.stat_engine.probability_vector()[0])[None, :]

        logistic_now = tree_now = nn_now = None
        if self.tree_ensemble is not None and self.tree_ensemble.fitted:
            a, b = self.tree_ensemble.predict_components(feat_now)
            if a is not None:
                logistic_now = a
                tree_now = b
        if self.neural is not None and self.neural.fitted:
            nn_now = self.neural.predict(seq_now, feat_now)

        combined = self._combine(
            [stat_now, logistic_now, tree_now, nn_now],
            weights=self.weights if self.weights is not None
                    else [0.40, 0.15, 0.30, 0.15],
        )[0]

        if self.calibrator is not None and self.calibrator.fitted:
            combined = self.calibrator.transform(combined)
        combined = normalize_mass(combined)

        log.log("Calculando intervalos de credibilidad bayesianos...", "PRED")
        self.bayes = EmpiricalBayes(self.data, n_bootstrap=1000, n_dirichlet=3000)
        self.bayes.fit(progress_callback=lambda p, m: log.progress(p, m))

        combined = normalize_mass(0.85 * combined + 0.15 * self.bayes.mean)

        log.log(
            f"Monte Carlo vectorizado ({mc_iterations:,} iteraciones)...",
            "PRED",
        )
        mc = VectorizedMonteCarlo(n_iterations=mc_iterations, seed=ULTIMATE_SEED)
        mc_freq = mc.sample(
            combined,
            constraints=True,
            progress_callback=lambda p, m: log.progress(p, m),
        )

        final_p = normalize_mass(0.85 * combined + 0.15 * mc_freq)

        log.log("Generando pool de combinaciones candidatas...", "PRED")
        mc_gen = VectorizedMonteCarlo(n_iterations=200_000, seed=ULTIMATE_SEED + 11)
        pool = mc_gen.top_combinations(
            final_p,
            n=n_combinations,
            n_candidates=300_000,
            progress_callback=lambda p, m: log.progress(p, m),
        )

        log.log("Seleccionando cartera diversificada...", "PRED")
        portfolio = BetPortfolio(self.data, final_p)
        selected = portfolio.select(
            pool,
            n=n_combinations,
            max_overlap=4,
            progress_callback=lambda p, m: log.progress(p, m),
        )

        conformal_set = None
        if self.conformal is not None and self.conformal.fitted:
            conformal_set = self.conformal.predict_set(normalize_rows(final_p))

        top_numbers = (np.argsort(final_p)[::-1] + 1)[:20].tolist()

        result = {
            "combinations": [tuple(sorted(c)) for c in selected],
            "primary_combination": tuple(sorted(selected[0])) if selected else None,
            "probabilities": final_p,
            "top_numbers": top_numbers,
            "bayes_mean": self.bayes.mean if self.bayes is not None else None,
            "bayes_q05": self.bayes.q05 if self.bayes is not None else None,
            "bayes_q95": self.bayes.q95 if self.bayes is not None else None,
            "bayes_p_top6": self.bayes.p_top6 if self.bayes is not None else None,
            "conformal_set": sorted(conformal_set) if conformal_set else None,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "training_report": self.training_report,
            "validation_report": self.validation_report,
            "backtest_report": self.backtest_report,
            "calibration_report": (
                self.calibrator.report if self.calibrator is not None else None
            ),
        }
        return result

    def _future_sample(self):
        """Construye la muestra que predice el sorteo SIGUIENTE."""
        draws = [tuple(d) for _, d in self.data]
        cum = FeatureEngineeringV3._build_prefix_counts(draws)
        i = len(draws) - 1
        seq = np.asarray(
            [one_hot_draw(draws[k]) for k in range(i - SEQ_LEN + 1, i + 1)],
            dtype=np.float32,
        )[None, :, :]
        feat = FeatureEngineeringV3.number_features(
            self.data, i, cum=cum, draws=draws,
        )
        return seq, np.asarray(feat, dtype=np.float32)[None, :]


# ================================================================
# PARTE 10 — CLI ORQUESTADOR + MAIN
# ================================================================

def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, tuple):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    return obj


def _safe_json_write(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _fmt_combo(combo):
    return " - ".join(f"{int(n):02d}" for n in combo)


def run_fuxing_v3(mode="both"):
    overall_start = time.time()
    print_system_status()

    # 1. HISTÓRICO
    log.banner("PASO 1/6 — HISTÓRICO COMPLETO 1988 -> ACTUAL", "=")
    scraper = DataScraper(start_year=START_YEAR, end_year=datetime.now().year)
    data = scraper.scrape_with_cache(force_refresh=False)
    if not data:
        log.error("Histórico vacío. Abortando.")
        return None
    log.ok(
        f"Histórico listo: {len(data):,} sorteos | "
        f"{data[0][0]:%d/%m/%Y} -> {data[-1][0]:%d/%m/%Y}"
    )

    # 2. PREPARACIÓN
    predictor = FuxingPredictor()
    predictor.prepare(data)

    # 3. ENTRENAMIENTO
    predictor.train_models()

    # 4. CALIBRACIÓN + PESOS
    predictor.calibrate_and_optimize()

    # 5. VALIDACIÓN
    predictor.validate(run_walk_forward=True, run_backtest=True)

    # 6. PREDICCIÓN DIARIA + SEMANAL
    log.banner("PASO 6/6 — PREDICCIÓN DIARIA + SEMANAL", "=")

    daily_result = {}
    weekly_result = {}

    def run_daily():
        log.log(">>> DIARIA: iniciando predicción", "PRED")
        try:
            r = predictor.predict(
                mc_iterations=MC_ITERATIONS,
                n_combinations=15,
            )
            daily_result.update(r)
            log.ok(">>> DIARIA: completada")
        except Exception as e:
            log.error(f">>> DIARIA: {type(e).__name__}: {e}")
            traceback.print_exc()

    def run_weekly():
        log.log(">>> SEMANAL: iniciando predicción", "PRED")
        try:
            engine = WeeklyEngine(
                data,
                progress_callback=lambda p, m: log.progress(p, m),
                model_probabilities=(
                    daily_result.get("probabilities")
                    if daily_result else None
                ),
            )
            r = engine.predict(
                mc_iterations=MC_ITERATIONS_SEMANAL,
                run_backtest=True,
            )
            weekly_result.update(r)
            log.ok(">>> SEMANAL: completada")
        except Exception as e:
            log.error(f">>> SEMANAL: {type(e).__name__}: {e}")
            traceback.print_exc()

    if mode in ("daily", "both"):
        run_daily()
    if mode in ("weekly", "both"):
        run_weekly()

    # No se permite publicar como ejecución correcta una predicción ausente o inválida.
    def _valid_combo(combo):
        try:
            values = [int(x) for x in combo]
            return len(values) == 6 and len(set(values)) == 6 and all(1 <= x <= 49 for x in values)
        except Exception:
            return False

    if mode in ("daily", "both") and not _valid_combo(daily_result.get("primary_combination")):
        raise RuntimeError("La predicción diaria no se ha generado o no contiene 6 números únicos entre 1 y 49.")
    if mode in ("weekly", "both") and not _valid_combo(weekly_result.get("combination")):
        raise RuntimeError("La predicción semanal no se ha generado o no contiene 6 números únicos entre 1 y 49.")

    total_elapsed = time.time() - overall_start

    # RESULTADO FINAL
    log.banner("RESULTADO FINAL", "=")

    daily_combo = daily_result.get("primary_combination")
    weekly_combo = weekly_result.get("combination")

    log.log("", "INFO")
    log.log("=" * 60, "PRED")
    log.log("  COMBINACIÓN DIARIA RECOMENDADA (próximo sorteo)", "PRED")
    log.log("=" * 60, "PRED")
    if daily_combo:
        log.log(f"  >>>  {_fmt_combo(daily_combo)}  <<<", "OK")
    else:
        log.warn("  No se pudo generar combinación diaria.")
    log.log("=" * 60, "PRED")
    log.log("", "INFO")

    top20 = daily_result.get("top_numbers", [])
    if top20:
        log.log(
            "Top-20 números (probabilidad descendente): "
            + " ".join(f"{n:02d}" for n in top20),
            "PRED",
        )

    portfolio = daily_result.get("combinations", [])
    if len(portfolio) > 1:
        log.log("Cartera completa (15 combinaciones diversas):", "PRED")
        for i, c in enumerate(portfolio, 1):
            log.log(f"  #{i:02d}  {_fmt_combo(c)}", "PRED")

    conf_set = daily_result.get("conformal_set")
    if conf_set:
        log.log(
            f"Conjunto conformal (90% cobertura): "
            f"{len(conf_set)} números -> {conf_set}",
            "PRED",
        )

    bayes_p6 = daily_result.get("bayes_p_top6")
    if bayes_p6 is not None:
        top_bayes = np.argsort(bayes_p6)[::-1][:10] + 1
        log.log(
            "Top-10 números por frecuencia posterior de aparecer entre los 6 más altos del modelo Dirichlet: "
            + " ".join(f"{n:02d}({bayes_p6[n-1]:.2f})" for n in top_bayes),
            "PRED",
        )

    log.log("", "INFO")
    log.log("=" * 60, "PRED")
    log.log("  COMBINACIÓN SEMANAL RECOMENDADA (7 sorteos)", "PRED")
    log.log("=" * 60, "PRED")
    if weekly_combo:
        log.log(f"  >>>  {_fmt_combo(weekly_combo)}  <<<", "OK")
    else:
        log.warn("  No se pudo generar combinación semanal.")
    log.log("=" * 60, "PRED")
    log.log("", "INFO")

    if weekly_result:
        log.log(
            f"Estabilidad: {weekly_result.get('stability_score', 0)*100:.2f}% · "
            f"Concentración: {weekly_result.get('concentration_score', 0)*100:.2f}% · "
            f"Analogías: {weekly_result.get('analog_count', 0)}",
            "PRED",
        )
        bt = weekly_result.get("backtest", {})
        if bt.get("tests"):
            log.log(
                f"Backtest semanal: {bt['tests']} semanas · "
                f"media hits/sorteo={bt.get('mean_hits_per_draw', 0):.3f} · "
                f"mejor semana={bt.get('mean_best_hits_week', 0):.3f} · "
                f"lift={bt.get('lift_over_random', 0):.3f}",
                "AUDIT",
            )

    log.log("", "INFO")
    log.log("Validación rigurosa:", "AUDIT")
    wf = predictor.validation_report.get("walk_forward_statistical_only", {})
    if wf.get("tests"):
        log.log(
            f"  Walk-forward estadístico: {wf['tests']} splits · "
            f"Brier={wf.get('mean_brier', 0):.6f} · "
            f"hits@12={wf.get('mean_hits_top12', 0):.3f}",
            "AUDIT",
        )
    dm = predictor.validation_report.get("diebold_mariano", {})
    if dm.get("n"):
        log.log(
            f"  Diebold-Mariano: DM={dm.get('dm_stat', 0):+.3f} "
            f"p={dm.get('p_value', 1):.4f} · "
            f"modelo_mejor={dm.get('model_better')}",
            "AUDIT",
        )
    bt_d = predictor.backtest_report.get("daily_statistical_only", {})
    if bt_d.get("tests"):
        log.log(
            f"  Backtest diario: {bt_d['tests']} sorteos · "
            f"media={bt_d.get('mean_hits_top12', 0):.3f} · "
            f"lift={bt_d.get('lift_over_random', 0):.3f}",
            "AUDIT",
        )

    log.log("", "INFO")
    log.log(f"TIEMPO TOTAL: {total_elapsed:.1f}s ({total_elapsed/60:.2f} min)", "INFO")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "execution": {
            "historical_start": START_YEAR,
            "historical_end": datetime.now().year,
            "historical_draws": len(data),
            "first_date": data[0][0].strftime("%Y-%m-%d"),
            "last_date": data[-1][0].strftime("%Y-%m-%d"),
            "total_elapsed_seconds": total_elapsed,
        },
        "daily": {
            "primary_combination": daily_combo,
            "combinations": daily_result.get("combinations", []),
            "top_numbers": top20,
            "conformal_set": conf_set,
            "bayes_p_top6": (
                bayes_p6.tolist() if bayes_p6 is not None else None
            ),
            "weights": daily_result.get("weights"),
        },
        "weekly": {
            "combination": weekly_combo,
            "stability_score": weekly_result.get("stability_score"),
            "concentration_score": weekly_result.get("concentration_score"),
            "analog_count": weekly_result.get("analog_count"),
            "backtest": weekly_result.get("backtest"),
        },
        "validation": predictor.validation_report,
        "backtest_daily_statistical_only": predictor.backtest_report.get("daily_statistical_only"),
        "training": predictor.training_report,
    }
    _safe_json_write(RESULT_FILE, payload)
    _safe_json_write(RUN_REPORT_FILE, payload)
    log.ok(f"Resultado guardado en: {RESULT_FILE}")

    return payload


def main():
    parser = argparse.ArgumentParser(description="Fuxing Bonoloto")
    parser.add_argument(
        "--mode", choices=("daily", "weekly", "both"), default="both",
        help="Ejecutar sólo predicción diaria, sólo semanal o ambas (por defecto: ambas).",
    )
    args = parser.parse_args()
    try:
        run_fuxing_v3(mode=args.mode)
    except KeyboardInterrupt:
        log.warn("Proceso interrumpido por el usuario.")
        raise SystemExit(130)
    except Exception as e:
        log.error(f"FUXING v3 terminó con error: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        log.close()


if __name__ == "__main__":
    main()