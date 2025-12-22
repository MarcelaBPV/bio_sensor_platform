# raman_processing.py
# -*- coding: utf-8 -*-

import io
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd

from scipy.signal import find_peaks, savgol_filter, medfilt
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve

# ---------------------------------------------------------------------
# DEPENDÊNCIAS OPCIONAIS
# ---------------------------------------------------------------------
try:
    from lmfit.models import VoigtModel, GaussianModel, LorentzianModel
    LMFIT_AVAILABLE = True
except Exception:
    LMFIT_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except Exception:
    H5PY_AVAILABLE = False

# ---------------------------------------------------------------------
# MAPA MOLECULAR E REGRAS
# ---------------------------------------------------------------------
MOLECULAR_MAP = [
# =========================
# COMPONENTES DO SANGUE
# =========================    
{"range": (720, 735), "group": "Adenina / nucleotídeos (DNA/RNA)"},
{"range": (748, 755), "group": "Citocromo c / heme"},
{"range": (780, 790), "group": "DNA/RNA – ligações fosfato"},
{"range": (820, 850), "group": "Proteínas – C–C / tirosina"},
{"range": (935, 955), "group": "Proteínas – esqueleto α-hélice"},
{"range": (1000, 1008), "group": "Fenilalanina"},
{"range": (1120, 1135), "group": "Lipídios – C–C estiramento"},
{"range": (1240, 1280), "group": "Amida III (proteínas)"},
{"range": (1300, 1315), "group": "Lipídios – CH2 torção"},
{"range": (1335, 1365), "group": "Nucleotídeos / triptofano"},
{"range": (1440, 1475), "group": "Lipídios – CH2 deformação"},
{"range": (1540, 1580), "group": "Amida II"},
{"range": (1600, 1620), "group": "Tirosina / fenilalanina"},
{"range": (1650, 1670), "group": "Amida I (proteínas, C=O)"},
{"range": (2850, 2885), "group": "Lipídios – CH2 simétrico"},
{"range": (2920, 2960), "group": "Lipídios / proteínas – CH3"},

# =========================
# PAPEL / CELULOSE
# =========================
{"range": (375, 385), "group": "Celulose – modos coletivos"},
{"range": (435, 460), "group": "Celulose – deformação C–O–C"},
{"range": (895, 905), "group": "Celulose – β-glicosídica"},
{"range": (1030, 1060), "group": "Celulose – estiramento C–O"},
{"range": (1090, 1120), "group": "Celulose – C–O–C assimétrico"},
{"range": (1330, 1380), "group": "Celulose – CH / OH"},
{"range": (1450, 1470), "group": "Celulose – CH2 deformação"},
{"range": (2880, 2940), "group": "Celulose – CH"}

# =========================
# PRATA / SERS
# =========================
{"range": (180, 260), "group": "Prata – fônons / plasmon"},
{"range": (400, 430), "group": "Interação Ag–N"},
{"range": (520, 550), "group": "Interação Ag–S / Ag–O"},
{"range": (1000, 1025), "group": "Moléculas adsorvidas (SERS hotspot)"},
{"range": (1580, 1620), "group": "Aromáticos intensificados por SERS"}
]

DISEASE_RULES = [

    # =========================================================
    # ALTERAÇÕES RELACIONADAS À HEMOGLOBINA / HEME
    # =========================================================
    {
        "name": "Alteração hemoglobina",
        "description": (
            "Padrão espectral compatível com alterações estruturais "
            "ou conformacionais da hemoglobina, envolvendo o grupo heme "
            "e porfirinas."
        ),
        "groups_required": [
            "Hemoglobina / porfirinas",
            "Citocromo c / heme"
        ],
    },

    # =========================================================
    # ALTERAÇÕES PROTEICAS GERAIS
    # =========================================================
    {
        "name": "Alteração proteica",
        "description": (
            "Padrão compatível com alterações na estrutura secundária "
            "e conformação de proteínas plasmáticas e celulares, "
            "especialmente em regiões de amidas."
        ),
        "groups_required": [
            "Amida I (proteínas, C=O)",
            "Amida II",
            "Amida III (proteínas)"
        ],
    },

    # =========================================================
    # ALTERAÇÕES LIPÍDICAS DE MEMBRANA
    # =========================================================
    {
        "name": "Alteração lipídica de membrana",
        "description": (
            "Padrão compatível com modificações na composição ou "
            "organização de lipídios de membrana, incluindo fosfolipídios "
            "e cadeias alifáticas."
        ),
        "groups_required": [
            "Lipídios – CH2 deformação",
            "Lipídios – CH2 torção",
            "Lipídios – C–C estiramento"
        ],
    },

    # =========================================================
    # ESTRESSE OXIDATIVO
    # =========================================================
    {
        "name": "Estresse oxidativo",
        "description": (
            "Padrão espectral compatível com processos de oxidação "
            "de proteínas, lipídios e grupos heme, frequentemente "
            "associado a inflamação sistêmica."
        ),
        "groups_required": [
            "Citocromo c / heme",
            "Aromáticos intensificados por SERS",
            "Lipídios – CH3"
        ],
    },

    # =========================================================
    # ALTERAÇÕES EM ÁCIDOS NUCLEICOS
    # =========================================================
    {
        "name": "Alteração em ácidos nucleicos",
        "description": (
            "Padrão compatível com alterações na concentração ou "
            "organização estrutural de DNA/RNA, nucleotídeos e bases "
            "nitrogenadas."
        ),
        "groups_required": [
            "DNA/RNA – ligações fosfato",
            "Adenina / nucleotídeos (DNA/RNA)"
        ],
    },

    # =========================================================
    # PERFIL INFLAMATÓRIO SISTÊMICO
    # =========================================================
    {
        "name": "Perfil inflamatório",
        "description": (
            "Padrão compatível com resposta inflamatória sistêmica, "
            "envolvendo alterações simultâneas em proteínas, lipídios "
            "e componentes celulares do sangue."
        ),
        "groups_required": [
            "Amida I (proteínas, C=O)",
            "Lipídios – CH2 deformação",
            "Citocromo c / heme"
        ],
    },

    # =========================================================
    # ALTERAÇÃO METABÓLICA (ex.: diabetes, dislipidemias)
    # =========================================================
    {
        "name": "Alteração metabólica",
        "description": (
            "Padrão espectral compatível com alterações metabólicas, "
            "incluindo modificações em lipídios, proteínas e "
            "aminoácidos aromáticos."
        ),
        "groups_required": [
            "Fenilalanina",
            "Lipídios – CH2 deformação",
            "Amida I (proteínas, C=O)"
        ],
    },
]

# ---------------------------------------------------------------------
# DATACLASS PARA PICOS
# ---------------------------------------------------------------------
@dataclass
class Peak:
    position_cm1: float
    intensity: float
    width: Optional[float] = None
    group: Optional[str] = None
    fit_params: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------------------
# LEITURA DE ESPECTROS
# ---------------------------------------------------------------------
def load_spectrum(file_like) -> Tuple[np.ndarray, np.ndarray]:
    filename = getattr(file_like, "name", "").lower()
    try:
        if filename.endswith(".txt"):
            df = pd.read_csv(file_like, sep=r"\s+", comment="#", header=None, engine="python")
        elif filename.endswith(".csv"):
            try:
                df = pd.read_csv(file_like)
            except Exception:
                file_like.seek(0)
                df = pd.read_csv(file_like, sep=";")
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_like)
        else:
            file_like.seek(0)
            df = pd.read_csv(file_like, sep=r"\s+", comment="#", header=None, engine="python")
    except Exception as e:
        raise RuntimeError(f"Erro ao ler arquivo de espectro: {e}")

    df = df.dropna(axis=1, how="all")
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        raise RuntimeError("Arquivo não contém pelo menos duas colunas numéricas.")

    x = numeric_df.iloc[:, 0].to_numpy(dtype=float)
    y = numeric_df.iloc[:, 1].to_numpy(dtype=float)
    return x, y

# ---------------------------------------------------------------------
# DESPIKE
# ---------------------------------------------------------------------
def despike(y: np.ndarray, method: str = "median", kernel_size: int = 5) -> np.ndarray:
    y = y.copy()
    if method == "median":
        y_med = medfilt(y, kernel_size=kernel_size)
        mask = np.abs(y - y_med) > 3 * np.std(y)
        y[mask] = y_med[mask]
        return y
    elif method == "median_filter_nd":
        return median_filter(y, size=kernel_size)
    elif method == "zscore":
        mu = pd.Series(y).rolling(kernel_size, center=True, min_periods=1).median().to_numpy()
        resid = y - mu
        z = np.abs(resid) / (np.std(resid) + 1e-12)
        y[z > 6.0] = mu[z > 6.0]
        return y
    else:
        return y

def _despike_metric(y0, y1):
    return np.mean(np.abs(np.diff(y1, n=2))) + 0.1 * np.mean((y0 - y1) ** 2)

def compare_despike_algorithms(y, methods=None, kernel_size=5):
    if methods is None:
        methods = ["median", "zscore", "median_filter_nd"]
    best_y = y.copy()
    best_metric = np.inf
    best_method = None
    metrics = {}
    for m in methods:
        try:
            y_d = despike(y, method=m, kernel_size=kernel_size)
            metric = _despike_metric(y, y_d)
        except Exception:
            metric = np.inf
            y_d = y.copy()
        metrics[m] = metric
        if metric < best_metric:
            best_metric = metric
            best_y = y_d
            best_method = m
    return best_y, best_method, metrics

# ---------------------------------------------------------------------
# BASELINE + PRÉ-PROCESSAMENTO
# ---------------------------------------------------------------------
def baseline_als(y, lam=1e5, p=0.01, niter=10):
    y = np.asarray(y, dtype=float)
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = (W + lam * D.T @ D).tocsc()
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def preprocess_spectrum(
    x, y,
    despike_method="auto_compare",
    smooth=True,
    window_length=9,
    polyorder=3,
    baseline_method="als",
    normalize=True,
):
    meta = {}
    y_proc = y.astype(float).copy()

    if despike_method == "auto_compare":
        y_proc, best, metrics = compare_despike_algorithms(y_proc)
        meta["despike"] = {"method": best, "metrics": metrics}
    elif despike_method:
        y_proc = despike(y_proc, method=despike_method)

    if smooth:
        window_length = min(window_length, len(y_proc) - 1)
        if window_length % 2 == 0:
            window_length += 1
        y_proc = savgol_filter(y_proc, window_length, polyorder)

    if baseline_method == "als":
        base = baseline_als(y_proc)
        y_proc = y_proc - base

    if normalize:
        ymin, ymax = np.min(y_proc), np.max(y_proc)
        if ymax > ymin:
            y_proc = (y_proc - ymin) / (ymax - ymin)

    return x, y_proc, meta

# ---------------------------------------------------------------------
# PICOS
# ---------------------------------------------------------------------
def detect_peaks(x, y, height=0.05, distance=5, prominence=0.02):
    idx, _ = find_peaks(y, height=height, distance=distance, prominence=prominence)
    return [Peak(float(x[i]), float(y[i])) for i in idx]

def map_peaks_to_molecular_groups(peaks):
    for p in peaks:
        p.group = None
        for item in MOLECULAR_MAP:
            if item["range"][0] <= p.position_cm1 <= item["range"][1]:
                p.group = item["group"]
                break
    return peaks

def infer_diseases(peaks):
    groups = {p.group for p in peaks if p.group}
    results = []
    for rule in DISEASE_RULES:
        required = set(rule["groups_required"])
        present = len(required & groups)
        if present > 0:
            results.append({
                "name": rule["name"],
                "score": round(100 * present / len(required), 1),
                "description": rule["description"],
            })
    return sorted(results, key=lambda r: r["score"], reverse=True)

# ---------------------------------------------------------------------
# FUNÇÃO DE ALTO NÍVEL (USADA PELO APP)
# ---------------------------------------------------------------------
def process_raman_spectrum_with_groups(
    file_like,
    preprocess_kwargs=None,
    peak_height=0.05,
    peak_distance=5,
    peak_prominence=0.02,
):
    if preprocess_kwargs is None:
        preprocess_kwargs = {}

    x_raw, y_raw = load_spectrum(file_like)
    x_proc, y_proc, meta = preprocess_spectrum(x_raw, y_raw, **preprocess_kwargs)

    peaks = detect_peaks(x_proc, y_proc, peak_height, peak_distance, peak_prominence)
    peaks = map_peaks_to_molecular_groups(peaks)
    diseases = infer_diseases(peaks)

    return {
        "x_raw": x_raw,
        "y_raw": y_raw,
        "x_proc": x_proc,
        "y_proc": y_proc,
        "peaks": peaks,
        "diseases": diseases,
        "meta": meta,
    }

# ---------------------------------------------------------------------
# EXPORTAÇÃO HDF5 (NeXus-like)
# ---------------------------------------------------------------------
def save_to_nexus_bytes(x, y, metadata):
    if not H5PY_AVAILABLE:
        raise RuntimeError("h5py não instalado.")
    bio = io.BytesIO()
    with h5py.File(bio, "w") as f:
        e = f.create_group("entry")
        d = e.create_group("data")
        d.create_dataset("wavenumber", data=x)
        d.create_dataset("intensity", data=y)
        m = e.create_group("metadata")
        for k, v in metadata.items():
            m.attrs[str(k)] = str(v)
    bio.seek(0)
    return bio.read()
