# raman_processing.py
# -*- coding: utf-8 -*-

import io
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd

from scipy.signal import find_peaks, savgol_filter, medfilt
from scipy.ndimage import median_filter
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
# MAPA MOLECULAR
# ---------------------------------------------------------------------
MOLECULAR_MAP = [
    {"range": (730, 750), "group": "Hemoglobina / porfirinas"},
    {"range": (748, 755), "group": "Citocromo c / heme"},
    {"range": (720, 735), "group": "Adenina / nucleotídeos (DNA/RNA)"},
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
]

# ---------------------------------------------------------------------
# REGRAS EXPLORATÓRIAS
# ---------------------------------------------------------------------
DISEASE_RULES = [
    {
        "name": "Alteração hemoglobina",
        "description": "Alterações estruturais no grupo heme/porfirinas.",
        "groups_required": ["Hemoglobina / porfirinas", "Citocromo c / heme"],
    },
    {
        "name": "Alteração proteica",
        "description": "Alterações conformacionais em proteínas.",
        "groups_required": [
            "Amida I (proteínas, C=O)",
            "Amida II",
            "Amida III (proteínas)",
        ],
    },
    {
        "name": "Alteração lipídica de membrana",
        "description": "Modificações em lipídios de membrana.",
        "groups_required": [
            "Lipídios – CH2 deformação",
            "Lipídios – CH2 torção",
            "Lipídios – C–C estiramento",
        ],
    },
]

# ---------------------------------------------------------------------
# DATACLASS
# ---------------------------------------------------------------------
@dataclass
class Peak:
    position_cm1: float
    intensity: float
    width: Optional[float] = None
    group: Optional[str] = None
    fit_params: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------------------
# LEITURA DE ESPECTRO
# ---------------------------------------------------------------------
def load_spectrum(file_like) -> Tuple[np.ndarray, np.ndarray]:
    name = getattr(file_like, "name", "").lower()

    if name.endswith(".csv"):
        try:
            df = pd.read_csv(file_like)
        except Exception:
            file_like.seek(0)
            df = pd.read_csv(file_like, sep=";")
    elif name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_like)
    else:
        df = pd.read_csv(file_like, sep=r"\s+", header=None, comment="#", engine="python")

    df = df.select_dtypes(include=[np.number])
    if df.shape[1] < 2:
        raise RuntimeError("Arquivo precisa ter ao menos duas colunas numéricas.")

    return df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float)

# ---------------------------------------------------------------------
# SUBTRAÇÃO DE SUBSTRATO
# ---------------------------------------------------------------------
def subtract_substrate(x, y, x_sub, y_sub):
    if not np.array_equal(x, x_sub):
        y_sub = np.interp(x, x_sub, y_sub)
    y_corr = y - y_sub
    y_corr[y_corr < 0] = 0.0
    return y_corr

# ---------------------------------------------------------------------
# DESPIKE / BASELINE / PREPROCESS
# ---------------------------------------------------------------------
def despike(y, method="median", k=5):
    if method == "median":
        y_med = medfilt(y, k)
        s = np.std(y)
        if s > 0:
            y[np.abs(y - y_med) > 3 * s] = y_med[np.abs(y - y_med) > 3 * s]
    elif method == "median_filter":
        y = median_filter(y, size=k)
    return y

def baseline_als(y, lam=1e5, p=0.01, niter=10):
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
    despike_method="median",
    smooth=True,
    window_length=9,
    polyorder=3,
    baseline_method="als",
    normalize=True,
):
    y = y.astype(float).copy()

    if despike_method:
        y = despike(y, despike_method)

    if smooth and len(y) > window_length:
        if window_length % 2 == 0:
            window_length += 1
        y = savgol_filter(y, window_length, polyorder)

    if baseline_method == "als":
        y = y - baseline_als(y)

    if normalize:
        rng = y.max() - y.min()
        if rng > 0:
            y = (y - y.min()) / rng

    return x, y, {}

# ---------------------------------------------------------------------
# AJUSTE DE PICO (lmfit)
# ---------------------------------------------------------------------
def fit_peak_lmfit(x, y, center, window=10.0, model="voigt"):
    if not LMFIT_AVAILABLE:
        return None

    mask = (x >= center - window) & (x <= center + window)
    xw, yw = x[mask], y[mask]
    if len(xw) < 6:
        return None

    mdl = (
        GaussianModel() if model == "gauss"
        else LorentzianModel() if model == "lorentz"
        else VoigtModel()
    )

    params = mdl.guess(yw, x=xw)
    res = mdl.fit(yw, params, x=xw)

    return {
        "model": model,
        "center": res.params["center"].value,
        "fwhm": res.params.get("fwhm", None).value if "fwhm" in res.params else None,
        "amplitude": res.params["amplitude"].value,
        "r2": res.rsquared,
    }

# ---------------------------------------------------------------------
# DETECÇÃO DE PICOS
# ---------------------------------------------------------------------
def detect_peaks(x, y, height=0.05, distance=5, prominence=0.02, fit_model=None):
    idx, _ = find_peaks(y, height=height, distance=distance, prominence=prominence)
    peaks = []
    for i in idx:
        p = Peak(float(x[i]), float(y[i]))
        if fit_model:
            p.fit_params = fit_peak_lmfit(x, y, p.position_cm1, model=fit_model)
            if p.fit_params and p.fit_params.get("fwhm"):
                p.width = p.fit_params["fwhm"]
        peaks.append(p)
    return peaks

def map_peaks_to_molecular_groups(peaks):
    for p in peaks:
        for item in MOLECULAR_MAP:
            if item["range"][0] <= p.position_cm1 <= item["range"][1]:
                p.group = item["group"]
                break
    return peaks

# ---------------------------------------------------------------------
# MAPA DE PREDOMINÂNCIA DE PICOS
# ---------------------------------------------------------------------
def compute_peak_density(
    peaks,
    x_min=400,
    x_max=1800,
    bin_width=2.0,
    smooth_window=21,
    polyorder=3,
):
    if not peaks:
        return None, None

    pos = np.array([p.position_cm1 for p in peaks])
    bins = np.arange(x_min, x_max + bin_width, bin_width)
    hist, edges = np.histogram(pos, bins=bins)

    x_centers = 0.5 * (edges[:-1] + edges[1:])
    y = hist.astype(float)

    if len(y) > smooth_window:
        if smooth_window % 2 == 0:
            smooth_window += 1
        y = savgol_filter(y, smooth_window, polyorder)

    baseline = baseline_als(y, lam=1e4, p=0.01)
    y = y - baseline
    y[y < 0] = 0.0

    if y.max() > 0:
        y /= y.max()

    return x_centers, y

# ---------------------------------------------------------------------
# FEATURES ML-READY
# ---------------------------------------------------------------------
def build_ml_features_from_peaks(peaks):
    features = {}
    for item in MOLECULAR_MAP:
        g = item["group"]
        features[f"count_{g}"] = 0
        features[f"sum_intensity_{g}"] = 0.0

    for p in peaks:
        if p.group:
            features[f"count_{p.group}"] += 1
            features[f"sum_intensity_{p.group}"] += p.intensity

    intens = [p.intensity for p in peaks]
    features["n_peaks"] = len(peaks)
    features["mean_intensity"] = float(np.mean(intens)) if intens else 0.0
    features["max_intensity"] = float(np.max(intens)) if intens else 0.0

    return features

# ---------------------------------------------------------------------
# PIPELINE COMPLETO
# ---------------------------------------------------------------------
def process_raman_spectrum_with_groups(
    file_like,
    substrate_file_like=None,
    preprocess_kwargs=None,
    peak_height=0.05,
    peak_distance=5,
    peak_prominence=0.02,
    fit_model=None,
):
    if preprocess_kwargs is None:
        preprocess_kwargs = {}

    x_raw, y_raw = load_spectrum(file_like)

    if substrate_file_like:
        x_sub, y_sub = load_spectrum(substrate_file_like)
        y_raw = subtract_substrate(x_raw, y_raw, x_sub, y_sub)

    x_proc, y_proc, meta = preprocess_spectrum(x_raw, y_raw, **preprocess_kwargs)

    peaks = detect_peaks(
        x_proc, y_proc,
        height=peak_height,
        distance=peak_distance,
        prominence=peak_prominence,
        fit_model=fit_model,
    )

    peaks = map_peaks_to_molecular_groups(peaks)
    diseases = infer_diseases(peaks)
    features = build_ml_features_from_peaks(peaks)

    x_density, y_density = compute_peak_density(peaks)

    return {
        "x_raw": x_raw,
        "y_raw": y_raw,
        "x_proc": x_proc,
        "y_proc": y_proc,
        "peaks": peaks,
        "diseases": diseases,
        "features": features,
        "x_density": x_density,
        "y_density": y_density,
        "meta": meta,
    }
