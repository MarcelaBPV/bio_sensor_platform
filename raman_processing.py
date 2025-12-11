# raman_processing.py
# -*- coding: utf-8 -*-
"""
Módulo de processamento Raman:
- load_spectrum
- despike_median
- baseline_als (esparsa, robusta)
- preprocess_spectrum
- detect_peaks, map_peaks_to_molecular_groups, infer_diseases
- calibrate_with_fixed_pattern_and_silicon (requere silicon, sample, blank)
- process_raman_spectrum_with_groups (fluxo alto-nível)
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter, medfilt
from scipy.sparse import diags, csc_matrix, identity
from scipy.sparse.linalg import spsolve

# -------------------------
# Configs / mapas (edite conforme necessário)
# -------------------------
MOLECULAR_MAP: List[Dict[str, Any]] = [

    # =========================================
    # 1) AROMÁTICOS / AMINOÁCIDOS
    # =========================================
    {"range": (995, 1007), "group": "Fenilalanina (~1001 cm⁻¹)"},
    {"range": (1000, 1008), "group": "Fenilalanina (~1003 cm⁻¹)"},

    # =========================================
    # 2) NUCLEOTÍDEOS / DNA / RNA
    # =========================================
    {"range": (780, 795), "group": "DNA/RNA fosfodiéster (~786 cm⁻¹)"},
    {"range": (1078, 1092), "group": "PO₂⁻ simétrico / DNA (~1085 cm⁻¹)"},

    # =========================================
    # 3) PROTEÍNAS / PEPTÍDEOS
    # =========================================
    {"range": (1118, 1126), "group": "C–N (~1122 cm⁻¹)"},
    {"range": (1235, 1255), "group": "Amida III (~1247 cm⁻¹)"},
    {"range": (1510, 1545), "group": "Amida II (~1530–1540 cm⁻¹)"},
    {"range": (1650, 1675), "group": "Amida I (C=O) (~1655 cm⁻¹)"},

    # =========================================
    # 4) LIPÍDIOS / MEMBRANA CELULAR
    # =========================================
    {"range": (1300, 1315), "group": "CH₂ twist (~1305 cm⁻¹)"},
    {"range": (1328, 1344), "group": "CH₂/CH₃ (~1336 cm⁻¹)"},
    {"range": (1374, 1386), "group": "CH₃ stretch (~1380 cm⁻¹)"},
    {"range": (1440, 1470), "group": "CH₂/CH₃ deform. (~1450 cm⁻¹)"},
    {"range": (1730, 1745), "group": "C=O ester (lipídios oxid.) (~1738 cm⁻¹)"},

    # =========================================
    # 5) HEMOGLOBINA / PORFIRINAS
    # =========================================
    {"range": (700, 740),  "group": "Porfirinas (banda baixa)"},
    {"range": (1355, 1375), "group": "Heme ν₄ (~1365 cm⁻¹)"},
    {"range": (1540, 1580), "group": "Hemoglobina (~1568 cm⁻¹)"},
    {"range": (1590, 1628), "group": "Porfirina / Amida I (~1597–1624 cm⁻¹)"},
    {"range": (1620, 1640), "group": "Heme ν₁₀ (~1630 cm⁻¹)"},

    # =========================================
    # 6) CAROTENOIDES
    # =========================================
    {"range": (1145, 1165), "group": "Carotenoide ν₂ (~1156 cm⁻¹)"},
    {"range": (1505, 1525), "group": "Carotenoide ν₁ (~1515 cm⁻¹)"},

    # =========================================
    # 7) SUBSTRATO — PAPEL (celulose)
    # =========================================
    {"range": (380, 410), "group": "Celulose (~395 cm⁻¹)"},
    {"range": (520, 535), "group": "Celulose (~525 cm⁻¹)"},
    {"range": (890, 910), "group": "Celulose (~900 cm⁻¹)"},
    {"range": (1080, 1100), "group": "Celulose (~1095 cm⁻¹)"},
    {"range": (1110, 1130), "group": "Celulose (~1120 cm⁻¹)"},
    {"range": (1365, 1385), "group": "Celulose (~1375 cm⁻¹)"},
    {"range": (1415, 1435), "group": "Celulose (~1425 cm⁻¹)"},

    # =========================================
    # 8) SUBSTRATO — PAPEL + PRATA (Ag)
    # =========================================
    {"range": (228, 260), "group": "Ag–S ligação (~240 cm⁻¹)"},
    {"range": (1090, 1100), "group": "Realce SERS Ag sobre celulose (~1095 cm⁻¹)"},
    {"range": (1350, 1380), "group": "Realce SERS Ag – região de celulose (~1370 cm⁻¹)"},

    # =========================================
    # 9) SUBSTRATO — PAPEL + OURO (Au)
    # =========================================
    {"range": (230, 250), "group": "Au–S ligação (~240 cm⁻¹)"},
    {"range": (1085, 1100), "group": "Aumento Raman Au sobre celulose (~1090 cm⁻¹)"},
    {"range": (1340, 1390), "group": "Realce Au – região celulose (~1370 cm⁻¹)"},
]


DISEASE_RULES: List[Dict[str, Any]] = [
    {
        "name": "Alteração hemoglobina",
        "description": "Padrão compatível com alterações estruturais ou oxidativas em heme e porfirinas.",
        "groups_required": ["Hemoglobina (~1568 cm⁻¹)", "Porfirina / Amida I (~1597–1624 cm⁻¹)"],
    },

    {
        "name": "Alteração proteica",
        "description": "Sinais associados a mudanças conformacionais em proteínas (Amida I / III), podendo refletir inflamação ou desnaturação.",
        "groups_required": ["Amida III (~1247 cm⁻¹)", "Amida I (C=O) (~1655 cm⁻¹)"],
    },

    {
        "name": "Alteração lipídica",
        "description": "Padrão compatível com mudanças em lipídios de membrana, inflamação, danos celulares ou desbalanço metabólico.",
        "groups_required": ["CH₂/CH₃ (~1336 cm⁻¹)", "CH₂/CH₃ (~1452 cm⁻¹)"],
    },

    {
        "name": "Estresse oxidativo",
        "description": "Associação clássica entre alterações em Fenilalanina e sinais fortes de heme, ligados à inflamação e produção de espécies reativas.",
        "groups_required": ["Fenilalanina (~1001 cm⁻¹)", "Hemoglobina (~1568 cm⁻¹)"],
    },

    {
        "name": "Dano de membrana",
        "description": "Associação entre CH₃ (~1380 cm⁻¹) e CH₂/CH₃ indica disrupção lipídica e possível apoptose ou necrose.",
        "groups_required": ["CH₃ (~1380 cm⁻¹)", "CH₂/CH₃ (~1452 cm⁻¹)"],
    },

    {
        "name": "Desbalanço aromático",
        "description": "Variação em aminoácidos aromáticos (Fenilalanina) associada a alterações metabólicas ou inflamação de baixo grau.",
        "groups_required": ["Fenilalanina (~1001 cm⁻¹)"],
    },

    {
        "name": "Alteração estrutural proteica",
        "description": "Mudanças simultâneas em Amida III e bandas C–N sugerem alterações secundárias ou desnaturação.",
        "groups_required": ["Amida III (~1247 cm⁻¹)", "C–N (~1122 cm⁻¹)"],
    },
]


# -------------------------
# Data classes
# -------------------------
@dataclass
class Peak:
    position_cm1: float
    intensity: float
    group: Optional[str] = None

@dataclass
class DiseaseMatch:
    name: str
    score: int
    description: str

# -------------------------
# Leitura de espectro
# -------------------------
def load_spectrum(file) -> Tuple[np.ndarray, np.ndarray]:
    """Lê .txt/.csv/.xls/.xlsx - retorna (x, y)."""
    filename = getattr(file, "name", "").lower()
    try:
        if filename.endswith(".txt"):
            df = pd.read_csv(file, sep=r"\s+", comment="#", engine="python")
        elif filename.endswith(".csv"):
            try:
                df = pd.read_csv(file)
            except Exception:
                file.seek(0)
                df = pd.read_csv(file, sep=";")
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file, sep=r"\s+", comment="#", engine="python")
    except Exception:
        file.seek(0)
        text = file.read()
        if isinstance(text, bytes):
            text = text.decode(errors="ignore")
        from io import StringIO
        df = pd.read_csv(StringIO(text), sep=r"\s+", comment="#", engine="python")

    df = df.dropna(axis=1, how="all")
    if df.shape[1] < 2:
        raise ValueError("Não foi possível identificar colunas X e Y no arquivo.")

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        x = numeric_df.iloc[:, 0].to_numpy(dtype=float)
        y = numeric_df.iloc[:, 1].to_numpy(dtype=float)
    else:
        x = df.iloc[:, 0].astype(float).to_numpy()
        y = df.iloc[:, 1].astype(float).to_numpy()

    return x, y

# -------------------------
# Despike (mediana + zscore)
# -------------------------
def despike_median(y: np.ndarray, kernel_size: int = 5, z_thresh: float = 6.0) -> np.ndarray:
    y = y.astype(float).copy()
    try:
        y_med = medfilt(y, kernel_size=kernel_size)
    except Exception:
        k = min(kernel_size, len(y) if len(y) % 2 == 1 else len(y) - 1)
        if k < 3:
            return y
        y_med = medfilt(y, kernel_size=k)
    resid = y - y_med
    sigma = np.std(resid) + 1e-12
    z = np.abs(resid) / sigma
    y[z > z_thresh] = y_med[z > z_thresh]
    return y

# -------------------------
# Baseline ALS robusta (sparse)
# -------------------------
def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    L = y.size
    if L < 3:
        return np.zeros_like(y)
    # D is (L-2, L) second-difference operator (sparse)
    D = diags([np.ones(L-2), -2*np.ones(L-2), np.ones(L-2)], [0,1,2], shape=(L-2, L))
    H = (D.T @ D) * lam  # sparse (L x L)
    w = np.ones(L, dtype=float)
    z = np.zeros(L, dtype=float)
    for _ in range(niter):
        W = diags(w, 0, shape=(L, L))
        Z = W + H
        try:
            z = spsolve(Z, w * y)
        except Exception:
            Zd = Z.toarray()
            z = np.linalg.solve(Zd, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

# -------------------------
# Pré-processamento: despike -> baseline -> savgol -> normalize
# -------------------------
def preprocess_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    use_despike: bool = True,
    smooth: bool = True,
    window_length: int = 9,
    polyorder: int = 3,
    baseline_method: Optional[str] = "als",
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    y_proc = y.astype(float).copy()

    if use_despike and len(y_proc) >= 3:
        y_proc = despike_median(y_proc, kernel_size=5, z_thresh=6.0)
        meta['despike'] = 'median'

    base = np.zeros_like(y_proc)
    if baseline_method == "als":
        base = baseline_als(y_proc, lam=1e5, p=0.01, niter=10)
        meta['baseline'] = 'als'
    elif baseline_method is None:
        meta['baseline'] = 'none'
    else:
        raise ValueError(f"baseline_method desconhecido: {baseline_method}")

    y_proc = y_proc - base

    if smooth and len(y_proc) >= 5:
        wl = window_length
        if wl >= len(y_proc):
            wl = len(y_proc) - 1
        if wl % 2 == 0:
            wl += 1
        wl = max(wl, 3)
        try:
            y_proc = savgol_filter(y_proc, window_length=wl, polyorder=polyorder)
            meta['savgol'] = {'window_length': wl, 'polyorder': polyorder}
        except Exception as e:
            meta['savgol_error'] = str(e)

    if normalize:
        ymin = float(np.min(y_proc))
        ymax = float(np.max(y_proc))
        if ymax > ymin + 1e-12:
            y_proc = (y_proc - ymin) / (ymax - ymin)
            meta['normalize'] = 'minmax'
        else:
            meta['normalize'] = 'none'

    return x, y_proc, meta

# -------------------------
# Detect peaks
# -------------------------
def detect_peaks(
    x: np.ndarray,
    y: np.ndarray,
    height: float = 0.1,
    distance: int = 5,
    prominence: float = 0.02,
) -> List[Peak]:
    indices, props = find_peaks(y, height=height, distance=distance, prominence=prominence)
    peaks = [Peak(position_cm1=float(x[i]), intensity=float(y[i])) for i in indices]
    return peaks

# -------------------------
# Map peaks -> molecular groups
# -------------------------
def map_peaks_to_molecular_groups(peaks: List[Peak]) -> List[Peak]:
    for p in peaks:
        group_found = None
        for item in MOLECULAR_MAP:
            lo, hi = item["range"]
            if lo <= p.position_cm1 <= hi:
                group_found = item["group"]
                break
        p.group = group_found
    return peaks

# -------------------------
# Infer diseases (rules)
# -------------------------
def infer_diseases(peaks: List[Peak]) -> List[DiseaseMatch]:
    groups = {p.group for p in peaks if p.group is not None}
    matches: List[DiseaseMatch] = []
    for rule in DISEASE_RULES:
        required = set(rule["groups_required"])
        score = len(required.intersection(groups))
        if score > 0:
            matches.append(DiseaseMatch(name=rule["name"], score=score, description=rule["description"]))
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches

# -------------------------
# High-level processing
# -------------------------
def process_raman_spectrum_with_groups(
    file,
    preprocess_kwargs: Optional[Dict[str, Any]] = None,
    peak_height: float = 0.1,
    peak_distance: int = 5,
    peak_prominence: float = 0.02,
) -> Dict[str, Any]:
    if preprocess_kwargs is None:
        preprocess_kwargs = {}
    x_raw, y_raw = load_spectrum(file)
    x_proc, y_proc, meta = preprocess_spectrum(x_raw, y_raw, **preprocess_kwargs)
    peaks = detect_peaks(x_proc, y_proc, height=peak_height, distance=peak_distance, prominence=peak_prominence)
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

# -------------------------
# Calibration workflow requiring 3 uploads
# -------------------------
def calibrate_with_fixed_pattern_and_silicon(
    silicon_file,
    sample_file,
    blank_file,
    base_poly_coeffs: np.ndarray,
    silicon_ref_position: float = 520.7,
    preprocess_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Workflow que exige os 3 arquivos: silicon, sample, blank."""
    # validate uploads
    if silicon_file is None or sample_file is None or blank_file is None:
        raise ValueError("Forneça os três arquivos: silício, amostra e blank.")

    if preprocess_kwargs is None:
        preprocess_kwargs = {"use_despike": True, "smooth": True, "window_length": 9, "polyorder": 3, "baseline_method": "als", "normalize": False}

    # silicon
    x_si_raw, y_si_raw = load_spectrum(silicon_file)
    x_si, y_si, meta_si = preprocess_spectrum(x_si_raw, y_si_raw, **preprocess_kwargs)
    # apply base poly (assumes base_poly_coeffs is np.poly coefficients for polyval)
    x_si_base = np.polyval(np.asarray(base_poly_coeffs, dtype=float), x_si)
    mask_si = (x_si_base >= 480) & (x_si_base <= 560)
    if not np.any(mask_si):
        raise RuntimeError("Janela de Si (480-560 cm^-1) vazia.")
    idx_rel_max = int(np.argmax(y_si[mask_si]))
    si_cal_base = float(x_si_base[mask_si][idx_rel_max])
    delta = silicon_ref_position - si_cal_base
    def corrector_final(x_obs: np.ndarray) -> np.ndarray:
        return np.polyval(np.asarray(base_poly_coeffs, dtype=float), x_obs) + delta
    x_si_cal = corrector_final(x_si)

    # sample
    x_s_raw, y_s_raw = load_spectrum(sample_file)
    x_s, y_s, meta_s = preprocess_spectrum(x_s_raw, y_s_raw, **preprocess_kwargs)
    x_s_cal = corrector_final(x_s)

    # blank
    x_b_raw, y_b_raw = load_spectrum(blank_file)
    x_b, y_b, meta_b = preprocess_spectrum(x_b_raw, y_b_raw, **preprocess_kwargs)
    x_b_cal = corrector_final(x_b)

    # blank subtraction (interp)
    y_b_interp = np.interp(x_s_cal, x_b_cal, y_b, left=0.0, right=0.0)
    y_sample_blank_corrected = y_s - y_b_interp

    return {
        "x_sample_raw": x_s_raw, "y_sample_raw": y_s_raw,
        "x_sample_proc": x_s, "y_sample_proc": y_s,
        "x_sample_calibrated": x_s_cal,
        "x_blank_raw": x_b_raw, "y_blank_raw": y_b_raw,
        "x_blank_proc": x_b, "y_blank_proc": y_b,
        "x_blank_calibrated": x_b_cal,
        "x_silicon_raw": x_si_raw, "y_silicon_raw": y_si_raw,
        "x_silicon_proc": x_si, "y_silicon_proc": y_si,
        "x_silicon_calibrated": x_si_cal,
        "y_sample_blank_corrected": y_sample_blank_corrected,
        "meta_sample": meta_s, "meta_blank": meta_b, "meta_silicon": meta_si,
        "calibration": {"base_poly_coeffs": np.asarray(base_poly_coeffs, dtype=float).tolist(), "si_cal_base": si_cal_base, "silicon_ref_position": silicon_ref_position, "laser_zero_delta": delta},
    }
