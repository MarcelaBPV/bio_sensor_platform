# raman_processing_harmonized.py
# -*- coding: utf-8 -*-
"""
Pipeline Raman harmonizado inspirado no ramanchada2:
- Leitura de espectros (txt/csv/xlsx)
- Despike com comparador de algoritmos
- Suavização + baseline + normalização
- Detecção e ajuste multi-peak (lmfit Voigt/Gauss)
- Mapeamento de grupos moleculares e regras de "doenças"
- Calibração completa do eixo Raman (Neon/Poliestireno + Si)
- Geração de espectros sintéticos
- Cache simples em HDF5 e export minimal NeXus-like
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter, medfilt
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

# Dependências opcionais
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


# ======================================================================
# MAPAS E REGRAS BIOMOLECULARES (como no seu código original)
# ======================================================================

MOLECULAR_MAP: List[Dict] = [
    {"range": (700, 740), "group": "Hemoglobina / porfirinas"},
    {"range": (995, 1005), "group": "Fenilalanina (anéis aromáticos)"},
    {"range": (1440, 1470), "group": "Lipídios / CH2 deformação"},
    {"range": (1650, 1670), "group": "Amidas / proteínas (C=O)"},
    # TODO: expandir conforme sua tabela completa
]

DISEASE_RULES: List[Dict] = [
    {
        "name": "Alteração hemoglobina",
        "description": "Padrão compatível com alterações em heme / porfirinas.",
        "groups_required": ["Hemoglobina / porfirinas"],
    },
    {
        "name": "Alteração proteica",
        "description": "Padrão compatível com alterações em proteínas (amida I).",
        "groups_required": ["Amidas / proteínas (C=O)"],
    },
    {
        "name": "Alteração lipídica",
        "description": "Padrão compatível com alterações em lipídios de membrana.",
        "groups_required": ["Lipídios / CH2 deformação"],
    },
]


# ======================================================================
# DATA CLASSES
# ======================================================================

@dataclass
class Peak:
    position_cm1: float
    intensity: float
    width: Optional[float] = None
    group: Optional[str] = None
    fit_params: Optional[Dict[str, Any]] = None


@dataclass
class DiseaseMatch:
    name: str
    score: int
    description: str


# ======================================================================
# 1) CARREGAMENTO DE ESPECTRO
# ======================================================================

def load_spectrum(file) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lê arquivo de espectro e retorna (x, y) como numpy arrays.
    Suporta txt/csv/xls/xlsx de forma robusta.
    """
    filename = getattr(file, "name", "").lower()

    if filename.endswith(".txt"):
        df = pd.read_csv(file, sep=r"\s+", comment="#", engine="python", header=None)
    elif filename.endswith(".csv"):
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, sep=";")
    elif filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file, sep=r"\s+", comment="#", engine="python", header=None)

    df = df.dropna(axis=1, how="all")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        x = numeric_df.iloc[:, 0].to_numpy(dtype=float)
        y = numeric_df.iloc[:, 1].to_numpy(dtype=float)
    else:
        x = df.iloc[:, 0].astype(float).to_numpy()
        y = df.iloc[:, 1].astype(float).to_numpy()

    return x, y


# ======================================================================
# 2) DESPIKE + COMPARADOR (A)
# ======================================================================

def despike(y: np.ndarray, method: str = "median", kernel_size: int = 5,
            z_thresh: float = 6.0) -> np.ndarray:
    """
    Métodos simples de despike:
    - 'median'          : filtro mediano 1D
    - 'zscore'          : outliers locais por z-score
    - 'median_filter_nd': median_filter do ndimage
    """
    y = y.copy()
    if method == "median":
        y_filtered = medfilt(y, kernel_size=kernel_size)
        mask = np.abs(y - y_filtered) > (np.std(y) * 3)
        y[mask] = y_filtered[mask]
        return y

    elif method == "zscore":
        mu = pd.Series(y).rolling(window=kernel_size, center=True,
                                  min_periods=1).median().to_numpy()
        resid = y - mu
        z = np.abs(resid) / (np.std(resid) + 1e-12)
        y[z > z_thresh] = mu[z > z_thresh]
        return y

    elif method == "median_filter_nd":
        return median_filter(y, size=kernel_size)

    else:
        raise ValueError(f"Método despike desconhecido: {method}")


def _despike_metric(y_original: np.ndarray, y_despiked: np.ndarray) -> float:
    """
    Métrica para comparar algoritmos de despike:
    - queremos série mais suave mas ainda próxima do original.
    metric = mean(|2a derivada|) + alpha * MSE
    """
    if len(y_despiked) < 3:
        return np.inf

    second_deriv = np.diff(y_despiked, n=2)
    smooth_term = np.mean(np.abs(second_deriv))

    mse = np.mean((y_original - y_despiked) ** 2)
    # alpha escala a importância de permanecer próximo
    alpha = 0.1 / (np.var(y_original) + 1e-12)
    return float(smooth_term + alpha * mse)


def compare_despike_algorithms(
    y: np.ndarray,
    methods: Optional[List[str]] = None,
    kernel_size: int = 5,
    z_thresh: float = 6.0,
) -> Tuple[np.ndarray, str, Dict[str, float]]:
    """
    A) Roda múltiplos métodos de despike e escolhe o melhor por métrica.
    Retorna:
        y_best, best_method, metrics_por_método
    """
    if methods is None:
        methods = ["median", "zscore", "median_filter_nd"]

    metrics: Dict[str, float] = {}
    best_y = y.copy()
    best_method = None
    best_metric = np.inf

    for m in methods:
        y_d = despike(y, method=m, kernel_size=kernel_size, z_thresh=z_thresh)
        metric = _despike_metric(y, y_d)
        metrics[m] = metric
        if metric < best_metric:
            best_metric = metric
            best_y = y_d
            best_method = m

    if best_method is None:
        best_method = "none"

    return best_y, best_method, metrics


# ======================================================================
# 3) BASELINE (ALS + FFT LOWPASS)
# ======================================================================

def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01,
                 niter: int = 10) -> np.ndarray:
    """
    Asymmetric Least Squares baseline (Eilers).
    """
    L = len(y)
    D = np.diff(np.eye(L), 2)
    H = lam * D.T.dot(D)
    w = np.ones(L)
    for _ in range(niter):
        W = np.diag(w)
        Z = W + H
        z = np.linalg.solve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def baseline_fft_smooth(y: np.ndarray, cutoff_fraction: float = 0.02) -> np.ndarray:
    """
    Baseline aproximada via filtragem de baixa frequência no domínio de Fourier.
    """
    Y = np.fft.rfft(y)
    n = len(Y)
    cutoff = max(1, int(n * cutoff_fraction))
    Y[cutoff:-cutoff] = 0
    baseline = np.fft.irfft(Y, n=len(y))
    return baseline


# ======================================================================
# 4) PRÉ-PROCESSAMENTO COMPLETO
# ======================================================================

def preprocess_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    despike_method: Optional[str] = "median",
    smooth: bool = True,
    window_length: int = 9,
    polyorder: int = 3,
    baseline_method: str = "als",
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Pipeline:
    - Despike (método fixo ou 'auto_compare' para rodar o comparador)
    - Suavização Savitzky-Golay
    - Baseline subtraction (ALS ou FFT)
    - Normalização 0–1
    """
    y_proc = y.astype(float).copy()
    meta: Dict[str, Any] = {}

    # DESPIKE
    if despike_method == "auto_compare":
        y_proc, best_method, metrics = compare_despike_algorithms(y_proc)
        meta["despike_method"] = best_method
        meta["despike_metrics"] = metrics
    elif despike_method is not None:
        y_proc = despike(y_proc, method=despike_method)
        meta["despike_method"] = despike_method

    # SUAVIZAÇÃO
    if smooth:
        if window_length >= len(y_proc):
            window_length = len(y_proc) - 1
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(window_length, 3)
        y_proc = savgol_filter(y_proc, window_length=window_length,
                               polyorder=polyorder)
        meta["savgol"] = {"window_length": window_length,
                          "polyorder": polyorder}

    # BASELINE
    if baseline_method == "als":
        base = baseline_als(y_proc, lam=1e5, p=0.01, niter=10)
    elif baseline_method == "fft":
        base = baseline_fft_smooth(y_proc, cutoff_fraction=0.02)
    elif baseline_method is None:
        base = np.zeros_like(y_proc)
    else:
        raise ValueError(f"baseline_method desconhecido: {baseline_method}")

    y_proc = y_proc - base
    meta["baseline"] = {"method": baseline_method}

    # NORMALIZAÇÃO
    if normalize:
        ymin = float(np.min(y_proc))
        ymax = float(np.max(y_proc))
        if ymax > ymin:
            y_proc = (y_proc - ymin) / (ymax - ymin)
        meta["normalize"] = True

    return x, y_proc, meta


# ======================================================================
# 5) DETECÇÃO DE PICOS
# ======================================================================

def detect_peaks(
    x: np.ndarray,
    y: np.ndarray,
    height: float = 0.05,
    distance: int = 5,
    prominence: float = 0.02,
) -> List[Peak]:
    indices, properties = find_peaks(
        y,
        height=height,
        distance=distance,
        prominence=prominence,
    )
    peaks: List[Peak] = []
    for idx in indices:
        intensity = float(y[idx])
        peaks.append(Peak(position_cm1=float(x[idx]),
                          intensity=intensity,
                          width=None))
    return peaks


# ======================================================================
# 6) MAPEAMENTO MOLECULAR
# ======================================================================

def map_peaks_to_molecular_groups(peaks: List[Peak]) -> List[Peak]:
    for peak in peaks:
        group_found = None
        for item in MOLECULAR_MAP:
            x_min, x_max = item["range"]
            if x_min <= peak.position_cm1 <= x_max:
                group_found = item["group"]
                break
        peak.group = group_found
    return peaks


# ======================================================================
# 7) FIT SIMPLES (GAUSSIANO) – FALLBACK
# ======================================================================

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))


def fit_peak_simple(
    x: np.ndarray,
    y: np.ndarray,
    center: float,
    window: float = 10.0,
) -> Dict[str, Any]:
    mask = (x >= center - window) & (x <= center + window)
    xi, yi = x[mask], y[mask]
    if len(xi) < 5:
        return {}
    amp0 = float(np.max(yi))
    cen0 = float(center)
    wid0 = 2.0
    try:
        popt, _ = curve_fit(gaussian, xi, yi, p0=[amp0, cen0, wid0],
                            maxfev=2000)
        return {
            "model": "gaussian",
            "params": {
                "amp": float(popt[0]),
                "cen": float(popt[1]),
                "wid": float(popt[2]),
            },
        }
    except Exception:
        return {}


# ======================================================================
# 8) AJUSTE MULTI-PEAK COM LMFIT (B)
# ======================================================================

def fit_peaks_lmfit_global(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[Peak],
    model_type: str = "Voigt",
    x_window: Optional[Tuple[float, float]] = None,
) -> List[Peak]:
    """
    B) Ajuste multi-peak global usando lmfit.
    - model_type: 'Voigt', 'Gaussian', 'Lorentzian'
    - x_window: se fornecido, usa apenas esse range de x.
    """
    if not LMFIT_AVAILABLE:
        raise RuntimeError("lmfit não está disponível. Instale com 'pip install lmfit'.")

    # Seleciona faixa de x para o ajuste
    if x_window is not None:
        xmin, xmax = x_window
    else:
        xmin = min(p.position_cm1 for p in peaks) - 20
        xmax = max(p.position_cm1 for p in peaks) + 20

    mask = (x >= xmin) & (x <= xmax)
    x_fit = x[mask]
    y_fit = y[mask]

    if len(x_fit) < 5 or len(peaks) == 0:
        return peaks

    # Escolhe modelo base
    def make_model(prefix: str):
        if model_type.lower() == "gaussian":
            return GaussianModel(prefix=prefix)
        elif model_type.lower() == "lorentzian":
            return LorentzianModel(prefix=prefix)
        else:
            return VoigtModel(prefix=prefix)

    # Constrói modelo multi-peak
    model = None
    for i, p in enumerate(peaks):
        m = make_model(prefix=f"p{i}_")
        model = m if model is None else (model + m)

    params = model.make_params()

    # Chutes iniciais razoáveis
    for i, p in enumerate(peaks):
        pref = f"p{i}_"
        cen = p.position_cm1
        amp = p.intensity * 10.0  # Voigt/Gauss amplitude é área aproximada
        sigma0 = 3.0

        params[f"{pref}center"].set(value=cen, min=cen-10, max=cen+10)
        params[f"{pref}amplitude"].set(value=amp, min=0)
        if f"{pref}sigma" in params:
            params[f"{pref}sigma"].set(value=sigma0, min=0.1, max=50)
        if f"{pref}gamma" in params:
            params[f"{pref}gamma"].set(value=sigma0, min=0.1, max=50)

    # Ajuste
    result = model.fit(y_fit, params, x=x_fit)

    # Atualiza objetos Peak
    for i, p in enumerate(peaks):
        pref = f"p{i}_"
        fit_params = {}
        for name, val in result.params.items():
            if name.startswith(pref):
                fit_params[name.replace(pref, "")] = float(val.value)

        p.fit_params = fit_params
        if "center" in fit_params:
            p.position_cm1 = fit_params["center"]
        if "amplitude" in fit_params:
            # intensidade "ajustada" ~ amplitude/sigma (simplificado)
            p.intensity = fit_params["amplitude"]
        if "sigma" in fit_params:
            p.width = fit_params["sigma"]

    return peaks


def fit_peaks(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[Peak],
    use_lmfit: bool = True,
    model_type: str = "Voigt",
    window_simple: float = 8.0,
) -> List[Peak]:
    """
    Função de alto nível:
    - Se lmfit disponível e use_lmfit=True -> fit multi-peak global.
    - Caso contrário, usa fit gaussiano simples por pico.
    """
    if use_lmfit and LMFIT_AVAILABLE and len(peaks) > 0:
        return fit_peaks_lmfit_global(x, y, peaks, model_type=model_type)

    # fallback: ajusta pico a pico com curve_fit
    for p in peaks:
        res = fit_peak_simple(x, y, p.position_cm1, window=window_simple)
        if res:
            p.fit_params = res["params"]
            if "cen" in res["params"]:
                p.position_cm1 = float(res["params"]["cen"])
            if "amp" in res["params"]:
                p.intensity = float(res["params"]["amp"])
            if "wid" in res["params"]:
                p.width = float(res["params"]["wid"])
    return peaks


# ======================================================================
# 9) REGRAS DE "DOENÇA" / PADRÕES
# ======================================================================

def infer_diseases(peaks: List[Peak]) -> List[DiseaseMatch]:
    groups_present = {p.group for p in peaks if p.group is not None}
    matches: List[DiseaseMatch] = []
    for rule in DISEASE_RULES:
        required = set(rule["groups_required"])
        score = len(required.intersection(groups_present))
        if score > 0:
            matches.append(
                DiseaseMatch(
                    name=rule["name"],
                    score=score,
                    description=rule["description"],
                )
            )
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches


# ======================================================================
# 10) CALIBRAÇÃO DE EIXO WAVENUMBER (FUNÇÃO BASE)
# ======================================================================

def calibrate_wavenumber(
    observed_positions: np.ndarray,
    reference_positions: np.ndarray,
    degree: int = 1,
):
    """
    Ajusta polinômio que mapeia observed -> reference.
    Retorna:
        corrector(x_obs), coeffs
    """
    if len(observed_positions) < degree + 1:
        raise ValueError("Pontos insuficientes para o grau do polinômio.")

    coeffs = np.polyfit(observed_positions, reference_positions, deg=degree)

    def corrector(x_obs: np.ndarray) -> np.ndarray:
        return np.polyval(coeffs, x_obs)

    return corrector, coeffs


def calibrate_intensity(y_meas: np.ndarray, y_ref: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    """
    Calibração relativa de intensidade:
    - y_ref: medição do padrão no instrumento
    - y_std: espectro de referência (NIST/Lâmpada calibrada) na mesma escala de x
    """
    ratio = np.where(y_ref > 0, y_std / (y_ref + 1e-12), 1.0)
    return y_meas * ratio


# ======================================================================
# 11) WORKFLOW COMPLETO DE CALIBRAÇÃO (CWA-STYLE) (C)
# ======================================================================

def _match_peaks_to_refs(
    peak_positions: np.ndarray,
    ref_positions: np.ndarray,
    max_diff: float = 15.0,
) -> Tuple[List[float], List[float]]:
    """
    Casa cada ref_position com o pico mais próximo (se dentro de max_diff).
    """
    matched_obs = []
    matched_ref = []
    for ref in ref_positions:
        idx = np.argmin(np.abs(peak_positions - ref))
        obs = peak_positions[idx]
        if abs(obs - ref) <= max_diff:
            matched_obs.append(float(obs))
            matched_ref.append(float(ref))
    return matched_obs, matched_ref


def calibrate_instrument_from_files(
    neon_file,
    polystyrene_file,
    silicon_file,
    sample_file,
    neon_ref_positions: np.ndarray,
    poly_ref_positions: np.ndarray,
    silicon_ref_position: float = 520.7,
    poly_degree: int = 2,
    preprocess_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    C) Workflow de calibração:
    1) Lê espectros de Neon, Poliestireno e Silício.
    2) Pré-processa (despike/baseline/etc).
    3) Detecta picos e associa com posições de referência.
    4) Ajusta polinômio de calibração (Neon+Poli).
    5) Ajusta zero do laser usando Silício.
    6) Aplica correção ao espectro da amostra.

    Retorna dict com:
        - x_sample_raw, y_sample_raw
        - x_sample_calibrated
        - detalhes dos padrões, coeficientes e offset
    """
    if preprocess_kwargs is None:
        preprocess_kwargs = {
            "despike_method": "auto_compare",
            "smooth": True,
            "baseline_method": "als",
            "normalize": False,
        }

    # --- NEON ---
    x_neon_raw, y_neon_raw = load_spectrum(neon_file)
    x_neon, y_neon, _ = preprocess_spectrum(x_neon_raw, y_neon_raw, **preprocess_kwargs)
    neon_peaks = detect_peaks(x_neon, y_neon, height=0.1, distance=3, prominence=0.05)
    neon_positions = np.array([p.position_cm1 for p in neon_peaks], dtype=float)

    obs_neon, ref_neon = _match_peaks_to_refs(neon_positions, neon_ref_positions)

    # --- POLIESTIRENO ---
    x_poly_raw, y_poly_raw = load_spectrum(polystyrene_file)
    x_poly, y_poly, _ = preprocess_spectrum(x_poly_raw, y_poly_raw, **preprocess_kwargs)
    poly_peaks = detect_peaks(x_poly, y_poly, height=0.1, distance=3, prominence=0.05)
    poly_positions = np.array([p.position_cm1 for p in poly_peaks], dtype=float)

    obs_poly, ref_poly = _match_peaks_to_refs(poly_positions, poly_ref_positions)

    # Combina pontos de calibração (Neon + Poli)
    obs_all = np.array(obs_neon + obs_poly, dtype=float)
    ref_all = np.array(ref_neon + ref_poly, dtype=float)

    if len(obs_all) < poly_degree + 1:
        raise RuntimeError("Pontos de calibração insuficientes para o grau do polinômio.")

    corrector_base, coeffs_base = calibrate_wavenumber(obs_all, ref_all, degree=poly_degree)

    # --- SILÍCIO (zero do laser) ---
    x_si_raw, y_si_raw = load_spectrum(silicon_file)
    x_si, y_si, _ = preprocess_spectrum(x_si_raw, y_si_raw, **preprocess_kwargs)

    # procura pico mais intenso na região típica do Si
    mask_si = (x_si >= 480) & (x_si <= 560)
    if not np.any(mask_si):
        raise RuntimeError("Não há pontos suficientes na janela de Si (480–560 cm-1).")
    idx_max = np.argmax(y_si[mask_si])
    x_si_region = x_si[mask_si]
    si_obs_position = float(x_si_region[idx_max])

    # valor calibrado pelo polinômio base
    si_cal_base = float(corrector_base(np.array([si_obs_position]))[0])
    delta = silicon_ref_position - si_cal_base

    def corrector_final(x_arr: np.ndarray) -> np.ndarray:
        return corrector_base(x_arr) + delta

    # --- AMOSTRA ---
    x_s_raw, y_s_raw = load_spectrum(sample_file)
    x_s, y_s, meta_s = preprocess_spectrum(x_s_raw, y_s_raw, **preprocess_kwargs)
    x_s_cal = corrector_final(x_s)

    return {
        "x_sample_raw": x_s_raw,
        "y_sample_raw": y_s_raw,
        "x_sample_proc": x_s,
        "y_sample_proc": y_s,
        "x_sample_calibrated": x_s_cal,
        "meta_sample": meta_s,
        "calibration": {
            "obs_neon": obs_neon,
            "ref_neon": ref_neon,
            "obs_poly": obs_poly,
            "ref_poly": ref_poly,
            "coeffs_base": coeffs_base.tolist(),
            "si_obs_position": si_obs_position,
            "si_cal_base": si_cal_base,
            "silicon_ref_position": silicon_ref_position,
            "laser_zero_delta": delta,
        },
        "standards": {
            "neon_peaks": neon_positions.tolist(),
            "poly_peaks": poly_positions.tolist(),
        },
    }


# ======================================================================
# 12) GERAÇÃO DE ESPECTROS SINTÉTICOS
# ======================================================================

def generate_synthetic_spectrum(
    x: np.ndarray,
    peaks_spec: List[Dict[str, float]],
    baseline_level: float = 0.0,
    noise_std: float = 0.01,
) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    for spec in peaks_spec:
        pos = spec["position"]
        amp = spec.get("amp", 1.0)
        fwhm = spec.get("fwhm", 4.0)
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        y += amp * np.exp(-0.5 * ((x - pos) / sigma) ** 2)
    y += baseline_level + 0.0001 * (x - np.mean(x)) ** 2
    y += np.random.normal(scale=noise_std, size=y.shape)
    return y


# ======================================================================
# 13) CACHE HDF5 + NEXUS-LIKE
# ======================================================================

def cache_processed_spectrum(
    cache_file: str,
    key: str,
    x: np.ndarray,
    y: np.ndarray,
    meta: Dict[str, Any],
):
    if not H5PY_AVAILABLE:
        raise RuntimeError("h5py não está disponível para cache HDF5.")
    with h5py.File(cache_file, "a") as f:
        if key in f:
            del f[key]
        grp = f.create_group(key)
        grp.create_dataset("x", data=x)
        grp.create_dataset("y", data=y)
        import json
        grp.attrs["meta"] = json.dumps(meta)


def load_cached_spectrum(
    cache_file: str,
    key: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    if not H5PY_AVAILABLE:
        return None
    try:
        with h5py.File(cache_file, "r") as f:
            if key not in f:
                return None
            grp = f[key]
            x = grp["x"][:]
            y = grp["y"][:]
            import json
            meta = json.loads(grp.attrs.get("meta", "{}"))
            return x, y, meta
    except Exception:
        return None


def save_to_nexus(
    fname: str,
    x: np.ndarray,
    y: np.ndarray,
    metadata: Dict[str, Any],
):
    if not H5PY_AVAILABLE:
        raise RuntimeError("h5py não disponível para salvar NeXus.")
    with h5py.File(fname, "w") as f:
        nxentry = f.create_group("entry")
        nxentry.attrs["NX_class"] = "NXentry"
        instrument = nxentry.create_group("instrument")
        instrument.attrs["NX_class"] = "NXinstrument"
        nxdata = nxentry.create_group("data")
        nxdata.attrs["NX_class"] = "NXdata"
        nxdata.create_dataset("wavenumber", data=x)
        nxdata.create_dataset("intensity", data=y)
        meta_grp = nxentry.create_group("metadata")
        for k, v in metadata.items():
            meta_grp.attrs[str(k)] = str(v)


# ======================================================================
# 14) PIPELINE COMPLETO PARA UM ARQUIVO
# ======================================================================

def process_file_pipeline(
    file_obj,
    cache_file: Optional[str] = None,
    cache_key: Optional[str] = None,
    use_lmfit: bool = True,
) -> Dict[str, Any]:
    x_raw, y_raw = load_spectrum(file_obj)

    if cache_file and cache_key:
        cached = load_cached_spectrum(cache_file, cache_key)
        if cached is not None:
            x, y, meta = cached
            meta["from_cache"] = True
        else:
            x, y, meta = preprocess_spectrum(x_raw, y_raw,
                                             despike_method="auto_compare")
            cache_processed_spectrum(cache_file, cache_key, x, y, meta)
    else:
        x, y, meta = preprocess_spectrum(x_raw, y_raw,
                                         despike_method="auto_compare")

    peaks = detect_peaks(x, y)
    peaks = fit_peaks(x, y, peaks, use_lmfit=use_lmfit, model_type="Voigt")
    peaks = map_peaks_to_molecular_groups(peaks)
    diseases = infer_diseases(peaks)

    result = {
        "x_raw": x_raw,
        "y_raw": y_raw,
        "x_proc": x,
        "y_proc": y,
        "peaks": peaks,
        "diseases": diseases,
        "meta": meta,
    }
    return result


# ======================================================================
# 15) EXEMPLO RÁPIDO (LOCAL)
# ======================================================================

if __name__ == "__main__":
    import io

    x = np.linspace(200, 2000, 1801)
    peaks_spec = [
        {"position": 710, "amp": 1.0, "fwhm": 6.0},
        {"position": 1450, "amp": 0.8, "fwhm": 10.0},
    ]
    y = generate_synthetic_spectrum(x, peaks_spec,
                                    baseline_level=0.01,
                                    noise_std=0.02)

    buf = io.BytesIO()
    pd.DataFrame({"x": x, "y": y}).to_csv(buf, index=False)
    buf.seek(0)
    buf.name = "synthetic.csv"

    res = process_file_pipeline(buf, use_lmfit=False)
    print("Peaks encontrados:")
    for p in res["peaks"]:
        print(f"  {p.position_cm1:.2f} cm-1  I={p.intensity:.3f} group={p.group} fit={p.fit_params}")
    print("Padrões:", [(d.name, d.score) for d in res["diseases"]])
