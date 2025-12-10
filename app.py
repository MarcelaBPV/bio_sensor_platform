# raman_processing.py
# -*- coding: utf-8 -*-

import io
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd

from scipy.signal import find_peaks, savgol_filter, medfilt
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve

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
    {"range": (700, 740), "group": "Hemoglobina / porfirinas"},
    {"range": (995, 1005), "group": "Fenilalanina (anéis aromáticos)"},
    {"range": (1440, 1470), "group": "Lipídios / CH2 deformação"},
    {"range": (1650, 1670), "group": "Amidas / proteínas (C=O)"},
]

DISEASE_RULES = [
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
    """
    Lê arquivo de espectro (txt, csv, xls/xlsx) e retorna (x, y) como numpy arrays.
    Tenta detectar automaticamente separador e ignora colunas não numéricas.
    """
    filename = getattr(file_like, "name", "").lower()
    try:
        if filename.endswith(".txt"):
            df = pd.read_csv(
                file_like,
                sep=r"\s+",
                comment="#",
                engine="python",
                header=None,
            )
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
            df = pd.read_csv(
                file_like,
                sep=r"\s+",
                comment="#",
                engine="python",
                header=None,
            )
    except Exception as e:
        raise RuntimeError(f"Erro ao ler arquivo de espectro: {e}")

    df = df.dropna(axis=1, how="all")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        x = numeric_df.iloc[:, 0].to_numpy(dtype=float)
        y = numeric_df.iloc[:, 1].to_numpy(dtype=float)
    else:
        x = df.iloc[:, 0].astype(float).to_numpy()
        y = df.iloc[:, 1].astype(float).to_numpy()
    return x, y

# ---------------------------------------------------------------------
# DESPIKE
# ---------------------------------------------------------------------
def despike(
    y: np.ndarray,
    method: str = "median",
    kernel_size: int = 5,
    z_thresh: float = 6.0,
) -> np.ndarray:
    """
    Remove spikes do espectro por diferentes métodos.
    """
    y = y.copy()
    if method == "median":
        y_filtered = medfilt(y, kernel_size=kernel_size)
        mask = np.abs(y - y_filtered) > (np.std(y) * 3)
        y[mask] = y_filtered[mask]
        return y
    elif method == "zscore":
        mu = pd.Series(y).rolling(
            window=kernel_size, center=True, min_periods=1
        ).median().to_numpy()
        resid = y - mu
        z = np.abs(resid) / (np.std(resid) + 1e-12)
        y[z > z_thresh] = mu[z > z_thresh]
        return y
    elif method == "median_filter_nd":
        return median_filter(y, size=kernel_size)
    else:
        raise ValueError("Método despike desconhecido")

def _despike_metric(y_original: np.ndarray, y_despiked: np.ndarray) -> float:
    second_deriv = np.diff(y_despiked, n=2)
    smooth_term = np.mean(np.abs(second_deriv))
    mse = np.mean((y_original - y_despiked) ** 2)
    alpha = 0.1 / (np.var(y_original) + 1e-12)
    return float(smooth_term + alpha * mse)

def compare_despike_algorithms(
    y: np.ndarray,
    methods: Optional[List[str]] = None,
    kernel_size: int = 5,
):
    """
    Compara algoritmos de despike e retorna:
    (melhor_espectro, melhor_método, dicionário_métricas)
    """
    if methods is None:
        methods = ["median", "zscore", "median_filter_nd"]
    metrics: Dict[str, float] = {}
    best_metric = np.inf
    best_y = y.copy()
    best_method = None
    for m in methods:
        y_d = despike(y, method=m, kernel_size=kernel_size)
        metric = _despike_metric(y, y_d)
        metrics[m] = metric
        if metric < best_metric:
            best_metric = metric
            best_y = y_d
            best_method = m
    return best_y, best_method, metrics

# ---------------------------------------------------------------------
# BASELINE + PRÉ-PROCESSAMENTO
# ---------------------------------------------------------------------
def baseline_als(
    y: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    niter: int = 10,
) -> np.ndarray:
    """
    Baseline ALS estável (versão esparsa).

    Evita erros de broadcast do tipo (L,L) vs (L-2,L-2) na soma W + H.
    """
    y = np.asarray(y, dtype=float)
    L = y.size

    # matriz de segunda derivada (L-2, L)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    w = np.ones(L)

    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.T.dot(D)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return z

def baseline_fft_smooth(y: np.ndarray, cutoff_fraction: float = 0.02) -> np.ndarray:
    Y = np.fft.rfft(y)
    n = len(Y)
    cutoff = max(1, int(n * cutoff_fraction))
    Y[cutoff:-cutoff] = 0
    baseline = np.fft.irfft(Y, n=len(y))
    return baseline

def preprocess_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    despike_method: Optional[str] = "auto_compare",
    smooth: bool = True,
    window_length: int = 9,
    polyorder: int = 3,
    baseline_method: str = "als",
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Pipeline de pré-processamento:
      - despike
      - suavização Savitzky-Golay
      - correção de baseline (ALS ou FFT)
      - normalização (0–1)
    Retorna (x, y_processado, metadados)
    """
    y_proc = y.astype(float).copy()
    meta: Dict[str, Any] = {}

    # despike
    if despike_method == "auto_compare":
        y_proc, best, metrics = compare_despike_algorithms(y_proc)
        meta["despike_method"] = best
        meta["despike_metrics"] = metrics
    elif despike_method is not None:
        y_proc = despike(y_proc, method=despike_method)
        meta["despike_method"] = despike_method

    # smoothing
    if smooth:
        if window_length >= len(y_proc):
            window_length = len(y_proc) - 1
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(window_length, 3)
        y_proc = savgol_filter(
            y_proc, window_length=window_length, polyorder=polyorder
        )
        meta["savgol"] = {
            "window_length": window_length,
            "polyorder": polyorder,
        }

    # baseline
    if baseline_method == "als":
        base = baseline_als(y_proc)
    elif baseline_method == "fft":
        base = baseline_fft_smooth(y_proc)
    else:
        base = np.zeros_like(y_proc)
    y_proc = y_proc - base
    meta["baseline"] = {"method": baseline_method}

    # normalização
    if normalize:
        ymin = float(np.min(y_proc))
        ymax = float(np.max(y_proc))
        if ymax > ymin:
            y_proc = (y_proc - ymin) / (ymax - ymin)
        meta["normalize"] = True

    return x, y_proc, meta

# ---------------------------------------------------------------------
# PICOS
# ---------------------------------------------------------------------
def detect_peaks(
    x: np.ndarray,
    y: np.ndarray,
    height: float = 0.05,
    distance: int = 5,
    prominence: float = 0.02,
) -> List[Peak]:
    """
    Detecta picos no espectro processado.
    """
    indices, _ = find_peaks(
        y, height=height, distance=distance, prominence=prominence
    )
    return [Peak(position_cm1=float(x[i]), intensity=float(y[i])) for i in indices]

def map_peaks_to_molecular_groups(peaks: List[Peak]) -> List[Peak]:
    """
    Atribui grupos moleculares aos picos com base em MOLECULAR_MAP.
    """
    for p in peaks:
        p.group = None
        for item in MOLECULAR_MAP:
            x_min, x_max = item["range"]
            if x_min <= p.position_cm1 <= x_max:
                p.group = item["group"]
                break
    return peaks

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))

def fit_peak_simple(x, y, center, window=10.0):
    """
    Ajuste Gaussiano simples ao redor de um pico (sem lmfit).
    """
    mask = (x >= center - window) & (x <= center + window)
    xi, yi = x[mask], y[mask]
    if len(xi) < 5:
        return {}
    amp0 = float(np.max(yi))
    cen0 = float(center)
    wid0 = 2.0
    try:
        popt, _ = curve_fit(
            gaussian, xi, yi, p0=[amp0, cen0, wid0], maxfev=2000
        )
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

def fit_peaks_lmfit_global(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[Peak],
    model_type: str = "Voigt",
) -> List[Peak]:
    """
    Ajuste global multi-pico usando lmfit (Voigt/Gauss/Lorentz).
    """
    if not LMFIT_AVAILABLE:
        raise RuntimeError("lmfit não está disponível.")
    xmin = min(p.position_cm1 for p in peaks) - 20
    xmax = max(p.position_cm1 for p in peaks) + 20
    mask = (x >= xmin) & (x <= xmax)
    x_fit = x[mask]
    y_fit = y[mask]
    if len(x_fit) < 5:
        return peaks

    def make_model(prefix):
        if model_type.lower() == "gaussian":
            return GaussianModel(prefix=prefix)
        elif model_type.lower() == "lorentzian":
            return LorentzianModel(prefix=prefix)
        else:
            return VoigtModel(prefix=prefix)

    model = None    # type: ignore
    for i, p in enumerate(peaks):
        m = make_model(f"p{i}_")
        model = m if model is None else (model + m)

    params = model.make_params()
    for i, p in enumerate(peaks):
        pref = f"p{i}_"
        cen = p.position_cm1
        amp = max(1e-6, p.intensity * 10.0)
        sigma0 = 3.0
        params[f"{pref}center"].set(value=cen, min=cen - 10, max=cen + 10)
        params[f"{pref}amplitude"].set(value=amp, min=0)
        if f"{pref}sigma" in params:
            params[f"{pref}sigma"].set(value=sigma0, min=0.1, max=50)
        if f"{pref}gamma" in params:
            params[f"{pref}gamma"].set(value=sigma0, min=0.1, max=50)

    result = model.fit(y_fit, params, x=x_fit)

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
            p.intensity = fit_params["amplitude"]
        if "sigma" in fit_params:
            p.width = fit_params["sigma"]
    return peaks

def fit_peaks(x, y, peaks: List[Peak], use_lmfit: bool = True) -> List[Peak]:
    """
    Wrapper: usa lmfit se disponível, senão cai no ajuste simples.
    """
    if use_lmfit and LMFIT_AVAILABLE and len(peaks) > 0:
        return fit_peaks_lmfit_global(x, y, peaks, model_type="Voigt")
    for p in peaks:
        res = fit_peak_simple(x, y, p.position_cm1, window=8.0)
        if res:
            p.fit_params = res["params"]
            if "cen" in res["params"]:
                p.position_cm1 = float(res["params"]["cen"])
            if "amp" in res["params"]:
                p.intensity = float(res["params"]["amp"])
            if "wid" in res["params"]:
                p.width = float(res["params"]["wid"])
    return peaks

def infer_diseases(peaks: List[Peak]):
    """
    Aplica regras simples para sugerir “alterações” com base nos grupos presentes.
    """
    groups_present = {p.group for p in peaks if p.group is not None}
    matches = []
    for rule in DISEASE_RULES:
        required = set(rule["groups_required"])
        score = len(required.intersection(groups_present))
        if score > 0:
            matches.append(
                {
                    "name": rule["name"],
                    "score": score,
                    "description": rule["description"],
                }
            )
    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches

# ---------------------------------------------------------------------
# CALIBRAÇÃO
# ---------------------------------------------------------------------
def apply_base_wavenumber_correction(
    x_obs: np.ndarray,
    base_poly_coeffs: np.ndarray,
) -> np.ndarray:
    """
    Aplica polinômio fixo de calibração (obtido previamente) ao eixo bruto.
    """
    base_poly_coeffs = np.asarray(base_poly_coeffs, dtype=float)
    return np.polyval(base_poly_coeffs, x_obs)

def calibrate_with_fixed_pattern_and_silicon(
    silicon_file,
    sample_file,
    base_poly_coeffs: np.ndarray,
    silicon_ref_position: float = 520.7,
    paper_file=None,
    progress_cb=None,
) -> Dict[str, Any]:
    """
    Calibração com polinômio fixo + Silício.
    Estratégia robusta para localizar pico de Silício:
      - aplica polinômio base
      - tenta janela 480-560 cm-1
      - se não houver pontos, expande janela (400-700)
      - se ainda não houver, busca pico mais próximo de silicon_ref_position
        usando find_peaks sobre espectro suavizado/interpolado
      - se tudo falhar, define delta = 0 e retorna aviso em 'calibration.warning'
    Também subtrai papel (paper_file) se fornecido.
    """
    def tick(p, text=""):
        if progress_cb is not None:
            progress_cb(p, text)

    preprocess_kwargs = {
        "despike_method": "auto_compare",
        "smooth": True,
        "baseline_method": "als",
        "normalize": False,
    }

    # 1) SILÍCIO
    tick(5, "Carregando Silício...")
    x_si_raw, y_si_raw = load_spectrum(silicon_file)

    # pré-processa Silício (mantemos normalize=False para preservar picos)
    x_si, y_si, _ = preprocess_spectrum(x_si_raw, y_si_raw, **preprocess_kwargs)

    tick(20, "Aplicando polinômio base ao Silício...")
    x_si_base = apply_base_wavenumber_correction(x_si, base_poly_coeffs)

    # tentativa 1: janela 480-560
    mask_si = (x_si_base >= 480) & (x_si_base <= 560)

    si_cal_base: Optional[float] = None
    warning: Optional[str] = None

    if np.any(mask_si):
        idx_max = np.argmax(y_si[mask_si])
        x_si_region = x_si_base[mask_si]
        si_cal_base = float(x_si_region[idx_max])
    else:
        # tentativa 2: janela expandida 400-700
        tick(30, "Janela Si vazia — expandindo janela (400–700 cm⁻¹)...")
        mask_si2 = (x_si_base >= 400) & (x_si_base <= 700)
        if np.any(mask_si2):
            idx_max = np.argmax(y_si[mask_si2])
            x_si_region = x_si_base[mask_si2]
            si_cal_base = float(x_si_region[idx_max])
        else:
            # tentativa 3: buscar picos no espectro de silício (suavizado/interpolado)
            tick(45, "Buscando picos de Si por detecção automática...")
            try:
                x_uniform = np.linspace(
                    np.min(x_si_base), np.max(x_si_base),
                    max(800, len(x_si_base))
                )
                y_uniform = np.interp(x_uniform, x_si_base, y_si)

                # suaviza levemente para reduzir ruído
                lw = 11 if len(y_uniform) > 11 else (len(y_uniform) // 2) * 2 + 1
                if lw >= 5:
                    y_smooth = savgol_filter(y_uniform, lw, polyorder=3)
                else:
                    y_smooth = y_uniform

                peaks_idx, props = find_peaks(
                    y_smooth,
                    height=np.max(y_smooth) * 0.1,
                    distance=5,
                    prominence=0.02,
                )
                if len(peaks_idx) > 0:
                    distances = np.abs(x_uniform[peaks_idx] - silicon_ref_position)
                    sel = peaks_idx[np.argmin(distances)]
                    si_cal_base = float(x_uniform[sel])
                else:
                    warning = "Não foi possível localizar pico de Silício automaticamente (nenhum pico detectado)."
            except Exception:
                warning = "Erro ao tentar detectar pico de Silício automaticamente."

    # se ainda não achou, devolve delta = 0 (sem ajuste fino)
    if si_cal_base is None:
        tick(60, "Não encontrado pico Si: usando delta = 0 e prosseguindo (ver warning).")
        delta = 0.0
        warning = warning or "Pico de Silício não encontrado; calibração fina por Si não aplicada."
    else:
        delta = float(silicon_ref_position) - float(si_cal_base)

    def corrector_final(x_arr: np.ndarray) -> np.ndarray:
        x_base = apply_base_wavenumber_correction(x_arr, base_poly_coeffs)
        return x_base + delta

    # 2) AMOSTRA (+ PAPEL OPCIONAL)
    tick(65, "Carregando amostra...")
    x_s_raw, y_s_raw = load_spectrum(sample_file)

    if paper_file is not None:
        tick(70, "Carregando papel (background) e subtraindo...")
        x_p_raw, y_p_raw = load_spectrum(paper_file)
        try:
            y_p_interp = np.interp(x_s_raw, x_p_raw, y_p_raw)
            y_s_raw = y_s_raw - y_p_interp
        except Exception:
            warning = (warning or "") + " Falha ao subtrair papel (interp)."

    # pré-processa amostra
    tick(80, "Pré-processando amostra (despike / baseline / smooth)...")
    x_s, y_s, meta_s = preprocess_spectrum(x_s_raw, y_s_raw, **preprocess_kwargs)

    # aplica correção final ao eixo
    tick(90, "Aplicando correção final ao eixo da amostra...")
    x_s_cal = corrector_final(x_s)

    tick(100, "Calibração concluída.")
    calibration_info: Dict[str, Any] = {
        "base_poly_coeffs": np.asarray(base_poly_coeffs, dtype=float).tolist(),
        "si_cal_base": float(si_cal_base) if si_cal_base is not None else None,
        "silicon_ref_position": float(silicon_ref_position),
        "laser_zero_delta": float(delta),
    }
    if warning:
        calibration_info["warning"] = str(warning)

    return {
        "x_sample_raw": x_s_raw,
        "y_sample_raw": y_s_raw,
        "x_sample_proc": x_s,
        "y_sample_proc": y_s,
        "x_sample_calibrated": x_s_cal,
        "meta_sample": meta_s,
        "calibration": calibration_info,
    }

# ---------------------------------------------------------------------
# EXPORT HDF5
# ---------------------------------------------------------------------
def save_to_nexus_bytes(x: np.ndarray, y: np.ndarray, metadata: Dict[str, Any]) -> bytes:
    """
    Exporta espectro em um HDF5 simples no estilo NeXus.
    """
    if not H5PY_AVAILABLE:
        raise RuntimeError("h5py não instalado.")
    bio = io.BytesIO()
    with h5py.File(bio, "w") as f:
        nxentry = f.create_group("entry")
        nxentry.attrs["NX_class"] = "NXentry"
        nxdata = nxentry.create_group("data")
        nxdata.attrs["NX_class"] = "NXdata"
        nxdata.create_dataset("wavenumber", data=x)
        nxdata.create_dataset("intensity", data=y)
        meta_grp = nxentry.create_group("metadata")
        for k, v in metadata.items():
            meta_grp.attrs[str(k)] = str(v)
    bio.seek(0)
    return bio.read()
