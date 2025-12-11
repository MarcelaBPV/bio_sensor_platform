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

# ---------------------------------------------------------
# MAPA MOLECULAR E REGRAS
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# DATACLASS PARA PICOS
# ---------------------------------------------------------

@dataclass
class Peak:
    position_cm1: float
    intensity: float
    width: Optional[float] = None
    group: Optional[str] = None
    fit_params: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------
# LEITURA DE ESPECTROS
# ---------------------------------------------------------

def load_spectrum(file_like) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lê arquivo de espectro (txt, csv, xls/xlsx) e retorna (x, y).
    Robustecido para lidar com delimitadores variados.
    """
    filename = getattr(file_like, "name", "").lower()

    try:
        if filename.endswith(".txt"):
            df = pd.read_csv(file_like, sep=r"\s+", comment="#", engine="python", header=None)
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
            df = pd.read_csv(file_like, sep=r"\s+", comment="#", engine="python", header=None)

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
# ---------------------------------------------------------
# DESPIKE (remoção de spikes)
# ---------------------------------------------------------

def despike(
    y: np.ndarray,
    method: str = "median",
    kernel_size: int = 5,
    z_thresh: float = 6.0,
) -> np.ndarray:
    """
    Remove spikes do espectro por diferentes métodos.
    Implementado de forma robusta para lidar com espectros ruidosos.
    """
    y = y.copy()

    # Método 1: mediana 1D (padrão, funciona melhor para spikes curtos)
    if method == "median":
        y_filtered = medfilt(y, kernel_size=kernel_size)
        mask = np.abs(y - y_filtered) > (np.std(y) * 3)
        y[mask] = y_filtered[mask]
        return y

    # Método 2: Z-score rolling
    elif method == "zscore":
        series = pd.Series(y)
        mu = series.rolling(window=kernel_size, center=True, min_periods=1).median().to_numpy()
        resid = y - mu
        z = np.abs(resid) / (np.std(resid) + 1e-12)
        y[z > z_thresh] = mu[z > z_thresh]
        return y

    # Método 3: filtro mediano n-dimensional
    elif method == "median_filter_nd":
        return median_filter(y, size=kernel_size)

    else:
        raise ValueError("Método despike desconhecido")


# ---------------------------------------------------------
# MÉTRICA PARA ESCOLHER O MELHOR DESPIKE AUTOMATICAMENTE
# ---------------------------------------------------------

def _despike_metric(y_original: np.ndarray, y_despiked: np.ndarray) -> float:
    """
    Métrica usada para comparar algoritmos de despike:
    combina suavidade (segunda derivada) e fidelidade ao sinal original (MSE).
    Quanto menor, melhor.
    """
    second_deriv = np.diff(y_despiked, n=2)
    smooth_term = np.mean(np.abs(second_deriv))
    mse = np.mean((y_original - y_despiked) ** 2)

    # fator alfa pondera diferença pelo desvio do sinal original
    alpha = 0.1 / (np.var(y_original) + 1e-12)
    return float(smooth_term + alpha * mse)


# ---------------------------------------------------------
# COMPARADOR AUTOMÁTICO DE DESPIKE
# ---------------------------------------------------------

def compare_despike_algorithms(
    y: np.ndarray,
    methods: Optional[List[str]] = None,
    kernel_size: int = 5,
) -> Tuple[np.ndarray, str, Dict[str, float]]:
    """
    Compara algoritmos de despike e retorna:
        - melhor espectro processado,
        - nome do melhor método,
        - dicionário com métricas de todos os métodos.

    Isso permite visualizar barras (métricas) no app.py.
    """
    if methods is None:
        methods = ["median", "zscore", "median_filter_nd"]

    metrics: Dict[str, float] = {}
    best_metric = np.inf
    best_y = y.copy()
    best_method = None

    for m in methods:
        try:
            y_d = despike(y, method=m, kernel_size=kernel_size)
            metric = _despike_metric(y, y_d)
            metrics[m] = metric

            if metric < best_metric:
                best_metric = metric
                best_y = y_d
                best_method = m
        except Exception:
            metrics[m] = np.inf  # método falhou

    return best_y, best_method, metrics
# ---------------------------------------------------------
# BASELINE (ALS esparsa) e FFT-smoothing
# ---------------------------------------------------------

def baseline_als(
    y: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    niter: int = 10,
) -> np.ndarray:
    """
    Baseline ALS usando matriz esparsa para maior eficiência.
    Retorna o baseline z do mesmo tamanho que y.
    """
    y = np.asarray(y, dtype=float)
    L = y.size
    if L < 3:
        return np.zeros_like(y)

    # matriz D (segunda diferença) em formato esparso
    # shape (L-2, L)
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(L - 2, L))
    w = np.ones(L, dtype=float)

    for _ in range(max(1, int(niter))):
        # construir W e Z = W + lam * D^T D
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * (D.T @ D)
        # spsolve espera CSC/CSR — spsolve cuidará internamente, mas converte quando necessário
        z = spsolve(Z.tocsc(), w * y)
        # atualizar pesos
        w = p * (y > z) + (1.0 - p) * (y < z)

    return z


def baseline_fft_smooth(y: np.ndarray, cutoff_fraction: float = 0.02) -> np.ndarray:
    """
    Estima baseline por suavização no domínio da frequência.
    Remove componentes de alta frequência e reconstrói.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 4:
        return np.zeros_like(y)
    Y = np.fft.rfft(y)
    n = Y.size
    cutoff = max(1, int(n * cutoff_fraction))
    Y[cutoff:-cutoff] = 0
    baseline = np.fft.irfft(Y, n=len(y))
    return baseline


# ---------------------------------------------------------
# PREPROCESSAMENTO COMPLETO (DESPIKE + SG + BASELINE + NORMALIZE)
# ---------------------------------------------------------

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
      - despike (auto-compare ou método definido)
      - Savitzky-Golay smoothing
      - baseline (ALS ou FFT)
      - normalização 0..1

    Retorna (x, y_processado, meta)
    """
    y_proc = np.asarray(y, dtype=float).copy()
    meta: Dict[str, Any] = {}

    # 1) despike
    if despike_method == "auto_compare":
        try:
            y_proc, best, metrics = compare_despike_algorithms(y_proc)
            meta["despike_method"] = best
            meta["despike_metrics"] = metrics
        except Exception:
            # fallback simples
            y_proc = despike(y_proc, method="median")
            meta["despike_method"] = "median_fallback"
    elif despike_method is not None:
        y_proc = despike(y_proc, method=despike_method)
        meta["despike_method"] = despike_method

    # 2) smoothing Savitzky-Golay
    if smooth:
        wl = int(window_length)
        if wl >= len(y_proc):
            wl = len(y_proc) - 1
        if wl % 2 == 0:
            wl += 1
        wl = max(wl, 3)
        try:
            y_proc = savgol_filter(y_proc, window_length=wl, polyorder=polyorder)
            meta["savgol"] = {"window_length": wl, "polyorder": polyorder}
        except Exception:
            # se falhar, mantém o sinal original
            meta["savgol"] = {"error": "savgol failed"}

    # 3) baseline
    if baseline_method == "als":
        try:
            base = baseline_als(y_proc)
        except Exception:
            base = np.zeros_like(y_proc)
            meta["baseline_error"] = "als_failed"
    elif baseline_method == "fft":
        base = baseline_fft_smooth(y_proc)
    else:
        base = np.zeros_like(y_proc)

    y_proc = y_proc - base
    meta["baseline"] = {"method": baseline_method}

    # 4) normalização 0..1
    if normalize:
        ymin = float(np.min(y_proc))
        ymax = float(np.max(y_proc))
        if (ymax - ymin) > 0:
            y_proc = (y_proc - ymin) / (ymax - ymin)
            meta["normalize"] = True
        else:
            meta["normalize"] = False

    return x, y_proc, meta


# ---------------------------------------------------------
# DETECÇÃO DE PICOS E MAPEAMENTO (início)
# ---------------------------------------------------------

def detect_peaks(
    x: np.ndarray,
    y: np.ndarray,
    height: float = 0.05,
    distance: int = 5,
    prominence: float = 0.02,
) -> List[Peak]:
    """
    Detecta picos no espectro normalizado/processado.
    Retorna lista de objetos Peak.
    """
    if len(x) == 0 or len(y) == 0:
        return []
    indices, _ = find_peaks(y, height=height, distance=distance, prominence=prominence)
    peaks: List[Peak] = []
    for i in indices:
        try:
            peaks.append(Peak(position_cm1=float(x[i]), intensity=float(y[i])))
        except Exception:
            continue
    return peaks


def map_peaks_to_molecular_groups(peaks: List[Peak]) -> List[Peak]:
    """
    Atribui grupo molecular de acordo com MOLECULAR_MAP.
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
    """
    Gaussiana auxiliar (usada também no plot final).
    """
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))


def fit_peak_simple(x, y, center, window=10.0):
    """
    Ajuste gaussiano simples ao redor de 'center' (sem lmfit).
    Retorna dicionário com params ou {} se falhar.
    """
    mask = (x >= center - window) & (x <= center + window)
    xi, yi = x[mask], y[mask]
    if len(xi) < 5:
        return {}
    amp0 = float(np.max(yi))
    cen0 = float(center)
    wid0 = max(0.5, (xi.max() - xi.min()) / 6.0)
    try:
        popt, _ = curve_fit(gaussian, xi, yi, p0=[amp0, cen0, wid0], maxfev=3000)
        return {"model": "gaussian", "params": {"amp": float(popt[0]), "cen": float(popt[1]), "wid": float(popt[2])}}
    except Exception:
        return {}
# ---------------------------------------------------------
# AJUSTE MULTI-PIKO (lmfit) + WRAPPER
# ---------------------------------------------------------

def fit_peaks_lmfit_global(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[Peak],
    model_type: str = "Voigt",
) -> List[Peak]:
    """
    Ajuste global multi-pico usando lmfit (Voigt/Gauss/Lorentz).
    Retorna lista de Peak com fit_params preenchidos quando possível.
    """
    if not LMFIT_AVAILABLE:
        raise RuntimeError("lmfit não está disponível.")
    if len(peaks) == 0:
        return peaks

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

    model = None
    for i, p in enumerate(peaks):
        m = make_model(f"p{i}_")
        model = m if model is None else (model + m)

    params = model.make_params()
    for i, p in enumerate(peaks):
        pref = f"p{i}_"
        cen = p.position_cm1
        amp = max(1e-6, p.intensity * 10.0)
        sigma0 = 3.0
        # alguns prefixes podem variar dependendo do model criado; adjust safe
        if f"{pref}center" in params:
            params[f"{pref}center"].set(value=cen, min=cen - 10, max=cen + 10)
        if f"{pref}amplitude" in params:
            params[f"{pref}amplitude"].set(value=amp, min=0)
        if f"{pref}sigma" in params:
            params[f"{pref}sigma"].set(value=sigma0, min=0.1, max=50)
        if f"{pref}gamma" in params:
            params[f"{pref}gamma"].set(value=sigma0, min=0.1, max=50)

    # ajuste com captura de erro para evitar crash do app
    try:
        result = model.fit(y_fit, params, x=x_fit)
    except Exception as e:
        # se lmfit falhar, retorna peaks originais sem fit_params
        return peaks

    for i, p in enumerate(peaks):
        pref = f"p{i}_"
        fit_params = {}
        for name, val in result.params.items():
            if name.startswith(pref):
                fit_params[name.replace(pref, "")] = float(val.value)
        p.fit_params = fit_params
        # padroniza nomes para compatibilidade com fit_peak_simple
        if "center" in fit_params:
            p.position_cm1 = fit_params["center"]
        if "amplitude" in fit_params:
            p.intensity = fit_params["amplitude"]
        if "sigma" in fit_params:
            p.width = fit_params["sigma"]
    return peaks


def fit_peaks(x: np.ndarray, y: np.ndarray, peaks: List[Peak], use_lmfit: bool = True) -> List[Peak]:
    """
    Wrapper: tenta lmfit (se disponível e solicitado), senão aplica ajuste simples por pico.
    """
    if use_lmfit and LMFIT_AVAILABLE and len(peaks) > 0:
        try:
            return fit_peaks_lmfit_global(x, y, peaks, model_type="Voigt")
        except Exception:
            # fallback silencioso
            pass

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


# ---------------------------------------------------------
# INFERÊNCIA DE DOENÇAS (REGRAS SIMPLES)
# ---------------------------------------------------------

def infer_diseases(peaks: List[Peak]):
    """
    Aplica regras definidas em DISEASE_RULES para sugerir possíveis condições.
    Retorna lista de dicionários ordenada por score.
    """
    groups_present = {p.group for p in peaks if p.group is not None}
    matches = []
    for rule in DISEASE_RULES:
        required = set(rule.get("groups_required", []))
        score = len(required.intersection(groups_present))
        if score > 0:
            matches.append({
                "name": rule["name"],
                "score": score,
                "required_groups": required,
                "description": rule.get("description", "")
            })
    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches


# ---------------------------------------------------------
# CALIBRAÇÃO: polinômio base + ajuste com pico de Silício (e subtração de papel)
# ---------------------------------------------------------

def apply_base_wavenumber_correction(
    x_obs: np.ndarray,
    base_poly_coeffs: np.ndarray,
) -> np.ndarray:
    """
    Converte eixo bruto usando coeficientes do polinômio (np.polyval style).
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
    Workflow de calibração robusto:
      - lê Silício, aplica polinômio base e tenta localizar pico de Si;
      - lê amostra, subtrai papel (se fornecido), pré-processa e aplica correção final;
      - devolve eixos raw, proc e calibrado + metadados de calibração.
    Em vez de lançar erro quando não encontra o pico de Si, preenche 'warning'.
    """
    def tick(p, text=""):
        if progress_cb is not None:
            progress_cb(p, text)

    preprocess_kwargs = dict(
        despike_method="auto_compare",
        smooth=True,
        baseline_method="als",
        normalize=False,
    )

    tick(5, "Carregando Silício...")
    x_si_raw, y_si_raw = load_spectrum(silicon_file)
    x_si, y_si, _ = preprocess_spectrum(x_si_raw, y_si_raw, **preprocess_kwargs)

    tick(20, "Aplicando polinômio base ao Silício...")
    x_si_base = apply_base_wavenumber_correction(x_si, base_poly_coeffs)

    si_cal_base = None
    warning = None

    # estratégia em camadas para localizar Si
    mask_si = (x_si_base >= 480) & (x_si_base <= 560)
    if np.any(mask_si):
        idx_max = np.argmax(y_si[mask_si])
        x_si_region = x_si_base[mask_si]
        si_cal_base = float(x_si_region[idx_max])
    else:
        # expandir janela
        mask_si2 = (x_si_base >= 400) & (x_si_base <= 700)
        if np.any(mask_si2):
            idx_max = np.argmax(y_si[mask_si2])
            x_si_region = x_si_base[mask_si2]
            si_cal_base = float(x_si_region[idx_max])
        else:
            # tentativa automática por detecção de picos no sinal reamostrado
            try:
                x_uniform = np.linspace(np.min(x_si_base), np.max(x_si_base), max(600, len(x_si_base)))
                y_uniform = np.interp(x_uniform, x_si_base, y_si)
                lw = 11 if len(y_uniform) > 11 else (len(y_uniform) // 2) * 2 + 1
                if lw >= 5:
                    y_smooth = savgol_filter(y_uniform, lw, polyorder=3)
                else:
                    y_smooth = y_uniform
                peaks_idx, _ = find_peaks(y_smooth, height=np.max(y_smooth) * 0.1, distance=5, prominence=0.02)
                if len(peaks_idx) > 0:
                    sel = peaks_idx[np.argmin(np.abs(x_uniform[peaks_idx] - silicon_ref_position))]
                    si_cal_base = float(x_uniform[sel])
                else:
                    warning = "Pico de Silício não detectado automaticamente."
            except Exception:
                warning = "Erro ao detectar pico de Silício automaticamente."

    if si_cal_base is None:
        delta = 0.0
        warning = warning or "Pico de Silício não encontrado; usando delta=0."
    else:
        delta = float(silicon_ref_position) - float(si_cal_base)

    def corrector_final(x_arr: np.ndarray) -> np.ndarray:
        x_base = apply_base_wavenumber_correction(x_arr, base_poly_coeffs)
        return x_base + delta

    # AMOSTRA + SUBTRAÇÃO DE PAPEL (se houver)
    tick(60, "Carregando amostra...")
    x_s_raw, y_s_raw = load_spectrum(sample_file)

    if paper_file is not None:
        tick(65, "Carregando papel (background) e aplicando subtração por interp...")
        try:
            x_p_raw, y_p_raw = load_spectrum(paper_file)
            y_p_interp = np.interp(x_s_raw, x_p_raw, y_p_raw)
            y_s_raw = y_s_raw - y_p_interp
        except Exception as e:
            warning = (warning or "") + f" Falha na subtração do papel: {e}"

    tick(80, "Pré-processando amostra...")
    x_s, y_s, meta_s = preprocess_spectrum(x_s_raw, y_s_raw, **preprocess_kwargs)

    tick(90, "Aplicando correção final ao eixo calibrado...")
    x_s_cal = corrector_final(x_s)

    tick(100, "Concluído.")
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


# ---------------------------------------------------------
# EXPORT HDF5 (NeXus-like)
# ---------------------------------------------------------

def save_to_nexus_bytes(x: np.ndarray, y: np.ndarray, metadata: Dict[str, Any]) -> bytes:
    """
    Exporta um HDF5 simples (NeXus-like) em bytes.
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
# ---------------------------------------------------------
# FUNÇÕES AUXILIARES DE APOIO (CONSISTÊNCIA, SEGURANÇA)
# ---------------------------------------------------------

def safe_min(arr: np.ndarray) -> float:
    """Retorna mínimo seguro mesmo se array estiver vazio."""
    try:
        return float(np.min(arr))
    except Exception:
        return 0.0

def safe_max(arr: np.ndarray) -> float:
    """Retorna máximo seguro mesmo se array estiver vazio."""
    try:
        return float(np.max(arr))
    except Exception:
        return 0.0

def ensure_sorted(x: np.ndarray, y: np.ndarray):
    """Garante que o eixo x está em ordem crescente."""
    if len(x) < 2:
        return x, y
    if np.any(np.diff(x) < 0):
        idx = np.argsort(x)
        return x[idx], y[idx]
    return x, y

# ---------------------------------------------------------
# FUNÇÕES DE DIAGNÓSTICO / DEBUG OPCIONAL
# ---------------------------------------------------------

def describe_spectrum(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Retorna estatísticas simples do espectro — útil para debug.
    """
    return {
        "x_min": safe_min(x),
        "x_max": safe_max(x),
        "y_min": safe_min(y),
        "y_max": safe_max(y),
        "n_points": int(len(x)),
    }

def print_peak_summary(peaks: List[Peak]):
    """
    Imprime no console um resumo rápido dos picos detectados.
    (Não aparece no Streamlit, apenas em logs locais.)
    """
    print("\n=== Peak Summary ===")
    for p in peaks:
        print(
            f"Peak @ {p.position_cm1:.1f} cm⁻¹ | Int={p.intensity:.3f} | "
            f"Width={p.width} | Group={p.group}"
        )
    print("====================\n")

# ---------------------------------------------------------
# FUNÇÃO PARA PADRONIZAR PEAKS EM DATAFRAME
# ---------------------------------------------------------

def peaks_to_dataframe(peaks: List[Peak]) -> pd.DataFrame:
    """
    Converte lista de Peak em DataFrame padronizado.
    """
    if not peaks:
        return pd.DataFrame(columns=["position_cm-1", "intensity", "width", "group"])

    rows = []
    for p in peaks:
        rows.append({
            "position_cm-1": float(p.position_cm1),
            "intensity": float(p.intensity),
            "width": float(p.width) if p.width else None,
            "group": p.group or "Sem classificação",
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------
# FUNÇÃO PARA AGREGAÇÃO DE GRUPOS → TABELA
# ---------------------------------------------------------

def summarize_groups(peaks: List[Peak]) -> pd.DataFrame:
    """
    Cria tabela de contagem e % de picos por grupo molecular.
    """
    df = peaks_to_dataframe(peaks)
    if df.empty:
        return pd.DataFrame(
            columns=["group", "n_peaks", "pct_of_peaks", "linked_diseases"]
        )

    total = max(1, len(df))
    group_counts = df["group"].value_counts(dropna=False).reset_index()
    group_counts.columns = ["group", "n_peaks"]
    group_counts["pct_of_peaks"] = (group_counts["n_peaks"] / total * 100).round(1)

    # doenças relacionadas a cada grupo
    def diseases_for_group(gname: str):
        return [
            rule["name"]
            for rule in DISEASE_RULES
            if gname in rule.get("groups_required", [])
        ]

    group_counts["linked_diseases"] = group_counts["group"].apply(diseases_for_group)
    return group_counts

# ---------------------------------------------------------
# FUNÇÃO PARA MATRIZ DE CORRELAÇÃO GRUPO–DOENÇA
# ---------------------------------------------------------

def group_disease_correlation(peaks: List[Peak]) -> pd.DataFrame:
    """
    Calcula correlação percentual entre grupos presentes vs. grupos exigidos nas regras.
    """
    df = peaks_to_dataframe(peaks)
    present = set(df["group"].unique())

    rows = []
    for rule in DISEASE_RULES:
        required = set(rule["groups_required"])
        n_present = len(required.intersection(present))
        total_required = len(required) if len(required) > 0 else 1
        score_pct = round((n_present / total_required) * 100, 1)
        rows.append({
            "disease": rule["name"],
            "required_groups": ", ".join(required),
            "n_required_present": n_present,
            "n_required_total": total_required,
            "correlation_%": score_pct,
            "description": rule.get("description", "")
        })

    return pd.DataFrame(rows).sort_values("correlation_%", ascending=False)

# ---------------------------------------------------------
# FINALIZAÇÃO — UTILIDADES EXPORTADAS
# ---------------------------------------------------------

__all__ = [
    "Peak",
    "load_spectrum",
    "despike",
    "compare_despike_algorithms",
    "baseline_als",
    "preprocess_spectrum",
    "detect_peaks",
    "fit_peaks",
    "map_peaks_to_molecular_groups",
    "infer_diseases",
    "calibrate_with_fixed_pattern_and_silicon",
    "save_to_nexus_bytes",
    # utilidades extras
    "peaks_to_dataframe",
    "summarize_groups",
    "group_disease_correlation",
    "describe_spectrum",
]
# ---------------------------------------------------------
# APLICA CORREÇÃO POLINOMIAL AO EIXO
# ---------------------------------------------------------

def apply_base_wavenumber_correction(x_obs: np.ndarray, base_poly_coeffs: np.ndarray) -> np.ndarray:
    """Aplica polinômio fixo de calibração (obtido previamente)."""
    base_poly_coeffs = np.asarray(base_poly_coeffs, dtype=float)
    return np.polyval(base_poly_coeffs, x_obs)


# ---------------------------------------------------------
# CALIBRAÇÃO COM POLINÔMIO + SILÍCIO + PAPEL
# ---------------------------------------------------------

def calibrate_with_fixed_pattern_and_silicon(
    silicon_file,
    sample_file,
    base_poly_coeffs: np.ndarray,
    silicon_ref_position: float = 520.7,
    paper_file=None,
    progress_cb=None,
) -> Dict[str, Any]:
    """
    Calibração robusta:
      1. lẽ Si
      2. aplica polinômio base
      3. busca pico de Si em janelas sucessivas
      4. calcula delta = ref - pico_encontrado
      5. aplica delta ao eixo da amostra
      6. se paper_file for fornecido → subtrai background
    """

    def tick(p, text=""):
        if progress_cb is not None:
            progress_cb(p, text)

    preprocess_kwargs = dict(
        despike_method="auto_compare",
        smooth=True,
        baseline_method="als",
        normalize=False,
    )

    # -----------------------------
    # 1) LER SILÍCIO
    # -----------------------------
    tick(5, "Lendo espectro de Silício…")
    x_si_raw, y_si_raw = load_spectrum(silicon_file)
    x_si, y_si, _ = preprocess_spectrum(x_si_raw, y_si_raw, **preprocess_kwargs)

    # aplicação do polinômio
    x_si_base = apply_base_wavenumber_correction(x_si, base_poly_coeffs)

    # -----------------------------
    # 2) DETECTAR PICO DO SILÍCIO
    # -----------------------------
    warning = None
    si_peak_est = None

    # janela principal
    mask = (x_si_base >= 480) & (x_si_base <= 560)
    if np.any(mask):
        region = x_si_base[mask]
        idx_max = np.argmax(y_si[mask])
        si_peak_est = float(region[idx_max])

    # janela expandida
    if si_peak_est is None:
        mask2 = (x_si_base >= 400) & (x_si_base <= 700)
        if np.any(mask2):
            region = x_si_base[mask2]
            idx_max = np.argmax(y_si[mask2])
            si_peak_est = float(region[idx_max])

    # detecção automática
    if si_peak_est is None:
        try:
            x_uni = np.linspace(x_si_base.min(), x_si_base.max(), max(800, len(x_si_base)))
            y_uni = np.interp(x_uni, x_si_base, y_si)

            lw = 11 if len(y_uni) > 11 else (len(y_uni)//2)*2 + 1
            y_sm = savgol_filter(y_uni, lw, polyorder=3)

            peaks_idx, props = find_peaks(
                y_sm,
                height=np.max(y_sm)*0.1,
                distance=5,
                prominence=0.02,
            )
            if len(peaks_idx) > 0:
                d = np.abs(x_uni[peaks_idx] - silicon_ref_position)
                sel = peaks_idx[np.argmin(d)]
                si_peak_est = float(x_uni[sel])
            else:
                warning = "Nenhum pico do Si detectado automaticamente."
        except Exception:
            warning = "Erro ao tentar detectar pico de Silício automaticamente."

    # -----------------------------
    # 3) CALCULAR DELTA
    # -----------------------------

    if si_peak_est is None:
        delta = 0.0
        warning = warning or "Pico de Si não localizado; delta=0 usado."
    else:
        delta = float(silicon_ref_position) - float(si_peak_est)

    def corrector(x_arr):
        xb = apply_base_wavenumber_correction(x_arr, base_poly_coeffs)
        return xb + delta

    # -----------------------------
    # 4) LER AMOSTRA (+ papel)
    # -----------------------------
    tick(70, "Lendo espectro da amostra…")
    x_raw, y_raw = load_spectrum(sample_file)

    if paper_file is not None:
        tick(75, "Subtraindo background do papel…")
        try:
            x_p, y_p = load_spectrum(paper_file)
            y_p_interp = np.interp(x_raw, x_p, y_p)
            y_raw = y_raw - y_p_interp
        except Exception:
            warning = (warning or "") + " Falha ao subtrair papel; ignorado."

    # pré-processar
    tick(85, "Pré-processando amostra…")
    x_proc, y_proc, meta = preprocess_spectrum(x_raw, y_raw, **preprocess_kwargs)

    # aplicar correção final
    tick(95, "Aplicando calibração final…")
    x_cal = corrector(x_proc)

    tick(100, "Calibração concluída.")

    # empacotar tudo
    calib_info = dict(
        base_poly_coeffs=np.asarray(base_poly_coeffs).tolist(),
        si_peak_est=si_peak_est,
        silicon_ref=float(silicon_ref_position),
        delta=float(delta),
    )
    if warning:
        calib_info["warning"] = warning

    return dict(
        x_sample_raw=x_raw,
        y_sample_raw=y_raw,
        x_sample_proc=x_proc,
        y_sample_proc=y_proc,
        x_sample_calibrated=x_cal,
        meta_sample=meta,
        calibration=calib_info,
    )
# ---------------------------------------------------------
# EXPORTAÇÃO HDF5 (NeXus-like)
# ---------------------------------------------------------

def save_to_nexus_bytes(x: np.ndarray, y: np.ndarray, metadata: Dict[str, Any]) -> bytes:
    """
    Exporta espectro para um arquivo HDF5 simples no formato NeXus.
    """
    if not H5PY_AVAILABLE:
        raise RuntimeError("h5py não está instalado no ambiente.")

    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"

        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"

        data.create_dataset("wavenumber", data=x)
        data.create_dataset("intensity", data=y)

        meta_grp = entry.create_group("metadata")
        for k, v in metadata.items():
            meta_grp.attrs[str(k)] = str(v)

    buf.seek(0)
    return buf.read()
# ---------------------------------------------------------
# EXPORTAÇÃO DOS SÍMBOLOS (PARA IMPORTAÇÃO CONTROLADA)
# ---------------------------------------------------------

__all__ = [
    "Peak",
    "MOLECULAR_MAP",
    "DISEASE_RULES",
    "load_spectrum",
    "despike",
    "compare_despike_algorithms",
    "baseline_als",
    "baseline_fft_smooth",
    "preprocess_spectrum",
    "detect_peaks",
    "fit_peaks",
    "fit_peaks_lmfit_global",
    "map_peaks_to_molecular_groups",
    "infer_diseases",
    "apply_base_wavenumber_correction",
    "calibrate_with_fixed_pattern_and_silicon",
    "save_to_nexus_bytes",
    # tabelas
    "peaks_to_dataframe",
    "summarize_groups",
    "group_disease_correlation",
]
