# appy.py
# -*- coding: utf-8 -*-
"""
Streamlit app para harmonização de espectros Raman:
- Comparador de despike (A)
- Ajuste multi-peak com lmfit (B) quando disponível
- Workflow de calibração CWA-like com Neon / Poliestireno / Silício (C)
- Export HDF5 (NeXus-like)
"""

import io
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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

# =====================================================================
# CONFIG STREAMLIT
# =====================================================================
st.set_page_config(page_title="Raman Harmonization", layout="wide")

# =====================================================================
# TABELAS PADRÃO DE REFERÊNCIA (ajuste conforme seu setup!)
# =====================================================================

# ⚠️ Estes valores são EXEMPLOS típicos. Ajuste para o seu laser/padrão.
# Poliestireno – picos Raman mais usados para calibração (cm-1)
POLYSTYRENE_REF_CM1 = np.array([
    620.0,
    1001.4,
    1031.0,
    1157.0,
    1584.0,
    1602.0,
])

# Neon – posições de “picos” em cm-1 (Raman shift) aqui são genéricas
# e precisam ser ajustadas à sua configuração experimental. Em muitos
# trabalhos usam o espectro de emissão e convertem para shift.
NEON_REF_CM1 = np.array([
    540.0,
    585.2,
    703.2,
    743.9,
    810.0,
])

# =====================================================================
# MAPA MOLECULAR E REGRAS
# =====================================================================
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


# =====================================================================
# DATA CLASS
# =====================================================================
@dataclass
class Peak:
    position_cm1: float
    intensity: float
    width: Optional[float] = None
    group: Optional[str] = None
    fit_params: Optional[Dict[str, Any]] = None


# =====================================================================
# FUNÇÕES AUXILIARES – CARREGAMENTO
# =====================================================================
def load_spectrum(file_like) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lê arquivo de espectro e retorna (x, y).
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
        raise RuntimeError(f"Erro ao ler arquivo: {e}")

    df = df.dropna(axis=1, how="all")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        x = numeric_df.iloc[:, 0].to_numpy(dtype=float)
        y = numeric_df.iloc[:, 1].to_numpy(dtype=float)
    else:
        x = df.iloc[:, 0].astype(float).to_numpy()
        y = df.iloc[:, 1].astype(float).to_numpy()
    return x, y


# =====================================================================
# DESPIKE + COMPARADOR
# =====================================================================
def despike(y: np.ndarray, method: str = "median", kernel_size: int = 5, z_thresh: float = 6.0) -> np.ndarray:
    y = y.copy()
    if method == "median":
        y_filtered = medfilt(y, kernel_size=kernel_size)
        mask = np.abs(y - y_filtered) > (np.std(y) * 3)
        y[mask] = y_filtered[mask]
        return y
    elif method == "zscore":
        mu = pd.Series(y).rolling(window=kernel_size, center=True, min_periods=1).median().to_numpy()
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


def compare_despike_algorithms(y: np.ndarray, methods: Optional[List[str]] = None, kernel_size: int = 5):
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


# =====================================================================
# BASELINE
# =====================================================================
def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
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
    Y = np.fft.rfft(y)
    n = len(Y)
    cutoff = max(1, int(n * cutoff_fraction))
    Y[cutoff:-cutoff] = 0
    baseline = np.fft.irfft(Y, n=len(y))
    return baseline


# =====================================================================
# PRÉ-PROCESSAMENTO
# =====================================================================
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
        y_proc = savgol_filter(y_proc, window_length=window_length, polyorder=polyorder)
        meta["savgol"] = {"window_length": window_length, "polyorder": polyorder}

    # baseline
    if baseline_method == "als":
        base = baseline_als(y_proc)
    elif baseline_method == "fft":
        base = baseline_fft_smooth(y_proc)
    else:
        base = np.zeros_like(y_proc)
    y_proc = y_proc - base
    meta["baseline"] = {"method": baseline_method}

    # normalize
    if normalize:
        ymin = float(np.min(y_proc))
        ymax = float(np.max(y_proc))
        if ymax > ymin:
            y_proc = (y_proc - ymin) / (ymax - ymin)
        meta["normalize"] = True

    return x, y_proc, meta


# =====================================================================
# DETECÇÃO + FIT DE PICOS (lmfit se disponível)
# =====================================================================
def detect_peaks(
    x: np.ndarray,
    y: np.ndarray,
    height: float = 0.05,
    distance: int = 5,
    prominence: float = 0.02,
) -> List[Peak]:
    indices, _ = find_peaks(y, height=height, distance=distance, prominence=prominence)
    return [Peak(position_cm1=float(x[i]), intensity=float(y[i])) for i in indices]


def map_peaks_to_molecular_groups(peaks: List[Peak]) -> List[Peak]:
    for peak in peaks:
        for item in MOLECULAR_MAP:
            x_min, x_max = item["range"]
            if x_min <= peak.position_cm1 <= x_max:
                peak.group = item["group"]
                break
    return peaks


def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))


def fit_peak_simple(x, y, center, window=10.0):
    mask = (x >= center - window) & (x <= center + window)
    xi, yi = x[mask], y[mask]
    if len(xi) < 5:
        return {}
    amp0 = float(np.max(yi))
    cen0 = float(center)
    wid0 = 2.0
    try:
        popt, _ = curve_fit(gaussian, xi, yi, p0=[amp0, cen0, wid0], maxfev=2000)
        return {"model": "gaussian", "params": {"amp": float(popt[0]), "cen": float(popt[1]), "wid": float(popt[2])}}
    except Exception:
        return {}


def fit_peaks_lmfit_global(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[Peak],
    model_type: str = "Voigt",
) -> List[Peak]:
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


# =====================================================================
# REGRAS DE PADRÕES
# =====================================================================
def infer_diseases(peaks: List[Peak]):
    groups_present = {p.group for p in peaks if p.group is not None}
    matches = []
    for rule in DISEASE_RULES:
        required = set(rule["groups_required"])
        score = len(required.intersection(groups_present))
        if score > 0:
            matches.append({"name": rule["name"], "score": score, "description": rule["description"]})
    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches


# =====================================================================
# CALIBRAÇÃO
# =====================================================================
def calibrate_wavenumber(observed_positions: np.ndarray, reference_positions: np.ndarray, degree: int = 1):
    if len(observed_positions) < degree + 1:
        raise ValueError("Pontos insuficientes para calibrar com esse grau.")
    coeffs = np.polyfit(observed_positions, reference_positions, deg=degree)
    def corrector(x_obs):
        return np.polyval(coeffs, x_obs)
    return corrector, coeffs


def _match_peaks_to_refs(peak_positions: np.ndarray, ref_positions: np.ndarray, max_diff: float = 15.0):
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
    progress_cb=None,
) -> Dict[str, Any]:
    """
    Workflow completo de calibração CWA-like.
    progress_cb: função opcional para atualizar barra de progresso (0–100).
    """
    preprocess_kwargs = {
        "despike_method": "auto_compare",
        "smooth": True,
        "baseline_method": "als",
        "normalize": False,
    }

    def _tick(pct, text=""):
        if progress_cb is not None:
            progress_cb(pct, text)

    # 1) NEON
    _tick(10, "Carregando Neon...")
    x_neon_raw, y_neon_raw = load_spectrum(neon_file)
    x_neon, y_neon, _ = preprocess_spectrum(x_neon_raw, y_neon_raw, **preprocess_kwargs)
    _tick(20, "Detectando picos em Neon...")
    neon_peaks = detect_peaks(x_neon, y_neon, height=0.1, distance=3, prominence=0.05)
    neon_positions = np.array([p.position_cm1 for p in neon_peaks])

    obs_neon, ref_neon = _match_peaks_to_refs(neon_positions, neon_ref_positions)

    # 2) POLIESTIRENO
    _tick(35, "Carregando Poliestireno...")
    x_poly_raw, y_poly_raw = load_spectrum(polystyrene_file)
    x_poly, y_poly, _ = preprocess_spectrum(x_poly_raw, y_poly_raw, **preprocess_kwargs)
    _tick(45, "Detectando picos em Poliestireno...")
    poly_peaks = detect_peaks(x_poly, y_poly, height=0.1, distance=3, prominence=0.05)
    poly_positions = np.array([p.position_cm1 for p in poly_peaks])

    obs_poly, ref_poly = _match_peaks_to_refs(poly_positions, poly_ref_positions)

    obs_all = np.array(obs_neon + obs_poly, dtype=float)
    ref_all = np.array(ref_neon + ref_poly, dtype=float)

    if len(obs_all) < poly_degree + 1:
        raise RuntimeError("Pontos de calibração insuficientes para o grau do polinômio.")

    _tick(55, "Ajustando polinômio de calibração (Neon + Poli)...")
    corrector_base, coeffs_base = calibrate_wavenumber(obs_all, ref_all, degree=poly_degree)

    # 3) SILÍCIO
    _tick(65, "Carregando Silício (zero do laser)...")
    x_si_raw, y_si_raw = load_spectrum(silicon_file)
    x_si, y_si, _ = preprocess_spectrum(x_si_raw, y_si_raw, **preprocess_kwargs)
    mask_si = (x_si >= 480) & (x_si <= 560)
    if not np.any(mask_si):
        raise RuntimeError("Sem dados na janela de silício (480–560 cm-1).")
    idx_max = np.argmax(y_si[mask_si])
    x_si_region = x_si[mask_si]
    si_obs_position = float(x_si_region[idx_max])

    si_cal_base = float(corrector_base(np.array([si_obs_position]))[0])
    delta = silicon_ref_position - si_cal_base

    def corrector_final(x_arr):
        return corrector_base(x_arr) + delta

    # 4) AMOSTRA
    _tick(75, "Carregando espectro da amostra...")
    x_s_raw, y_s_raw = load_spectrum(sample_file)
    x_s, y_s, meta_s = preprocess_spectrum(x_s_raw, y_s_raw, **preprocess_kwargs)

    _tick(85, "Aplicando calibração à amostra...")
    x_s_cal = corrector_final(x_s)

    _tick(100, "Calibração concluída.")
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


# =====================================================================
# EXPORT HDF5 (NeXus-like)
# =====================================================================
def save_to_nexus_bytes(x: np.ndarray, y: np.ndarray, metadata: Dict[str, Any]) -> bytes:
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


# =====================================================================
# UI STREAMLIT
# =====================================================================
st.title("Raman Harmonization — Pipeline tipo ramanchada2")
st.caption("Upload da amostra + padrões (Neon, Poliestireno, Silício) • Pré-processamento • Calibração • Detecção & Fit de picos")

# Layout principal com colunas
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Arquivos de entrada")
    sample_file = st.file_uploader("Amostra (sample)", type=["txt", "csv", "xlsx"])
    neon_file = st.file_uploader("Neon (padrão de emissão)", type=["txt", "csv", "xlsx"])
    poly_file = st.file_uploader("Poliestireno (padrão Raman)", type=["txt", "csv", "xlsx"])
    si_file = st.file_uploader("Silício (zero do laser)", type=["txt", "csv", "xlsx"])

with right_col:
    st.subheader("Configurações")
    use_lmfit = st.checkbox("Usar lmfit (ajuste multi-peak Voigt)", value=LMFIT_AVAILABLE)
    calibrate_degree = st.selectbox("Grau polinômio calibração (Neon+Poli)", options=[1, 2, 3], index=1)
    silicon_ref_value = st.number_input("Posição de referência do pico do Silício (cm⁻¹)", value=520.7, format="%.2f")

    st.markdown("### Posições de referência")
    ref_mode = st.radio(
        "Modo de referência:",
        ["Usar tabelas padrão (recomendado)", "Informar manualmente"],
        index=0,
    )

    if ref_mode == "Informar manualmente":
        neon_ref_text = st.text_area(
            "Neon ref positions (cm⁻¹) – separado por vírgulas",
            value="",
            height=60,
        )
        poly_ref_text = st.text_area(
            "Poliestireno ref positions (cm⁻¹) – separado por vírgulas",
            value="",
            height=60,
        )
    else:
        st.write("**Neon (padrão)**:", ", ".join([f"{v:.1f}" for v in NEON_REF_CM1]))
        st.write("**Poliestireno (padrão)**:", ", ".join([f"{v:.1f}" for v in POLYSTYRENE_REF_CM1]))
        st.caption("⚠️ Ajuste esses valores no código conforme as linhas usadas no seu setup real.")

def parse_positions(text: str) -> np.ndarray:
    try:
        if not text or text.strip() == "":
            return np.array([])
        parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip() != ""]
        return np.array([float(p) for p in parts], dtype=float)
    except Exception:
        st.error("Erro ao interpretar posições. Use números separados por vírgula.")
        return np.array([])


if ref_mode == "Informar manualmente":
    neon_refs = parse_positions(st.session_state.get("neon_ref_text", "") if "neon_ref_text" in st.session_state else neon_ref_text)
    poly_refs = parse_positions(st.session_state.get("poly_ref_text", "") if "poly_ref_text" in st.session_state else poly_ref_text)
else:
    neon_refs = NEON_REF_CM1.copy()
    poly_refs = POLYSTYRENE_REF_CM1.copy()

st.markdown("---")

tab_main, tab_debug = st.tabs(["🔬 Calibração & Picos", "🛠 Detalhes técnicos"])

with tab_main:
    run_btn = st.button("Executar pipeline completo (A+B+C)")

    if run_btn:
        if not sample_file:
            st.error("Carregue o espectro da amostra.")
        elif not neon_file or not poly_file or not si_file:
            st.error("Carregue também os espectros de Neon, Poliestireno e Silício.")
        elif neon_refs.size == 0 or poly_refs.size == 0:
            st.error("Posições de referência de Neon ou Poliestireno vazias.")
        else:
            progress = st.progress(0, text="Iniciando pipeline...")

            def set_progress(p, text=""):
                progress.progress(int(p), text=text)

            with st.spinner("Processando..."):
                try:
                    res_calib = calibrate_instrument_from_files(
                        neon_file=neon_file,
                        polystyrene_file=poly_file,
                        silicon_file=si_file,
                        sample_file=sample_file,
                        neon_ref_positions=neon_refs,
                        poly_ref_positions=poly_refs,
                        silicon_ref_position=float(silicon_ref_value),
                        poly_degree=int(calibrate_degree),
                        progress_cb=set_progress,
                    )
                except Exception as e:
                    progress.empty()
                    st.error(f"Erro na calibração: {e}")
                    st.exception(e)
                    st.stop()

            progress.empty()
            st.success("Pipeline concluído com sucesso.")

            x_raw = res_calib["x_sample_raw"]
            y_raw = res_calib["y_sample_raw"]
            x_proc = res_calib["x_sample_proc"]
            y_proc = res_calib["y_sample_proc"]
            x_cal = res_calib["x_sample_calibrated"]

            # PLots
            fig, axs = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
            axs[0].plot(x_raw, y_raw, lw=0.6, label="Raw")
            axs[0].plot(x_proc, y_proc, lw=0.9, label="Processado")
            axs[0].set_xlabel("Deslocamento Raman (cm⁻¹, eixo original)")
            axs[0].set_title("Raw vs Processado")
            axs[0].legend()

            axs[1].plot(x_cal, y_proc, lw=0.9)
            axs[1].set_xlabel("Deslocamento Raman (cm⁻¹, calibrado)")
            axs[1].set_title("Processado no eixo calibrado")
            st.pyplot(fig)

            # Picos e fit no eixo calibrado
            peaks = detect_peaks(x_cal, y_proc, height=0.05, distance=5, prominence=0.02)
            peaks = fit_peaks(x_cal, y_proc, peaks, use_lmfit=use_lmfit)
            peaks = map_peaks_to_molecular_groups(peaks)
            diseases = infer_diseases(peaks)

            if peaks:
                df_peaks = pd.DataFrame([
                    {
                        "position_cm-1": p.position_cm1,
                        "intensity": p.intensity,
                        "width": p.width or "",
                        "group": p.group,
                        "fit_params": json.dumps(p.fit_params) if p.fit_params else "",
                    }
                    for p in peaks
                ])
                st.subheader("Picos detectados (eixo calibrado)")
                st.dataframe(df_peaks)
            else:
                st.info("Nenhum pico detectado com os parâmetros atuais.")

            st.subheader("Padrões sugeridos (pesquisa, NÃO diagnóstico)")
            if diseases:
                st.table(pd.DataFrame(diseases))
            else:
                st.write("Nenhum padrão encontrado com as regras atuais.")

            # Export HDF5
            if H5PY_AVAILABLE:
                bytes_h5 = save_to_nexus_bytes(
                    x_cal,
                    y_proc,
                    {"calibration": json.dumps(res_calib["calibration"])},
                )
                st.download_button(
                    "Baixar espectro calibrado (NeXus-like .h5)",
                    data=bytes_h5,
                    file_name="sample_calibrated.h5",
                    mime="application/octet-stream",
                )
            else:
                st.info("Instale 'h5py' para habilitar export HDF5 (pip install h5py).")

with tab_debug:
    st.markdown("### Tabelas padrão usadas")
    st.write("**Neon (cm⁻¹, exemplo):**", NEON_REF_CM1)
    st.write("**Poliestireno (cm⁻¹, exemplo):**", POLYSTYRENE_REF_CM1)

    st.markdown("### Observação importante")
    st.info(
        "As posições de referência acima são exemplos típicos da literatura. "
        "Ajuste-as no código para refletir exatamente o seu padrão (laser, geometria, "
        "arquivo de referência NIST, etc.)."
    )

st.markdown("---")
st.caption("Se quiser, posso separar esse código em módulo backend + app Streamlit, ou integrar com seu banco de dados / Supabase.")
