# app.py
# -*- coding: utf-8 -*-
"""
Plataforma Bio-Raman com cadastro de pacientes e pipeline harmonizado.
Aba 1: Pacientes & Formulários (estatísticas + gráficos)
Aba 2: Raman & Correlação (pipeline: despike comparator, lmfit multi-peak,
       calibração via polinômio base fixo + ajuste fino com Silício)
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

# ---------------------------------------------------------------------
# CONFIG STREAMLIT
# ---------------------------------------------------------------------
st.set_page_config(page_title="Plataforma Bio-Raman", layout="wide")

# ---------------------------------------------------------------------
# MAPA MOLECULAR E REGRAS (para correlação de picos)
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
# FUNÇÕES ABA 1 (PACIENTES)
# ---------------------------------------------------------------------
def load_patient_table(file) -> pd.DataFrame:
    """Lê CSV / XLS / XLSX com dados de pacientes."""
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    elif name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    return df

def guess_gender_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if any(k in c.lower() for k in ["sexo", "gênero", "genero", "sex", "gender"])]
    return candidates[0] if candidates else None

def guess_smoker_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if any(k in c.lower() for k in ["fuma", "fumante", "smoker"])]
    return candidates[0] if candidates else None

def guess_disease_column(df: pd.DataFrame) -> Optional[str]:
    keys = ["doença", "doenca", "doencas", "comorb", "diagnostico", "diagnóstico", "disease"]
    candidates = [c for c in df.columns if any(k in c.lower() for k in keys)]
    return candidates[0] if candidates else None

def normalize_gender(value) -> str:
    if pd.isna(value):
        return "Não informado"
    v = str(value).strip().lower()
    if v in ["f", "fem", "feminino", "female", "woman", "mulher"]:
        return "Feminino"
    if v in ["m", "masc", "masculino", "male", "man", "homem"]:
        return "Masculino"
    return "Não informado"

def normalize_yesno(value) -> str:
    if pd.isna(value):
        return "Não informado"
    v = str(value).strip().lower()
    if v in ["sim", "s", "yes", "y", "1", "true", "verdadeiro"]:
        return "Sim"
    if v in ["não", "nao", "n", "no", "0", "false", "falso"]:
        return "Não"
    return "Não informado"

def compute_patient_stats(
    df: pd.DataFrame,
    col_gender: Optional[str],
    col_smoker: Optional[str],
    col_disease: Optional[str],
) -> Dict[str, pd.DataFrame]:
    stats: Dict[str, pd.DataFrame] = {}
    if col_gender:
        g_norm = df[col_gender].map(normalize_gender)
        stats["sexo"] = (
            g_norm.value_counts(normalize=True) * 100
        ).round(1).rename("percentual").to_frame()
    if col_smoker:
        s_norm = df[col_smoker].map(normalize_yesno)
        stats["fumante"] = (
            s_norm.value_counts(normalize=True) * 100
        ).round(1).rename("percentual").to_frame()
    if col_disease:
        d_norm = df[col_disease].map(normalize_yesno)
        stats["doenca"] = (
            d_norm.value_counts(normalize=True) * 100
        ).round(1).rename("percentual").to_frame()
    return stats

def compute_association_gender_smoker_disease(
    df: pd.DataFrame,
    col_gender: Optional[str],
    col_smoker: Optional[str],
    col_disease: Optional[str],
) -> Optional[pd.DataFrame]:
    """
    Calcula uma tabela (% por sexo) da presença de doença ('Sim')
    estratificada por fumante (Sim/Não/Não informado).
    """
    if not (col_gender and col_smoker and col_disease):
        return None

    g_norm = df[col_gender].map(normalize_gender)
    s_norm = df[col_smoker].map(normalize_yesno)
    d_norm = df[col_disease].map(normalize_yesno)

    df_norm = pd.DataFrame(
        {"Sexo": g_norm, "Fumante": s_norm, "Doenca": d_norm}
    )

    # Considera apenas quem declarou doença = "Sim"
    df_pos = df_norm[df_norm["Doenca"] == "Sim"].copy()
    if df_pos.empty:
        return None

    # % de pacientes COM doença, por sexo, estratificado por fumante
    tab = pd.crosstab(
        df_pos["Sexo"],
        df_pos["Fumante"],
        normalize="index",
    ) * 100.0
    return tab.round(1)

# ---------------------------------------------------------------------
# GRÁFICOS (BARRAS) USADOS NA ABA 1
# ---------------------------------------------------------------------
def plot_percentage_bar(series: pd.Series, title: str, ylabel: str = "% de pacientes"):
    fig, ax = plt.subplots(figsize=(4, 4))
    labels = [str(i) for i in series.index]
    values = series.values
    ax.bar(labels, values)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel("")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, max(100, values.max() * 1.1))
    for i, v in enumerate(values):
        ax.text(
            i,
            v + max(values) * 0.02,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    return fig

def plot_association_bar(tab: pd.DataFrame, title: str = ""):
    """
    tab: linhas = Sexo, colunas = categorias de Fumante, valores = %
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    index = np.arange(len(tab.index))
    cols = list(tab.columns)
    ncols = len(cols)
    width = 0.8 / max(ncols, 1)

    for i, col in enumerate(cols):
        ax.bar(
            index + i * width,
            tab[col].values,
            width,
            label=str(col),
        )

    ax.set_xticks(index + width * (ncols - 1) / 2 if ncols > 1 else index)
    ax.set_xticklabels(tab.index)
    ax.set_ylabel("% de pacientes com doença")
    ax.set_title(title or "Associação sexo × fumante entre pacientes com doença")
    ax.set_ylim(0, 100)

    for i, sexo in enumerate(tab.index):
        for j, col in enumerate(cols):
            v = tab.loc[sexo, col]
            ax.text(
                i + j * width,
                v + 1,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.legend(title="Fumante")
    return fig

# ---------------------------------------------------------------------
# FUNÇÕES ABA 2 (RAMAN)
# ---------------------------------------------------------------------
def load_spectrum(file_like) -> Tuple[np.ndarray, np.ndarray]:
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

# Despike methods
def despike(y: np.ndarray, method: str = "median", kernel_size: int = 5, z_thresh: float = 6.0) -> np.ndarray:
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

# Baseline
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

    # normalize
    if normalize:
        ymin = float(np.min(y_proc))
        ymax = float(np.max(y_proc))
        if ymax > ymin:
            y_proc = (y_proc - ymin) / (ymax - ymin)
        meta["normalize"] = True

    return x, y_proc, meta

# Detect peaks + map + fit
def detect_peaks(x: np.ndarray, y: np.ndarray, height: float = 0.05, distance: int = 5, prominence: float = 0.02) -> List[Peak]:
    indices, _ = find_peaks(y, height=height, distance=distance, prominence=prominence)
    return [Peak(position_cm1=float(x[i]), intensity=float(y[i])) for i in indices]

def map_peaks_to_molecular_groups(peaks: List[Peak]) -> List[Peak]:
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

def fit_peaks_lmfit_global(x: np.ndarray, y: np.ndarray, peaks: List[Peak], model_type: str = "Voigt") -> List[Peak]:
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
        params[f"{pref}center"].set(
            value=cen, min=cen - 10, max=cen + 10
        )
        params[f"{pref}amplitude"].set(value=amp, min=0)
        if f"{pref}sigma" in params:
            params[f"{pref}sigma"].set(
                value=sigma0, min=0.1, max=50
            )
        if f"{pref}gamma" in params:
            params[f"{pref}gamma"].set(
                value=sigma0, min=0.1, max=50
            )

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

def infer_diseases(peaks: List[Peak]):
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
# CALIBRAÇÃO – POLINÔMIO BASE FIXO + AJUSTE COM SILÍCIO
# ---------------------------------------------------------------------
def apply_base_wavenumber_correction(
    x_obs: np.ndarray,
    base_poly_coeffs: np.ndarray,
) -> np.ndarray:
    """
    Aplica o polinômio de calibração global (padrão fixo) ao eixo observado.

    base_poly_coeffs:
        Coeficientes do polinômio (formato np.polyfit),
        que mapeia x_obs -> deslocamento Raman calibrado.
    """
    base_poly_coeffs = np.asarray(base_poly_coeffs, dtype=float)
    return np.polyval(base_poly_coeffs, x_obs)

def calibrate_with_fixed_pattern_and_silicon(
    silicon_file,
    sample_file,
    base_poly_coeffs: np.ndarray,
    silicon_ref_position: float = 520.7,
    progress_cb=None,
) -> Dict[str, Any]:
    """
    Workflow de calibração simplificado:

    - Assume que já existe um polinômio global de calibração do eixo Raman
      (base_poly_coeffs), obtido anteriormente com padrões (Neon/Poliestireno, etc.).
      Esse polinômio é considerado fixo para o equipamento/campanha.

    - Para cada sessão/medida:
        1) Lê espectro de Silício.
        2) Pré-processa (despike/baseline/etc).
        3) Aplica o polinômio base (padrão fixo).
        4) Localiza o pico de Si (~520.7 cm-1) e calcula o desvio residual.
        5) Usa esse desvio para refinar o eixo (offset de laser).
        6) Lê e pré-processa a amostra.
        7) Aplica correção base + offset de Si ao eixo da amostra.
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
    tick(10, "Carregando Silício...")
    x_si_raw, y_si_raw = load_spectrum(silicon_file)
    x_si, y_si, _ = preprocess_spectrum(x_si_raw, y_si_raw, **preprocess_kwargs)

    tick(25, "Aplicando polinômio base ao Silício...")
    x_si_base = apply_base_wavenumber_correction(x_si, base_poly_coeffs)

    # Pico de Si na região 480–560 cm-1
    mask_si = (x_si_base >= 480) & (x_si_base <= 560)
    if not np.any(mask_si):
        raise RuntimeError("Não há pontos suficientes na janela de Si (480–560 cm-1).")

    idx_max = np.argmax(y_si[mask_si])
    x_si_region = x_si_base[mask_si]
    si_cal_base = float(x_si_region[idx_max])  # posição do pico de Si após correção base

    # Offset residual
    delta = silicon_ref_position - si_cal_base

    def corrector_final(x_arr: np.ndarray) -> np.ndarray:
        """
        Correção final aplicada à amostra:
            x_corr = poly_base(x_obs) + delta_Si
        """
        x_base = apply_base_wavenumber_correction(x_arr, base_poly_coeffs)
        return x_base + delta

    # 2) AMOSTRA
    tick(50, "Carregando amostra...")
    x_s_raw, y_s_raw = load_spectrum(sample_file)
    x_s, y_s, meta_s = preprocess_spectrum(x_s_raw, y_s_raw, **preprocess_kwargs)

    tick(75, "Aplicando calibração à amostra...")
    x_s_cal = corrector_final(x_s)

    tick(100, "Calibração concluída.")
    return {
        "x_sample_raw": x_s_raw,
        "y_sample_raw": y_s_raw,
        "x_sample_proc": x_s,
        "y_sample_proc": y_s,
        "x_sample_calibrated": x_s_cal,
        "meta_sample": meta_s,
        "calibration": {
            "base_poly_coeffs": np.asarray(base_poly_coeffs, dtype=float).tolist(),
            "si_cal_base": si_cal_base,
            "silicon_ref_position": float(silicon_ref_position),
            "laser_zero_delta": delta,
        },
    }

# Export HDF5 (NeXus-like)
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

# ---------------------------------------------------------------------
# INTERFACE – ABAS
# ---------------------------------------------------------------------
st.title("Plataforma Bio-Raman")
tab_pacientes, tab_raman = st.tabs(["🧑‍⚕️ Pacientes & Formulários", "🔬 Raman & Correlação"])

# ABA 1 – PACIENTES
with tab_pacientes:
    st.header("Cadastro de pacientes via planilha")
    patient_file = st.file_uploader(
        "Carregar planilha de pacientes (XLS, XLSX ou CSV)",
        type=["xls", "xlsx", "csv"],
    )
    if patient_file:
        df_pac = load_patient_table(patient_file)
        st.subheader("Pré-visualização da planilha")
        st.dataframe(df_pac.head())

        st.subheader("Mapeamento de colunas")
        cols = list(df_pac.columns)
        default_gender = guess_gender_column(df_pac)
        default_smoker = guess_smoker_column(df_pac)
        default_disease = guess_disease_column(df_pac)

        col_g, col_s, col_d = st.columns(3)
        with col_g:
            col_gender = st.selectbox(
                "Coluna de sexo/gênero",
                options=["(nenhuma)"] + cols,
                index=(cols.index(default_gender) + 1)
                if default_gender in cols
                else 0,
            )
            col_gender = None if col_gender == "(nenhuma)" else col_gender
        with col_s:
            col_smoker = st.selectbox(
                "Coluna de fumante (sim/não)",
                options=["(nenhuma)"] + cols,
                index=(cols.index(default_smoker) + 1)
                if default_smoker in cols
                else 0,
            )
            col_smoker = None if col_smoker == "(nenhuma)" else col_smoker
        with col_d:
            col_disease = st.selectbox(
                "Coluna de 'tem alguma doença?' (sim/não)",
                options=["(nenhuma)"] + cols,
                index=(cols.index(default_disease) + 1)
                if default_disease in cols
                else 0,
            )
            col_disease = None if col_disease == "(nenhuma)" else col_disease

        if st.button("Calcular estatísticas dos pacientes"):
            stats = compute_patient_stats(
                df_pac, col_gender, col_smoker, col_disease
            )
            st.subheader("Resumo estatístico")
            st.markdown(f"**Total de registros:** {len(df_pac)}")

            # Sexo – só barras
            if "sexo" in stats:
                st.markdown("### Distribuição de sexo/gênero (%)")
                st.dataframe(stats["sexo"])
                fig_sexo_bar = plot_percentage_bar(
                    stats["sexo"]["percentual"],
                    "Sexo/gênero – barras",
                )
                st.pyplot(fig_sexo_bar)

            # Fumante – só barras
            if "fumante" in stats:
                st.markdown("### Fumante (%)")
                st.dataframe(stats["fumante"])
                fig_fum_bar = plot_percentage_bar(
                    stats["fumante"]["percentual"],
                    "Fumante – barras",
                )
                st.pyplot(fig_fum_bar)

            # Doença – só barras
            if "doenca" in stats:
                st.markdown("### Alguma doença declarada (%)")
                st.dataframe(stats["doenca"])
                fig_doenc_bar = plot_percentage_bar(
                    stats["doenca"]["percentual"],
                    "Doença declarada – barras",
                )
                st.pyplot(fig_doenc_bar)

            # Associação sexo × fumante entre quem declarou doença
            assoc_tab = compute_association_gender_smoker_disease(
                df_pac, col_gender, col_smoker, col_disease
            )
            st.markdown("### Associação entre sexo, tabagismo e presença de doença")
            if assoc_tab is not None:
                st.caption(
                    "Tabela e gráfico mostram, entre os pacientes COM doença declarada, "
                    "a distribuição percentual por sexo e status de fumante."
                )
                st.dataframe(assoc_tab)
                fig_assoc = plot_association_bar(
                    assoc_tab,
                    "Pacientes com doença declarada – % por sexo × fumante",
                )
                st.pyplot(fig_assoc)
            else:
                st.info(
                    "Não foi possível calcular a associação (faltam colunas mapeadas "
                    "ou não há pacientes com doença = 'Sim')."
                )

            st.caption(
                "Obs.: 'Não informado' inclui valores vazios, nulos ou não reconhecidos."
            )

# ABA 2 – RAMAN
with tab_raman:
    st.header("Pipeline Raman – calibração, picos e correlação")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Arquivos de espectros")
        sample_file = st.file_uploader(
            "Amostra (sangue em papel, etc.)", type=["txt", "csv", "xlsx"]
        )
        si_file = st.file_uploader(
            "Silício (padrão para ajuste fino)", type=["txt", "csv", "xlsx"]
        )

    with col2:
        st.subheader("Configurações do processamento")
        use_lmfit = st.checkbox(
            "Usar lmfit (ajuste multi-peak Voigt)", value=LMFIT_AVAILABLE
        )
        silicon_ref_value = st.number_input(
            "Posição de referência do pico do Silício (cm⁻¹)",
            value=520.7,
            format="%.2f",
        )
        coeffs_str = st.text_input(
            "Coeficientes do polinômio base (np.polyfit, separados por vírgula)",
            help=(
                "Informe os coeficientes do polinômio que converte o eixo bruto em cm⁻¹.\n"
                "Exemplo (grau 2): 1.2e-7, -0.03, 550.0"
            ),
        )

    st.markdown("---")
    run_raman = st.button(
        "Executar pipeline Raman (calibração + picos + correlação)"
    )

    if run_raman:
        if not sample_file:
            st.error("Carregue o espectro da amostra.")
        elif not si_file:
            st.error("Carregue o espectro de Silício.")
        elif not coeffs_str.strip():
            st.error("Informe os coeficientes do polinômio base.")
        else:
            # Parse dos coeficientes
            try:
                base_poly_coeffs = np.fromstring(coeffs_str, sep=",")
                if base_poly_coeffs.size == 0:
                    raise ValueError("Nenhum coeficiente encontrado.")
            except Exception as e:
                st.error(
                    f"Erro ao interpretar os coeficientes do polinômio base: {e}"
                )
                st.stop()

            progress = st.progress(0, text="Iniciando pipeline...")

            def set_progress(p, text=""):
                progress.progress(int(p), text=text)

            with st.spinner("Processando espectros..."):
                try:
                    res = calibrate_with_fixed_pattern_and_silicon(
                        silicon_file=si_file,
                        sample_file=sample_file,
                        base_poly_coeffs=base_poly_coeffs,
                        silicon_ref_position=float(silicon_ref_value),
                        progress_cb=set_progress,
                    )
                except Exception as e:
                    progress.empty()
                    st.error(f"Erro no pipeline Raman: {e}")
                    st.exception(e)
                    st.stop()

            progress.empty()
            st.success("Pipeline Raman concluído.")

            x_raw = res["x_sample_raw"]
            y_raw = res["y_sample_raw"]
            x_proc = res["x_sample_proc"]
            y_proc = res["y_sample_proc"]
            x_cal = res["x_sample_calibrated"]

            # Plots
            fig, axs = plt.subplots(
                1, 2, figsize=(13, 4), constrained_layout=True
            )
            axs[0].plot(x_raw, y_raw, lw=0.6, label="Raw")
            axs[0].plot(x_proc, y_proc, lw=0.9, label="Processado")
            axs[0].set_xlabel("Eixo bruto (unidades do equipamento)")
            axs[0].set_title("Raw vs Processado")
            axs[0].legend()

            axs[1].plot(x_cal, y_proc, lw=0.9)
            axs[1].set_xlabel("Deslocamento Raman (cm⁻¹, calibrado)")
            axs[1].set_title(
                "Processado no eixo calibrado (padrão fixo + Si)"
            )

            st.pyplot(fig)

            # Picos & correlação
            peaks = detect_peaks(
                x_cal, y_proc, height=0.05, distance=5, prominence=0.02
            )
            peaks = fit_peaks(
                x_cal, y_proc, peaks, use_lmfit=use_lmfit
            )
            peaks = map_peaks_to_molecular_groups(peaks)
            diseases = infer_diseases(peaks)

            st.subheader("Picos detectados e ajustados (eixo calibrado)")
            if peaks:
                df_peaks = pd.DataFrame(
                    [
                        {
                            "position_cm-1": p.position_cm1,
                            "intensity": p.intensity,
                            "width": p.width or "",
                            "group": p.group,
                            "fit_params": json.dumps(p.fit_params)
                            if p.fit_params
                            else "",
                        }
                        for p in peaks
                    ]
                )
                st.dataframe(df_peaks)
            else:
                st.info(
                    "Nenhum pico detectado com os parâmetros atuais."
                )

            st.subheader(
                "Correlação com padrões de 'doenças' (modo pesquisa)"
            )
            st.caption(
                "⚠ Uso apenas exploratório / pesquisa. NÃO é diagnóstico médico."
            )
            if diseases:
                st.table(pd.DataFrame(diseases))
            else:
                st.write(
                    "Nenhum padrão identificado com as regras atuais."
                )

            # Export HDF5
            if H5PY_AVAILABLE:
                bytes_h5 = save_to_nexus_bytes(
                    x_cal,
                    y_proc,
                    {"calibration": json.dumps(res["calibration"])},
                )
                st.download_button(
                    "Baixar espectro calibrado (NeXus-like .h5)",
                    data=bytes_h5,
                    file_name="sample_calibrated.h5",
                    mime="application/octet-stream",
                )
            else:
                st.info(
                    "Instale 'h5py' para habilitar export HDF5: pip install h5py"
                )

# Rodapé
st.markdown("---")
st.caption(
    "Aba 1: cadastro e estatísticas de pacientes (barras + associação) • "
    "Aba 2: Raman harmonizado + calibração por padrão fixo + ajuste com Si + correlação com padrões."
)
