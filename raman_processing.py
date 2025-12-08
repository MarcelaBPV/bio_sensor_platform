# File: raman_processing_v2.py
# -*- coding: utf-8 -*-
"""
Pipeline Raman - Versão 2 (padrão internacional)

Assinaturas principais:
    process_raman_pipeline(sample_input, substrate_input, resample_points=3000,
                           sg_window=11, sg_poly=3, asls_lambda=1e5, asls_p=0.01,
                           peak_prominence=0.02, trim_frac=0.02)

    process_raman_from_arrays(x_sample, y_sample, x_substrate=None, y_substrate=None,
                              resample_points=3000, sg_window=11, sg_poly=3,
                              asls_lambda=1e5, asls_p=0.01,
                              peak_prominence=0.02, trim_frac=0.02)

Retorno das duas funções:
    ((x_common, y_norm), peaks_df, fig)

peaks_df contém colunas:
  - peak_cm1: posição do pico detectado (cm^-1)
  - intensity: intensidade normalizada do pico (y_norm no índice do pico)
  - prominence: proeminência do pico (find_peaks)
  - index: índice no vetor x_common
  - fit_amp, fit_cen, fit_width, fit_fwhm, fit_offset: parâmetros do ajuste Lorentz
  - fit_amp_raw: amplitude do ajuste em unidades originais (antes da normalização)
  - fit_height: valor do modelo Lorentz no centro (normalizado)

Notas:
- process_raman_pipeline aceita file-like (BytesIO) com csv/txt com duas colunas numéricas (wavenumber, intensity).
- process_raman_from_arrays trabalha direto com vetores numpy (ou pandas .values)
- Se substrato não for fornecido, considera substrato como zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any

# tentativa segura de importar messagebox (em servidores/headless pode falhar)
try:
    from tkinter import messagebox
except Exception:
    class _DummyMsg:
        def showwarning(self, *args, **kwargs):
            print("warning:", args, kwargs)
        def showerror(self, *args, **kwargs):
            print("error:", args, kwargs)
    messagebox = _DummyMsg()


# ======================================================================
# Helpers: IO, modelos de pico, baseline
# ======================================================================

def read_spectrum(file_like) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lê espectro de arquivo-like (csv/txt). Retorna x, y ordenados.
    Formato esperado: duas colunas numéricas (wavenumber, intensity).
    """
    try:
        # tenta ler como CSV com separador autodetect
        df = pd.read_csv(
            file_like,
            sep=None,
            engine="python",
            comment="#",
            header=None
        )
    except Exception:
        try:
            file_like.seek(0)
        except Exception:
            pass
        df = pd.read_csv(
            file_like,
            delim_whitespace=True,
            header=None
        )

    df = df.select_dtypes(include=[np.number])
    if df.shape[1] < 2:
        raise ValueError("Arquivo deve ter ao menos duas colunas numéricas (x, y).")

    x = np.asarray(df.iloc[:, 0], dtype=float)
    y = np.asarray(df.iloc[:, 1], dtype=float)
    order = np.argsort(x)
    return x[order], y[order]


def lorentz(x, amp, cen, wid, offset):
    """Modelo Lorentziano padrão."""
    return amp * ((0.5 * wid) ** 2 / ((x - cen) ** 2 + (0.5 * wid) ** 2)) + offset


def asls_baseline(y, lam=1e5, p=0.01, niter=10):
    """
    ASLS (Asymmetric Least Squares) para correção de baseline.

    lam: suavidade (quanto maior, mais suave a linha base)
    p: assimetria (típico 0.001–0.05)
    """
    y = np.asarray(y, dtype=float)
    N = len(y)
    if N < 5:
        return np.zeros_like(y)

    diag0 = np.ones(N - 2)
    diag1 = -2.0 * np.ones(N - 2)
    diag2 = np.ones(N - 2)
    D = sparse.diags(
        [diag0, diag1, diag2],
        offsets=[0, 1, 2],
        shape=(N - 2, N),
        format="csc"
    )

    w = np.ones(N)
    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(N, N), format="csc")
        Z = W + lam * (D.T.dot(D))
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def _apply_smoothing(y_data, window, order):
    """
    Aplica Savitzky-Golay e retorna pd.Series com mesmo index se entrada for Series.
    Robusto a parâmetros ruins (devolve dado original se der erro).
    """
    try:
        win = int(window)
        if win % 2 == 0:
            win += 1
        poly = int(order)

        if isinstance(y_data, pd.Series):
            arr = y_data.values
            idx = y_data.index
        else:
            arr = np.asarray(y_data)
            idx = None

        if win < 3 or win <= poly or win >= arr.size:
            return pd.Series(arr, index=idx) if idx is not None else arr

        sm = savgol_filter(arr, window_length=win, polyorder=poly)
        return pd.Series(sm, index=idx) if idx is not None else sm
    except Exception as e:
        print("Erro em _apply_smoothing:", e)
        try:
            messagebox.showwarning(
                "Erro de Suavização",
                f"Não foi possível aplicar SavGol:\n{e}"
            )
        except Exception:
            pass
        return y_data


def fit_lorentzian(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    window: float = 20.0,
    noise: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """
    Ajusta um pico Lorentziano em torno de x0 usando uma janela fixa.
    Se noise for fornecido, usa como sigma (peso) no curve_fit (padrão internacional).
    """
    mask = (x >= x0 - window / 2) & (x <= x0 + window / 2)
    if mask.sum() < 5:
        return None

    xs = x[mask]
    ys = y[mask]

    # chutes iniciais
    amp0 = float(max(np.nanmax(ys) - np.nanmin(ys), 1e-6))
    off0 = float(np.nanmin(ys))
    wid0 = float(max((xs.max() - xs.min()) / 6.0, 1.0))
    p0 = [amp0, x0, wid0, off0]

    # pesos por ruído
    sigma = None
    if noise is not None and noise > 0:
        sigma = np.full_like(ys, noise, dtype=float)

    try:
        popt, _ = curve_fit(
            lorentz,
            xs,
            ys,
            p0=p0,
            sigma=sigma,
            absolute_sigma=bool(sigma is not None),
            bounds=(
                [0, x0 - 10, 1e-6, -np.inf],
                [np.inf, x0 + 10, (xs.ptp()) * 2, np.inf],
            ),
            maxfev=8000,
        )
        amp, cen, wid, off = popt
        return {
            "fit_amp": float(amp),
            "fit_cen": float(cen),
            "fit_width": float(wid),
            "fit_fwhm": float(2 * wid),
            "fit_offset": float(off),
        }
    except Exception:
        # retorna com cen = x0 para manter consistência
        return {
            "fit_amp": np.nan,
            "fit_cen": float(x0),
            "fit_width": np.nan,
            "fit_fwhm": np.nan,
            "fit_offset": np.nan,
        }


# ======================================================================
# Núcleo do pipeline (arrays) - padrão internacional
# ======================================================================

def _preprocess_raman(
    x_s: np.ndarray,
    y_s: np.ndarray,
    x_b: np.ndarray,
    y_b: np.ndarray,
    resample_points: int = 3000,
    sg_window: int = 11,
    sg_poly: int = 3,
    asls_lambda: float = 1e5,
    asls_p: float = 0.01,
    trim_frac: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Pré-processamento com padrão internacional:

    1) Reamostragem em eixo comum (x_common)
    2) Baseline ASLS separado para amostra e substrato
    3) Subtração de substrato (com alpha/beta)
    4) Suavização leve (Savitzky-Golay)
    5) Normalização por área (mais robusta para comparação clínica)

    Retorna:
        x_common, y_norm, baseline_norm, norm_factor
    """

    # ---------------------------
    # Reamostragem em eixo comum
    # ---------------------------
    resample_points = int(max(10, resample_points))

    x_min = max(min(x_s), min(x_b))
    x_max = min(max(x_s), max(x_b))
    if x_max <= x_min:
        x_min = min(min(x_s), min(x_b))
        x_max = max(max(x_s), max(x_b))

    x_common = np.linspace(x_min, x_max, resample_points)
    y_s_rs = np.interp(x_common, x_s, y_s)
    y_b_rs = np.interp(x_common, x_b, y_b)

    # ---------------------------
    # Trim central para regressão alpha (evita bordas ruidosas)
    # ---------------------------
    n = x_common.size
    i0 = int(np.floor(n * trim_frac))
    i1 = int(np.ceil(n * (1.0 - trim_frac)))
    i0 = max(i0, 0)
    i1 = min(i1, n)
    if (i1 - i0) < max(10, n // 20):
        i0 = 0
        i1 = n

    ys_trim = y_s_rs[i0:i1]
    yb_trim = y_b_rs[i0:i1]
    mask = np.isfinite(ys_trim) & np.isfinite(yb_trim)
    ys_f = ys_trim[mask]
    yb_f = yb_trim[mask]

    # ---------------------------
    # Baseline ASLS separado (padrão internacional)
    # ---------------------------
    baseline_sample = asls_baseline(y_s_rs, lam=asls_lambda, p=asls_p, niter=12)
    baseline_bg = asls_baseline(y_b_rs, lam=asls_lambda, p=asls_p, niter=12)

    y_sample_corr = y_s_rs - baseline_sample
    y_bg_corr = y_b_rs - baseline_bg

    # ---------------------------
    # Regressão alpha/beta em dados já baseline-corrigidos
    # ---------------------------
    alpha = 0.0
    beta = 0.0
    if len(yb_f) >= 5:
        # importante: usar as partes corrigidas de baseline
        ys_fit = y_sample_corr[i0:i1][mask]
        yb_fit = y_bg_corr[i0:i1][mask]

        if len(ys_fit) >= 5 and len(yb_fit) >= 5:
            A = np.vstack([yb_fit, np.ones_like(yb_fit)]).T
            sol, *_ = np.linalg.lstsq(A, ys_fit, rcond=None)
            alpha_raw, beta_raw = float(sol[0]), float(sol[1])
            alpha = max(0.0, alpha_raw)
            beta = float(beta_raw)

            # limitar alpha para evitar exageros
            max_alpha = max(
                5.0,
                np.median(np.abs(ys_fit))
                / (np.median(np.abs(yb_fit)) + 1e-12)
                * 5.0,
            )
            alpha = min(alpha, max_alpha)

    # ---------------------------
    # Subtração de substrato após baseline
    # ---------------------------
    y_corr = y_sample_corr - alpha * y_bg_corr - beta

    # ---------------------------
    # Suavização leve (SavGol)
    # ---------------------------
    sg_window = int(sg_window)
    if sg_window % 2 == 0:
        sg_window += 1
    if sg_window >= len(y_corr):
        sg_window = max(3, len(y_corr) - 1)
        if sg_window % 2 == 0:
            sg_window -= 1

    y_corr_series = pd.Series(y_corr, index=x_common)
    y_smooth_series = _apply_smoothing(y_corr_series, sg_window, sg_poly)
    try:
        y_smooth = np.asarray(
            y_smooth_series.values
            if isinstance(y_smooth_series, pd.Series)
            else y_smooth_series,
            dtype=float,
        )
    except Exception:
        y_smooth = np.asarray(y_corr, dtype=float)

    # ---------------------------
    # Normalização por área (mais estável que máximo)
    # ---------------------------
    area = float(np.trapz(np.abs(y_smooth), x_common))
    norm = area if area != 0 else 1.0

    y_norm = y_smooth / norm
    baseline_total = baseline_sample - alpha * baseline_bg - beta
    baseline_norm = baseline_total / norm

    return x_common, y_norm, baseline_norm, norm


def _detect_and_fit_peaks(
    x_common: np.ndarray,
    y_norm: np.ndarray,
    baseline_norm: np.ndarray,
    resample_points: int,
    peak_prominence: float,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Detecção e ajuste de picos com padrão internacional:

    - limiar adaptativo baseado no ruído
    - largura mínima
    - ajuste Lorentziano ponderado pelo ruído
    """

    # ---------------------------
    # Estimativa de ruído: desvio padrão em 10% inicial
    # ---------------------------
    n = len(y_norm)
    n0 = max(20, int(0.1 * n))
    noise_region = y_norm[:n0]
    noise = float(np.std(noise_region)) if n0 > 5 else 1e-3
    if noise == 0:
        noise = 1e-3

    # ---------------------------
    # Detecção de picos com limiar adaptativo
    # ---------------------------
    min_distance = max(3, int(resample_points / 200))

    try:
        peaks_idx, props = find_peaks(
            y_norm,
            height=3 * noise,         # pelo menos 3x ruído
            prominence=peak_prominence,
            distance=min_distance,
            width=2,
        )
    except Exception:
        peaks_idx = np.array([], dtype=int)
        props = {}

    peaks = []
    for i, idx in enumerate(peaks_idx):
        cen = float(x_common[idx])
        inten = float(y_norm[idx])
        prom = float(props.get("prominences", [np.nan] * len(peaks_idx))[i]) \
            if "prominences" in props else np.nan
        peaks.append(
            {
                "peak_cm1": cen,
                "intensity": inten,
                "prominence": prom,
                "index": int(idx),
            }
        )
    peaks_df = pd.DataFrame(peaks)

    # ---------------------------
    # Ajuste Lorentziano por pico (ponderado)
    # ---------------------------
    fit_results = []
    if not peaks_df.empty:
        for _, prow in peaks_df.iterrows():
            x0 = float(prow["peak_cm1"])
            window = (x_common.max() - x_common.min()) / 100.0 * 4.0
            res_fit = fit_lorentzian(
                x_common,
                y_norm,
                x0,
                window=window,
                noise=noise,
            )
            fit_results.append(
                res_fit
                or {
                    "fit_amp": np.nan,
                    "fit_cen": x0,
                    "fit_width": np.nan,
                    "fit_fwhm": np.nan,
                    "fit_offset": np.nan,
                }
            )
    fit_df = pd.DataFrame(fit_results)

    # ---------------------------
    # Sincroniza e concatena
    # ---------------------------
    peaks_df = peaks_df.reset_index(drop=True)
    fit_df = fit_df.reset_index(drop=True)

    if len(peaks_df) != len(fit_df):
        minlen = min(len(peaks_df), len(fit_df))
        peaks_df = peaks_df.iloc[:minlen].reset_index(drop=True)
        fit_df = fit_df.iloc[:minlen].reset_index(drop=True)

    peaks_df = pd.concat([peaks_df, fit_df], axis=1)

    # ---------------------------
    # Calcula fit_height (modelo no centro) e retorna residual/model_total
    # ---------------------------
    model_peaks = np.zeros_like(x_common, dtype=float)

    if "fit_amp" in peaks_df.columns:
        def _calc_fit_height(row):
            try:
                amp = row.get("fit_amp", np.nan)
                cen = row.get("fit_cen", row.get("peak_cm1", np.nan))
                wid = row.get("fit_width", np.nan)
                if not (np.isfinite(amp) and np.isfinite(cen)):
                    return np.nan
                if not np.isfinite(wid) or wid <= 0:
                    wid = max(1.0, (x_common.max() - x_common.min()) / 200.0)
                return float(lorentz(
                    np.array([cen]),
                    amp,
                    cen,
                    wid,
                    row.get("fit_offset", 0.0),
                )[0])
            except Exception:
                return np.nan

        peaks_df["fit_height"] = peaks_df.apply(_calc_fit_height, axis=1)

        for _, row in peaks_df.iterrows():
            try:
                amp = float(row.get("fit_amp", 0.0))
                cen = float(row.get("fit_cen", row.get("peak_cm1", np.nan)))
                wid = row.get("fit_width", np.nan)
                if not np.isfinite(wid) or wid <= 0:
                    wid = max(1.0, (x_common.max() - x_common.min()) / 200.0)
                if np.isfinite(amp) and np.isfinite(cen):
                    model_peaks += lorentz(x_common, amp, cen, wid, 0.0)
            except Exception:
                pass

    model_total = baseline_norm + model_peaks
    residual = y_norm - model_total

    return peaks_df, model_total, residual


def _build_raman_figure(
    x_common: np.ndarray,
    y_norm: np.ndarray,
    baseline_norm: np.ndarray,
    model_total: np.ndarray,
    peaks_df: pd.DataFrame,
) -> plt.Figure:
    """
    Cria figura com:
    - espectro normalizado
    - baseline
    - soma de picos ajustados
    - curva individual de cada pico
    - residual em subplot separado
    """

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[6, 0.2, 1], hspace=0.18)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[2, 0], sharex=ax_main)

    # Espectro normalizado
    ax_main.plot(
        x_common,
        y_norm,
        "-",
        color="0.3",
        linewidth=1.2,
        label="Dados (norm.)",
    )

    # Ajuste global
    ax_main.plot(
        x_common,
        model_total,
        "-",
        color="red",
        linewidth=2,
        label="Ajuste Total",
    )

    # Baseline normalizada
    ax_main.plot(
        x_common,
        baseline_norm,
        "--",
        color="magenta",
        linewidth=2,
        label="Linha Base",
    )

    # Curvas individuais de cada pico
    cmap = plt.cm.get_cmap("tab20")
    for i, row in peaks_df.iterrows():
        try:
            amp = float(row.get("fit_amp", np.nan))
            cen = float(row.get("fit_cen", row.get("peak_cm1", np.nan)))
            wid_val = row.get("fit_width", np.nan)
            if not np.isfinite(amp) or not np.isfinite(cen):
                continue
            if not np.isfinite(wid_val) or wid_val <= 0:
                wid = max(1.0, (x_common.max() - x_common.min()) / 200.0)
            else:
                wid = float(wid_val)

            ys = lorentz(x_common, amp, cen, wid, 0.0)
            ax_main.plot(
                x_common,
                ys,
                linestyle="--",
                linewidth=1.0,
                label=f"Pico {i + 1}",
                color=cmap(i % 20),
            )
        except Exception:
            continue

    ax_main.set_ylabel("Intens. Norm.")
    ax_main.set_title("Análise de Espectro Raman (v2)")
    ax_main.grid(True, linestyle="--", alpha=0.4)
    ax_main.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    # Residual
    ax_res.plot(x_common, residual := (y_norm - model_total), linewidth=1)
    ax_res.axhline(0, color="k", linestyle="--", linewidth=1)
    ax_res.set_ylabel("Residuo")
    ax_res.set_xlabel("Wave (cm⁻¹)")
    ax_res.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    return fig


# ======================================================================
# Função core (arrays) + wrappers públicos
# ======================================================================

def _pipeline_core(
    x_s: np.ndarray,
    y_s: np.ndarray,
    x_b: np.ndarray,
    y_b: np.ndarray,
    resample_points: int = 3000,
    sg_window: int = 11,
    sg_poly: int = 3,
    asls_lambda: float = 1e5,
    asls_p: float = 0.01,
    peak_prominence: float = 0.02,
    trim_frac: float = 0.02,
) -> Tuple[Tuple[np.ndarray, np.ndarray], pd.DataFrame, plt.Figure]:
    """
    Parte central do pipeline (trabalha com arrays):

    1) Pré-processamento completo
    2) Detecção e ajuste de picos
    3) Construção da figura

    Retorna:
        (x_common, y_norm), peaks_df, fig
    """

    x_common, y_norm, baseline_norm, norm_factor = _preprocess_raman(
        x_s=x_s,
        y_s=y_s,
        x_b=x_b,
        y_b=y_b,
        resample_points=resample_points,
        sg_window=sg_window,
        sg_poly=sg_poly,
        asls_lambda=asls_lambda,
        asls_p=asls_p,
        trim_frac=trim_frac,
    )

    peaks_df, model_total, residual = _detect_and_fit_peaks(
        x_common=x_common,
        y_norm=y_norm,
        baseline_norm=baseline_norm,
        resample_points=resample_points,
        peak_prominence=peak_prominence,
    )

    # amplitude "raw" (antes da normalização, aproximada)
    if "fit_amp" in peaks_df.columns:
        peaks_df["fit_amp_raw"] = peaks_df["fit_amp"] * norm_factor

    fig = _build_raman_figure(
        x_common=x_common,
        y_norm=y_norm,
        baseline_norm=baseline_norm,
        model_total=model_total,
        peaks_df=peaks_df,
    )

    return (x_common, y_norm), peaks_df, fig


# ----------------------------------------------------------------------
# Wrapper 1: a partir de arquivos file-like (compatível com app.py)
# ----------------------------------------------------------------------
def process_raman_pipeline(
    sample_input,
    substrate_input,
    resample_points: int = 3000,
    sg_window: int = 11,
    sg_poly: int = 3,
    asls_lambda: float = 1e5,
    asls_p: float = 0.01,
    peak_prominence: float = 0.02,
    trim_frac: float = 0.02,
    fit_profile: str = "lorentz",
) -> Tuple[Tuple[np.ndarray, np.ndarray], pd.DataFrame, plt.Figure]:
    """
    Executa pipeline a partir de arquivos file-like (csv/txt)
    e retorna (x_common, y_norm), peaks_df, fig.

    Esta função é a que o app Streamlit chama.
    """

    # leitura
    x_s, y_s = read_spectrum(sample_input)
    if substrate_input is not None:
        x_b, y_b = read_spectrum(substrate_input)
    else:
        x_b, y_b = x_s, np.zeros_like(y_s)

    return _pipeline_core(
        x_s=x_s,
        y_s=y_s,
        x_b=x_b,
        y_b=y_b,
        resample_points=resample_points,
        sg_window=sg_window,
        sg_poly=sg_poly,
        asls_lambda=asls_lambda,
        asls_p=asls_p,
        peak_prominence=peak_prominence,
        trim_frac=trim_frac,
    )


# ----------------------------------------------------------------------
# Wrapper 2: uso direto com arrays (para notebooks, testes, etc.)
# ----------------------------------------------------------------------
def process_raman_from_arrays(
    x_sample: np.ndarray,
    y_sample: np.ndarray,
    x_substrate: Optional[np.ndarray] = None,
    y_substrate: Optional[np.ndarray] = None,
    resample_points: int = 3000,
    sg_window: int = 11,
    sg_poly: int = 3,
    asls_lambda: float = 1e5,
    asls_p: float = 0.01,
    peak_prominence: float = 0.02,
    trim_frac: float = 0.02,
    fit_profile: str = "lorentz",
) -> Tuple[Tuple[np.ndarray, np.ndarray], pd.DataFrame, plt.Figure]:
    """
    Versão do pipeline para uso direto em código (sem arquivos).
    Recebe arrays numpy x/y da amostra e opcionalmente do substrato.
    """

    x_s = np.asarray(x_sample, dtype=float)
    y_s = np.asarray(y_sample, dtype=float)

    if x_substrate is None or y_substrate is None:
        x_b = x_s
        y_b = np.zeros_like(y_s)
    else:
        x_b = np.asarray(x_substrate, dtype=float)
        y_b = np.asarray(y_substrate, dtype=float)

    return _pipeline_core(
        x_s=x_s,
        y_s=y_s,
        x_b=x_b,
        y_b=y_b,
        resample_points=resample_points,
        sg_window=sg_window,
        sg_poly=sg_poly,
        asls_lambda=asls_lambda,
        asls_p=asls_p,
        peak_prominence=peak_prominence,
        trim_frac=trim_frac,
    )
