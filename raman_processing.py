# -*- coding: utf-8 -*-
"""
raman_processing.py — versão segura (nov/2025)
Pipeline robusto para espectros Raman:
- leitura e reamostragem
- subtração de substrato (com alpha)
- correção de baseline (ASLS)
- suavização (Savitzky-Golay)
- detecção e ajuste lorentziano de picos
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit

# ===============================
#  Funções auxiliares básicas
# ===============================

def read_spectrum(file_like):
    """Lê espectro .txt ou .csv com duas colunas (x, y)."""
    df = pd.read_csv(file_like, sep=None, engine="python", comment="#", header=None)
    if df.shape[1] < 2:
        raise ValueError("Arquivo deve conter ao menos duas colunas (x, y).")
    x = np.array(df.iloc[:, 0], dtype=float)
    y = np.array(df.iloc[:, 1], dtype=float)
    return x, y


def lorentz(x, amp, cen, wid, offset):
    """Função Lorentziana simples."""
    return amp * (0.5 * wid)**2 / ((x - cen)**2 + (0.5 * wid)**2) + offset


# ===============================
#  Baseline ASLS (robusto)
# ===============================

def asls_baseline(y, lam=1e5, p=0.01, niter=10):
    """
    Asymmetric Least Squares baseline correction (robusto).
    lam  — suavização (lambda)
    p    — peso assimétrico (0–1)
    niter — iterações
    """
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    y = np.asarray(y, dtype=float)
    N = len(y)
    if N < 5:
        # fallback para sinais muito curtos: retorna zero baseline (ou suaviza)
        return np.zeros_like(y)

    # --- construir a matriz de segunda diferença D com shape (N-2, N)
    # cada linha i tem [1, -2, 1] nas colunas [i, i+1, i+2]
    diag0 = np.ones(N - 2)
    diag1 = -2.0 * np.ones(N - 2)
    diag2 = np.ones(N - 2)
    # offsets [0,1,2] nas linhas -> forma (N-2, N)
    D = sparse.diags([diag0, diag1, diag2], offsets=[0, 1, 2], shape=(N - 2, N), format="csc")

    # inicializa pesos
    w = np.ones(N)
    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(N, N), format="csc")
        Z = W + lam * (D.T.dot(D))
        # resolve Z z = W y
        z = spsolve(Z, w * y)
        # atualizar pesos (assimetria)
        w = p * (y > z) + (1 - p) * (y < z)

    return z
# ===============================
#  Ajuste Lorentziano
# ===============================

def fit_lorentzian(x, y, x0, window=10.0):
    """Ajusta um pico individual em torno de x0 com perfil lorentziano."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Selecionar região ao redor do pico
    mask = (x >= x0 - window / 2) & (x <= x0 + window / 2)
    if np.sum(mask) < 5:
        return None

    x_fit = x[mask]
    y_fit = y[mask]
    amp0 = np.max(y_fit) - np.min(y_fit)
    off0 = np.min(y_fit)
    wid0 = window / 4

    try:
        popt, _ = curve_fit(lorentz, x_fit, y_fit, p0=[amp0, x0, wid0, off0], maxfev=5000)
        amp, cen, wid, off = popt
        fwhm = 2 * np.abs(wid)
        return {"fit_amp": amp, "fit_cen": cen, "fit_width": wid, "fit_fwhm": fwhm, "fit_offset": off}
    except Exception:
        return {"fit_amp": np.nan, "fit_cen": x0, "fit_width": np.nan, "fit_fwhm": np.nan, "fit_offset": np.nan}


# ===============================
#  Pipeline completo (compatível com app.py)
# ===============================

def process_raman_pipeline(
    sample_input,
    substrate_input,
    resample_points=3000,
    sg_window=11,
    sg_poly=3,
    asls_lambda=1e5,
    asls_p=0.01,
    peak_prominence=0.02,
    fit_profile="lorentz"
):
    """
    Executa o processamento completo do espectro Raman:
    - leitura e reamostragem
    - subtração de substrato (auto-alpha)
    - baseline (ASLS)
    - suavização (SG)
    - detecção e ajuste de picos
    Retorna ((x, y_norm), peaks_df, figura matplotlib)
    """

    import matplotlib.pyplot as plt

    # --- Leitura
    x_s, y_s = read_spectrum(sample_input)
    x_b, y_b = read_spectrum(substrate_input)

    resample_points = int(resample_points)
    if resample_points < 10:
        raise ValueError("resample_points deve ser >= 10")

    # --- Reamostragem
    x_min = max(min(x_s), min(x_b))
    x_max = min(max(x_s), max(x_b))
    if x_max <= x_min:
        x_min, x_max = min(min(x_s), min(x_b)), max(max(x_s), max(x_b))

    x_common = np.linspace(x_min, x_max, resample_points)
    y_s_rs = np.interp(x_common, x_s, y_s)
    y_b_rs = np.interp(x_common, x_b, y_b)

    # --- Subtração com alpha automático
    denom = np.dot(y_b_rs, y_b_rs)
    alpha = np.dot(y_s_rs, y_b_rs) / denom if denom > 0 else 0
    y_sub = y_s_rs - alpha * y_b_rs

    # --- Baseline ASLS
    baseline = asls_baseline(y_sub, lam=asls_lambda, p=asls_p, niter=12)
    y_corr = y_sub - baseline

    # --- Suavização SG
    sg_window = int(sg_window)
    if sg_window % 2 == 0:
        sg_window += 1
    if sg_window >= len(y_corr):
        sg_window = max(3, len(y_corr) - 1)
        if sg_window % 2 == 0:
            sg_window -= 1
    try:
        y_smooth = savgol_filter(y_corr, window_length=sg_window, polyorder=int(sg_poly))
    except Exception:
        y_smooth = y_corr

    # --- Normalização
    y_norm = y_smooth / np.max(np.abs(y_smooth))

    # --- Detecção de picos
    min_distance = max(3, int(resample_points / 200))
    peaks_idx, props = find_peaks(y_norm, prominence=peak_prominence, distance=min_distance)

    peaks = []
    for idx in peaks_idx:
        cen = float(x_common[idx])
        inten = float(y_norm[idx])
        prom = float(props["prominences"][np.where(peaks_idx == idx)][0]) if "prominences" in props else np.nan
        peaks.append({"peak_cm1": cen, "intensity": inten, "prominence": prom, "index": int(idx)})
    peaks_df = pd.DataFrame(peaks)

    # --- Ajuste Lorentziano
    fit_results = []
    for _, prow in peaks_df.iterrows():
        x0 = float(prow["peak_cm1"])
        res_fit = fit_lorentzian(x_common, y_norm, x0, window=(x_common.max() - x_common.min()) / 200.0 * 10.0)
        fit_results.append(res_fit or {"fit_amp": np.nan, "fit_cen": x0, "fit_width": np.nan, "fit_fwhm": np.nan, "fit_offset": np.nan})

    fit_df = pd.DataFrame(fit_results)
    # --- Concatenação segura
    peaks_df = peaks_df.reset_index(drop=True)
    fit_df = fit_df.reset_index(drop=True)
    if len(peaks_df) != len(fit_df):
        minlen = min(len(peaks_df), len(fit_df))
        peaks_df = peaks_df.iloc[:minlen].reset_index(drop=True)
        fit_df = fit_df.iloc[:minlen].reset_index(drop=True)
    peaks_df = pd.concat([peaks_df, fit_df], axis=1)

    # --- Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(x_common, y_s_rs, label="Sample", color="#333")
    axes[0].plot(x_common, alpha * y_b_rs, "--", label=f"Substrate (α={alpha:.3f})", color="#999")
    axes[0].legend()
    axes[0].set_title("Antes: Amostra e Substrato")
    axes[0].set_xlabel("Wavenumber (cm⁻¹)")

    axes[1].plot(x_common, y_norm, label="Corrigido (norm.)", color="#1f77b4")
    axes[1].plot(x_common, baseline / np.max(np.abs(y_corr)), "--", label="Baseline (esc.)", color="#ff7f0e")
    axes[1].set_title("Depois: Baseline + Correção")
    axes[1].set_xlabel("Wavenumber (cm⁻¹)")
    axes[1].legend()

    plt.tight_layout()

    return (x_common, y_norm), peaks_df, fig
