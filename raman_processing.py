# raman_processing.py
# -*- coding: utf-8 -*-
"""
Módulo para processamento Raman baseado em:
- remoção de substrato por interpolação e subtração
- suavização Savitzky-Golay
- correção de baseline por Asymmetric Least Squares (AsLS)
- detecção de picos e ajuste Lorentziano / Pseudo-Voigt
Referências/guia: Georgiev et al. (J. Raman Spectrosc., DOI 10.1002/jrs.6789) e ramanchada2.
"""
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit

# ---------------------------
# util: leitura do formato txt
# ---------------------------
def read_raman_txt(path_or_buffer) -> pd.DataFrame:
    """Lê arquivos com duas colunas (#Wave \t #Intensity) e retorna DataFrame.
    path_or_buffer pode ser caminho ou file-like (Streamlit upload)"""
    # detecta se é file-like (BytesIO) ou caminho
    df = pd.read_csv(path_or_buffer, sep=r'\s+', comment='#', header=None, engine='python', names=['wavenumber_cm1', 'intensity_a'])
    # ordena em ordem crescente de wavenumber (opcional: muitos arquivos vem decrescente)
    df = df.sort_values('wavenumber_cm1').reset_index(drop=True)
    return df

# ----------------------------------
# Asymmetric Least Squares baseline
# (implementação simples e estável)
# ----------------------------------
def baseline_asls(y, lam=1e5, p=0.01, niter=10):
    """
    Asymmetric Least Squares baseline correction (Eilers & Boelens style).
    y: 1D array (intensity)
    lam: smoothness parameter (larger = smoother baseline)
    p: asymmetry parameter (0 < p < 1). Typical 0.001-0.1 for Raman.
    niter: número de iterações (10-20)
    Returns baseline array.
    """
    # baseado em implementações amplamente usadas; sem dependências extras
    L = len(y)
    D = np.diff(np.eye(L), 2)
    D = np.vstack([D, np.zeros((2, L))])[:L-2, :]  # ensure shape if needed
    # construct difference matrix via banded ops is heavier; usamos forma direta (ok para <50k pontos)
    w = np.ones(L)
    for i in range(niter):
        W = np.diag(w)
        Z = W + lam * (D.T @ D)
        # resolver Z * z = w * y
        z = np.linalg.solve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

# -----------------------
# Funções de pico: Lorentz
# -----------------------
def lorentz(x, amp, cen, wid):
    return amp * (0.5 * wid)**2 / ((x - cen)**2 + (0.5 * wid)**2)

def pseudo_voigt(x, amp, cen, wid, eta):
    # eta em [0,1] : 0->Lorentz, 1->Gauss (here we combine crude)
    # gaussian part
    sigma = wid / (2 * np.sqrt(2 * np.log(2)))
    gauss = amp * np.exp(-((x - cen)**2) / (2 * sigma**2))
    lor = lorentz(x, amp, cen, wid)
    return eta * gauss + (1 - eta) * lor

# -----------------------------------
# main pipeline: substrate subtraction
# -----------------------------------
def process_raman_pipeline(sample_input, substrate_input=None,
                           resample_points=3000,
                           sg_window=11, sg_poly=3,
                           asls_lambda=1e5, asls_p=0.01,
                           peak_prominence=0.02,
                           fit_profile='lorentz'):
    """
    sample_input / substrate_input: caminho ou file-like ou DataFrame
    Returns: (x, y_corrected), peaks_df, fig
    peaks_df columns: peak_cm1, intensity, fit_amp, fit_cen, fit_width
    """
    # leitura flexível
    if isinstance(sample_input, pd.DataFrame):
        df_sample = sample_input.copy()
    else:
        df_sample = read_raman_txt(sample_input)
    if substrate_input is not None:
        if isinstance(substrate_input, pd.DataFrame):
            df_sub = substrate_input.copy()
        else:
            df_sub = read_raman_txt(substrate_input)
    else:
        df_sub = None

    # cria eixo comum (intervalo sobreposto)
    if df_sub is not None:
        x_min = max(df_sample.wavenumber_cm1.min(), df_sub.wavenumber_cm1.min())
        x_max = min(df_sample.wavenumber_cm1.max(), df_sub.wavenumber_cm1.max())
    else:
        x_min = df_sample.wavenumber_cm1.min()
        x_max = df_sample.wavenumber_cm1.max()

    x_common = np.linspace(x_min, x_max, resample_points)
    y_sample = np.interp(x_common, df_sample.wavenumber_cm1, df_sample.intensity_a)
    y_sub = np.interp(x_common, df_sub.wavenumber_cm1, df_sub.intensity_a) if df_sub is not None else 0.0

    # subtrai substrato
    y_diff = y_sample - y_sub

    # suaviza (Savitzky-Golay) - bom antes de baseline em muitos casos
    # assegura janela ímpar e menor que número de pontos
    if sg_window >= len(x_common):
        sg_window = len(x_common) - 1 if len(x_common) % 2 == 0 else len(x_common)
    if sg_window % 2 == 0:
        sg_window += 1
    y_sg = savgol_filter(y_diff, window_length=sg_window, polyorder=sg_poly, mode='interp')

    # baseline (AsLS)
    baseline = baseline_asls(y_sg, lam=asls_lambda, p=asls_p, niter=10)
    y_corr = y_sg - baseline

    # normaliza entre 0 e 1 pra facilitar detecção (mas não sobrescreve forma)
    y_corr_norm = (y_corr - np.min(y_corr)) / (np.max(y_corr) - np.min(y_corr) + 1e-12)

    # detectar picos (prominence relativo)
    peak_prom = peak_prominence
    peaks, props = find_peaks(y_corr_norm, prominence=peak_prom, height=peak_prom)
    peak_heights = props.get('peak_heights', np.zeros_like(peaks, dtype=float))

    # montar peaks_df
    peaks_list = []
    for i, idx in enumerate(peaks):
        cen = x_common[idx]
        amp = y_corr_norm[idx]
        # tentativa de ajustar cada pico com uma Lorentziana / Pseudo-Voigt local
        # seleciona jan
