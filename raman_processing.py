# -*- coding: utf-8 -*-
"""
raman_processing.py
Versão com _apply_smoothing integrada.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from typing import Tuple

# tentativa segura de importar messagebox (em servidores/headless pode falhar)
try:
    from tkinter import messagebox
except Exception:
    class _DummyMsg:
        def showwarning(self, *args, **kwargs):
            print("warning:", args, kwargs)
    messagebox = _DummyMsg()

# ---- sua função de suavização (copiada/adaptada) ----
def _apply_smoothing(self_or_none, y_data, window, order):
    """Aplica a suavização Savitzky-Golay (baseado no código existente).

    Nota: mantive a assinatura para aceitar um 'self' caso queira integrar como método.
    Se usar como função standalone, passe None no primeiro argumento.
    """
    try:
        # se for chamado como método, o primeiro argumento pode ser 'self'
        # então detectamos e realinhamos parâmetros quando necessário
        if y_data is None and window is None and order is None:
            # caso chamem erradamente, retorna sem mudanças
            return y_data

        win = int(window)
        if win % 2 == 0:
            win += 1
        poly = int(order)

        # Verifica se a janela é válida
        if win < 3:  # Janela mínima para SavGol é 3
            print(f"Aviso: Janela SavGol ({win}) inválida (<3). Suavização não aplicada.")
            return y_data
        if win >= len(y_data):
            # ajusta para o maior ímpar menor que len(y_data)
            if len(y_data) % 2 == 0:
                win = len(y_data) - 1
            else:
                win = len(y_data) - 2
            if win < 3:
                print(f"Aviso: Janela SavGol ajustada ({win}) inválida (<3). Suavização não aplicada.")
                return y_data
        if win <= poly:
            print(f"Aviso: Janela SavGol ({win}) <= Ordem ({poly}). Suavização não aplicada.")
            return y_data

        # espera um pd.Series (para preservar index), mas aceita array-like
        if isinstance(y_data, pd.Series):
            arr = y_data.values
            idx = y_data.index
            sm = savgol_filter(arr, window_length=win, polyorder=poly)
            return pd.Series(sm, index=idx)
        else:
            # se não for Series, aplica diretamente e retorna mesmo tipo (np.array)
            return pd.Series(savgol_filter(np.asarray(y_data), window_length=win, polyorder=poly))
    except Exception as e:
        print(f"Erro ao aplicar suavização: {e}")
        try:
            messagebox.showwarning("Erro de Suavização", f"Não foi possível aplicar a suavização.\nVerifique Janela/Ordem.\n{e}")
        except Exception:
            # já logamos acima — evitar crash em ambiente sem GUI
            pass
        return y_data

# ---- IO helper (flexível) ----
def read_spectrum(file_like):
    """Lê espectro de arquivo-like (csv/txt). Retorna x, y ordenados."""
    try:
        df = pd.read_csv(file_like, sep=None, engine='python', comment='#', header=None)
    except Exception:
        file_like.seek(0)
        df = pd.read_csv(file_like, delim_whitespace=True, header=None)
    df = df.select_dtypes(include=[np.number])
    if df.shape[1] < 2:
        raise ValueError('Arquivo deve ter ao menos duas colunas numéricas (x, y).')
    x = np.asarray(df.iloc[:,0], dtype=float)
    y = np.asarray(df.iloc[:,1], dtype=float)
    order = np.argsort(x)
    return x[order], y[order]

# ---- Lorentzian ----
def lorentz(x, amp, cen, wid, offset):
    return amp * ( (0.5*wid)**2 / ( (x-cen)**2 + (0.5*wid)**2 ) ) + offset

def fit_lorentzian(x, y, x0, window=20.0):
    mask = (x >= x0 - window/2) & (x <= x0 + window/2)
    if mask.sum() < 5:
        return None
    xs = x[mask]; ys = y[mask]
    amp0 = max(np.nanmax(ys)-np.nanmin(ys), 1e-6)
    off0 = np.nanmin(ys)
    wid0 = max((xs.max()-xs.min())/6.0, 1.0)
    p0 = [amp0, x0, wid0, off0]
    try:
        popt, _ = curve_fit(
            lorentz, xs, ys, p0=p0,
            bounds=([0, x0-10, 1e-6, -np.inf],[np.inf, x0+10, (xs.ptp())*2, np.inf]),
            maxfev=5000
        )
        amp, cen, wid, off = popt
        return {"fit_amp": float(amp), "fit_cen": float(cen), "fit_width": float(wid), "fit_fwhm": float(2*wid), "fit_offset": float(off)}
    except Exception:
        return {"fit_amp": np.nan, "fit_cen": x0, "fit_width": np.nan, "fit_fwhm": np.nan, "fit_offset": np.nan}

# ---- ASLS baseline robusta ----
def asls_baseline(y, lam=1e5, p=0.01, niter=10):
    y = np.asarray(y, dtype=float)
    N = len(y)
    if N < 5:
        return np.zeros_like(y)
    diag0 = np.ones(N-2)
    diag1 = -2.0 * np.ones(N-2)
    diag2 = np.ones(N-2)
    D = sparse.diags([diag0, diag1, diag2], offsets=[0,1,2], shape=(N-2, N), format='csc')
    w = np.ones(N)
    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(N, N), format='csc')
        Z = W + lam * (D.T.dot(D))
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

# ---- pipeline principal com plotting estilo "main + residual" ----
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
    fit_profile: str = 'lorentz'
) -> Tuple[Tuple[np.ndarray,np.ndarray], pd.DataFrame, plt.Figure]:
    """Executa pipeline e gera figura com main + residual."""

    # leitura
    x_s, y_s = read_spectrum(sample_input)
    x_b, y_b = read_spectrum(substrate_input)

    # resample robusto
    resample_points = int(max(10, resample_points))
    x_min = max(min(x_s), min(x_b))
    x_max = min(max(x_s), max(x_b))
    if x_max <= x_min:
        x_min = min(min(x_s), min(x_b)); x_max = max(max(x_s), max(x_b))
    x_common = np.linspace(x_min, x_max, resample_points)
    y_s_rs = np.interp(x_common, x_s, y_s)
    y_b_rs = np.interp(x_common, x_b, y_b)

    # trim central para fit alpha
    n = x_common.size
    i0 = int(np.floor(n * trim_frac))
    i1 = int(np.ceil(n * (1.0 - trim_frac)))
    i0 = max(i0, 0); i1 = min(i1, n)
    if (i1 - i0) < max(10, n//20):
        i0 = 0; i1 = n
    ys_trim = y_s_rs[i0:i1]; yb_trim = y_b_rs[i0:i1]
    mask = np.isfinite(ys_trim) & np.isfinite(yb_trim)
    ys_f = ys_trim[mask]; yb_f = yb_trim[mask]

    # regressão com intercepto para alpha, beta
    alpha = 0.0; beta = 0.0
    if len(yb_f) >= 5:
        A = np.vstack([yb_f, np.ones_like(yb_f)]).T
        sol, *_ = np.linalg.lstsq(A, ys_f, rcond=None)
        alpha_raw, beta_raw = float(sol[0]), float(sol[1])
        alpha = max(0.0, alpha_raw); beta = float(beta_raw)
    # cap alpha
    max_alpha = max(5.0, np.median(np.abs(ys_f)) / (np.median(np.abs(yb_f))+1e-12) * 5.0) if len(yb_f)>0 else 5.0
    if alpha > max_alpha:
        alpha = max_alpha

    # aplicar subtracao (inclui beta)
    y_sub = y_s_rs - alpha * y_b_rs - beta

    # baseline ASLS
    baseline = asls_baseline(y_sub, lam=asls_lambda, p=asls_p, niter=12)
    y_corr = y_sub - baseline

    # SG smoothing (AJUSTADO para usar _apply_smoothing)
    sg_window = int(sg_window)
    if sg_window % 2 == 0:
        sg_window += 1
    if sg_window >= len(y_corr):
        sg_window = max(3, len(y_corr)-1)
        if sg_window % 2 == 0:
            sg_window -= 1

    # converte para pd.Series com índice x_common para preservar index no retorno
    y_corr_series = pd.Series(y_corr, index=x_common)
    # chamando sua função: primeiro argumento pode ser None (não usado)
    y_smooth_series = _apply_smoothing(None, y_corr_series, sg_window, sg_poly)
    # garantir numpy array para as etapas seguintes
    try:
        y_smooth = np.asarray(y_smooth_series.values if isinstance(y_smooth_series, pd.Series) else y_smooth_series, dtype=float)
    except Exception:
        # fallback: se algo falhar, usa y_corr bruto
        y_smooth = np.asarray(y_corr, dtype=float)

    # normalizacao (para detecção e visual)
    denom = np.nanmax(np.abs(y_smooth))
    norm = denom if denom != 0 else 1.0
    y_norm = y_smooth / norm
    baseline_norm = baseline / norm
    y_s_rs_norm = y_s_rs / norm

    # detect peaks
    min_distance = max(3, int(resample_points / 200))
    try:
        peaks_idx, props = find_peaks(y_norm, prominence=peak_prominence, distance=min_distance)
    except Exception:
        peaks_idx = np.array([], dtype=int); props = {}

    peaks = []
    for idx in peaks_idx:
        cen = float(x_common[idx])
        inten = float(y_norm[idx])
        prom = float(props['prominences'][np.where(peaks_idx==idx)][0]) if 'prominences' in props else np.nan
        peaks.append({'peak_cm1': cen, 'intensity': inten, 'prominence': prom, 'index': int(idx)})
    peaks_df = pd.DataFrame(peaks)

    # fit peaks
    fit_results = []
    for _, prow in peaks_df.iterrows():
        x0 = float(prow['peak_cm1'])
        res_fit = fit_lorentzian(x_common, y_norm, x0, window=(x_common.max()-x_common.min())/100.0*4.0)
        fit_results.append(res_fit or {'fit_amp': np.nan, 'fit_cen': x0, 'fit_width': np.nan, 'fit_fwhm': np.nan, 'fit_offset': np.nan})
    fit_df = pd.DataFrame(fit_results)
    peaks_df = peaks_df.reset_index(drop=True); fit_df = fit_df.reset_index(drop=True)
    if len(peaks_df) != len(fit_df):
        minlen = min(len(peaks_df), len(fit_df))
        peaks_df = peaks_df.iloc[:minlen].reset_index(drop=True)
        fit_df = fit_df.iloc[:minlen].reset_index(drop=True)
    peaks_df = pd.concat([peaks_df, fit_df], axis=1)
    if 'fit_amp' in peaks_df.columns:
        peaks_df['fit_amp_raw'] = peaks_df['fit_amp'] * norm

    # build model: sum of peak lorentzians + baseline_norm
    model_peaks = np.zeros_like(x_common, dtype=float)
    cmap = plt.cm.get_cmap('tab20')
    for i, row in peaks_df.iterrows():
        try:
            amp = float(row.get('fit_amp', 0.0))
            cen = float(row.get('fit_cen', row.get('peak_cm1', np.nan)))
            wid = float(row.get('fit_width', np.nan)) if not np.isnan(row.get('fit_width', np.nan)) else max(1.0, (x_common.max()-x_common.min())/200.0)
            if np.isfinite(amp):
                model_peaks += lorentz(x_common, amp, cen, wid, 0.0)
        except Exception:
            pass
    model_total = baseline_norm + model_peaks
    residual = y_norm - model_total

    # ---- Plot: main + residual ----
    fig = plt.figure(figsize=(12,8))
    gs = fig.add_gridspec(3, 1, height_ratios=[6, 0.2, 1], hspace=0.18)
    ax_main = fig.add_subplot(gs[0,0])
    ax_res = fig.add_subplot(gs[2,0], sharex=ax_main)

    ax_main.plot(x_common, y_s_rs_norm, '.', color='0.4', markersize=3, label='Dados')
    ax_main.plot(x_common, model_total, '-', color='red', linewidth=2, label='Ajuste Total')
    ax_main.plot(x_common, baseline_norm, '--', color='magenta', linewidth=2, label='Linha Base')

    for i, row in peaks_df.iterrows():
        cen = float(row.get('fit_cen', row.get('peak_cm1', np.nan)))
        amp = float(row.get('fit_amp', np.nan)) if not np.isnan(row.get('fit_amp', np.nan)) else 0.0
        wid = float(row.get('fit_width', np.nan)) if not np.isnan(row.get('fit_width', np.nan)) else 1.0
        if amp != 0.0 and np.isfinite(cen):
            ys = lorentz(x_common, amp, cen, wid, 0.0)
            ax_main.plot(x_common, ys, linestyle='--', linewidth=1.2, label=f'Pico {i+1}', color=cmap(i % 20))

    ax_main.set_ylabel('Intens. Norm.')
    ax_main.set_title('Análise de Espectro Raman')
    ax_main.grid(True, linestyle='--', alpha=0.4)
    ax_main.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    ax_res.plot(x_common, residual, color='tab:blue', linewidth=1)
    ax_res.axhline(0, color='k', linestyle='--', linewidth=1)
    ax_res.set_ylabel('Residuo')
    ax_res.set_xlabel('Wave (cm^-1)')
    ax_res.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    return (x_common, y_norm), peaks_df, fig
