# raman_processing.py
# -*- coding: utf-8 -*-
"""
Robust Raman processing pipeline:
- read spectra (txt/csv)
- resample to common x axis
- substrate scaling & subtraction
- baseline correction (AsLS)
- smoothing (Savitzky-Golay)
- peak detection (scipy.signal.find_peaks)
- peak fitting (Lorentzian via scipy.optimize.curve_fit)
Returns: (x, y_corrected), peaks_df, fig
"""

from typing import Tuple, Optional, Dict
import io
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# -------------------------
# I/O helpers
# -------------------------
def read_spectrum(file_like) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads a two-column spectrum from a file-like object (csv/txt).
    Tries to detect delimiter and header. Returns (x, y) sorted by x ascending.
    """
    # If file_like is a stream (StringIO) or bytes, ensure we pass a buffer
    if isinstance(file_like, (bytes, bytearray)):
        buf = io.BytesIO(file_like)
        df = pd.read_csv(buf, sep=None, engine="python")
    else:
        try:
            # pandas can sniff delimiter with sep=None and engine='python'
            df = pd.read_csv(file_like, sep=None, engine="python")
        except Exception:
            # fallback: try whitespace delim
            file_like.seek(0)
            df = pd.read_csv(file_like, delim_whitespace=True, header=None)
    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])
    if df.shape[1] < 2:
        # try reading as two-column with headerless whitespace
        file_like.seek(0)
        df = pd.read_csv(file_like, delim_whitespace=True, header=None)
    # take first two numeric columns
    x = df.iloc[:, 0].values.astype(float)
    y = df.iloc[:, 1].values.astype(float)
    # sort by x
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    return x, y

# -------------------------
# AsLS baseline
# -------------------------
def asls_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """
    Asymmetric Least Squares baseline correction (Eilers).
    Returns baseline array.
    """
    # Implementation adapted for performance
    N = y.size
    D = np.diff(np.eye(N), 2)
    D = np.vstack([np.zeros(N), np.zeros(N)]) if N < 3 else D  # safe fallback
    w = np.ones(N)
    for i in range(niter):
        W = np.diag(w)
        # Solve (W + lam * D.T @ D) z = W y
        try:
            z = np.linalg.solve(W + lam * (D.T @ D), w * y)
        except Exception:
            # fallback: simple weighted moving average as baseline
            z = savgol_filter(y, window_length=max(3, (N//20)|1), polyorder=1)
            break
        w = p * (y > z) + (1 - p) * (y < z)
    return z

# -------------------------
# Lorentzian & fit helpers
# -------------------------
def lorentz(x, A, x0, gamma, offset):
    # A: amplitude, x0: center, gamma: half-width at half-maximum, offset: baseline offset
    return A * (gamma**2 / ((x - x0)**2 + gamma**2)) + offset

def fit_lorentzian(xdata, ydata, x0, window=10.0):
    """
    Fit a lorentzian near x0. window is half-width in x-units to select data for fitting.
    Returns params dict or None on failure.
    """
    mask = (xdata >= x0 - window) & (xdata <= x0 + window)
    if mask.sum() < 5:
        return None
    xs = xdata[mask]
    ys = ydata[mask]
    # initial guesses
    A0 = float(max(ys) - min(ys))
    gamma0 = (xs.max() - xs.min()) / 6.0 if xs.ptp() > 0 else 1.0
    offset0 = float(np.median(ys))
    p0 = [A0, x0, gamma0, offset0]
    # bounds to avoid nonsensical values
    try:
        popt, pcov = curve_fit(lorentz, xs, ys, p0=p0,
                               bounds=([0, x0 - window, 1e-6, -np.inf],
                                       [np.inf, x0 + window, window*2, np.inf]),
                               maxfev=4000)
        A, x0f, gamma, offset = popt
        # compute fitted peak amplitude and width (FWHM = 2*gamma)
        return {"fit_amp": float(A), "fit_cen": float(x0f), "fit_width": float(gamma), "fit_fwhm": float(2*gamma), "fit_offset": float(offset)}
    except Exception:
        return None

# -------------------------
# main pipeline
# -------------------------
def process_raman_pipeline(sample_input,
                           substrate_input,
                           resample_points: int = 3000,
                           sg_window: int = 11,
                           sg_poly: int = 3,
                           asls_lambda: float = 1e5,
                           asls_p: float = 0.01,
                           peak_prominence: float = 0.02,
                           fit_profile: str = "lorentz") -> Tuple[Tuple[np.ndarray, np.ndarray], pd.DataFrame, plt.Figure]:
    """
    Main entry point used by the Streamlit app.
    sample_input / substrate_input: file-like objects (StringIO / uploaded file)
    Returns: (x, y_corrected), peaks_df, fig
    """
    # 1) read
    x_s, y_s = read_spectrum(sample_input)
    x_b, y_b = read_spectrum(substrate_input)

    # 2) build common x axis (resample)
    x_min = max(min(x_s), min(x_b))
    x_max = min(max(x_s), max(x_b))
    if x_max <= x_min:
        # can't intersect; fall back to union and warn
        x_min = min(min(x_s), min(x_b))
        x_max = max(max(x_s), max(x_b))
    x_common = np.linspace(x_min, x_max, int(resample_points))

    y_s_rs = np.interp(x_common, x_s, y_s)
    y_b_rs = np.interp(x_common, x_b, y_b)

    # 3) scale substrate to sample (least-squares scalar alpha)
    denom = np.dot(y_b_rs, y_b_rs)
    if denom == 0:
        alpha = 0.0
    else:
        alpha = float(np.dot(y_s_rs, y_b_rs) / denom)
    # constrain alpha to non-negative reasonable range
    if not np.isfinite(alpha) or alpha < 0:
        alpha = 0.0

    y_subtracted = y_s_rs - alpha * y_b_rs

    # 4) baseline correction (AsLS)
    # make sure arrays are finite
    y_subtracted = np.nan_to_num(y_subtracted, nan=0.0, posinf=0.0, neginf=0.0)
    baseline = asls_baseline(y_subtracted, lam=asls_lambda, p=asls_p, niter=12)
    y_basecorr = y_subtracted - baseline

    # 5) smoothing (Savitzky-Golay) - ensure window <= length and odd
    sg_window = int(sg_window)
    if sg_window >= y_basecorr.size:
        sg_window = y_basecorr.size - 1 if (y_basecorr.size - 1) % 2 == 1 else y_basecorr.size - 2
    if sg_window < 3:
        sg_window = 3
    if sg_window % 2 == 0:
        sg_window += 1
    try:
        y_smooth = savgol_filter(y_basecorr, window_length=sg_window, polyorder=int(sg_poly))
    except Exception:
        y_smooth = y_basecorr

    # 6) normalization (optional) — scale intensities to max=1 to help detection
    maxv = np.max(np.abs(y_smooth)) if y_smooth.size else 1.0
    if maxv == 0:
        norm = 1.0
    else:
        norm = maxv
    y_norm = y_smooth / norm

    # 7) peak detection
    # use prominence and require minimal distance (~resample_points/100)
    min_distance = max(3, int(resample_points / 200))
    try:
        peaks_idx, props = find_peaks(y_norm, prominence=peak_prominence, distance=min_distance)
    except Exception:
        peaks_idx = np.array([], dtype=int)
        props = {}

    # Build peaks dataframe with preliminary values
    peaks = []
    for idx in peaks_idx:
        cen = float(x_common[idx])
        inten = float(y_norm[idx])
        prom = float(props["prominences"][np.where(peaks_idx == idx)][0]) if "prominences" in props and idx in peaks_idx else np.nan
        peaks.append({"peak_cm1": cen, "intensity": inten, "prominence": prom, "index": int(idx)})

    peaks_df = pd.DataFrame(peaks)
    # if no peaks, still return empty df
    if peaks_df.empty:
        # create figure showing results
        fig = plt.figure(figsize=(10, 4))
        plt.plot(x_common, y_s_rs, label="Original")
        plt.plot(x_common, alpha * y_b_rs, label=f"Scaled substrate (alpha={alpha:.3f})")
        plt.plot(x_common, y_subtracted, label="Subtracted")
        plt.plot(x_common, baseline, label="Baseline")
        plt.plot(x_common, y_norm, label="Corrected (norm)")
        plt.xlabel("Wavenumber (cm-1)")
        plt.ylabel("Intensity (a.u.)")
        plt.legend()
        plt.tight_layout()
        return (x_common, y_norm), peaks_df, fig

    # 8) Fit peaks with Lorentzian (or other profile)
    fit_results = []
    for _, prow in peaks_df.iterrows():
        x0 = float(prow["peak_cm1"])
        res = None
        if fit_profile == "lorentz":
            res = fit_lorentzian(x_common, y_norm, x0, window=(x_common.max()-x_common.min())/100.0 * 2.0)
        # fallback: no fit
        if res:
            fit_results.append(res)
        else:
            fit_results.append({"fit_amp": np.nan, "fit_cen": x0, "fit_width": np.nan, "fit_fwhm": np.nan, "fit_offset": np.nan})
    fit_df = pd.DataFrame(fit_results)
    peaks_df = pd.concat([peaks_df.reset_index(drop=True), fit_df.reset_index(drop=True)], axis=1)

    # 9) denormalize intensities in peaks_df (put back to original units)
    peaks_df["intensity_raw"] = peaks_df["intensity"] * norm
    if "fit_amp" in peaks_df and not peaks_df["fit_amp"].isnull().all():
        peaks_df["fit_amp_raw"] = peaks_df["fit_amp"] * norm
    else:
        peaks_df["fit_amp_raw"] = peaks_df["fit_amp"]

    # 10) plot final figure
    fig = plt.figure(figsize=(12, 4))
    plt.plot(x_common, y_s_rs, color="0.5", linewidth=1, label="Raw sample")
    plt.plot(x_common, alpha * y_b_rs, color="#7f7f7f", linestyle="--", linewidth=1, label="Scaled substrate")
    plt.plot(x_common, y_subtracted, color="#d62728", linewidth=1.4, label="Subtracted")
    plt.plot(x_common, baseline, color="#999999", linewidth=1, label="Baseline")
    plt.plot(x_common, y_norm, color="#1f77b4", linewidth=1.6, label="Corrected (norm)")
    # plot fitted peaks if available
    for _, row in peaks_df.iterrows():
        cen = float(row.get("fit_cen", row.get("peak_cm1", np.nan)))
        amp = float(row.get("fit_amp", np.nan)) if not np.isnan(row.get("fit_amp", np.nan)) else np.nan
        if np.isfinite(amp):
            xs = np.linspace(cen - 10, cen + 10, 200)
            ys = lorentz(xs, amp, cen, float(row.get("fit_width", 1.0)), float(row.get("fit_offset", 0.0)))
            plt.plot(xs, ys, linestyle="--", linewidth=1)
        plt.scatter([row["peak_cm1"]], [row["intensity"]], marker="x", color="k", zorder=5)
    plt.xlabel("Wavenumber (cm-1)")
    plt.ylabel("Intensity (a.u.)")
    plt.title("Processed Raman spectrum (substrate subtracted + baseline corrected)")
    plt.legend()
    plt.tight_layout()

    return (x_common, y_norm), peaks_df, fig
