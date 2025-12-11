# app.py
# -*- coding: utf-8 -*-
"""
Streamlit front-end que usa raman_processing.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np

import raman_processing as rp

st.set_page_config(page_title="BioRaman", layout="wide")
st.title("🧬 BioRaman — Preprocess & Peak Mapping")

with st.sidebar:
    st.header("Modo de operação")
    mode = st.radio("Escolha:", ["Processar 1 arquivo", "Calibração (3 arquivos: Si+amostra+blank)"])
    st.markdown("---")
    st.subheader("Parâmetros gerais")
    use_despike = st.checkbox("Remover spikes (mediana)", value=True)
    smooth = st.checkbox("Suavizar (Savitzky-Golay)", value=True)
    window_length = st.slider("Janela SG (ímpar)", 5, 101, 9, 2)
    polyorder = st.slider("Ordem SG", 2, 5, 3)
    baseline_method = st.selectbox("Linha de base", ["als", "none"], index=0)
    normalize = st.checkbox("Normalizar 0–1", value=True)
    st.markdown("---")
    st.subheader("Detecção de picos")
    peak_height = st.slider("Altura mínima", 0.0, 1.0, 0.1, 0.01)
    peak_prominence = st.slider("Proeminência mínima", 0.0, 1.0, 0.05, 0.01)
    peak_distance = st.slider("Distância mínima entre picos (pontos)", 1, 200, 5)

if mode == "Processar 1 arquivo":
    uploaded = st.file_uploader("Faça upload do espectro (.txt/.csv/.xlsx)", type=["txt","csv","xls","xlsx"])
    if uploaded is None:
        st.info("Envie um arquivo para processar.")
        st.stop()

    preprocess_kwargs = {"use_despike": use_despike, "smooth": smooth, "window_length": window_length, "polyorder": polyorder, "baseline_method": baseline_method, "normalize": normalize}
    try:
        res = rp.process_raman_spectrum_with_groups(
            uploaded,
            preprocess_kwargs=preprocess_kwargs,
            peak_height=peak_height,
            peak_distance=peak_distance,
            peak_prominence=peak_prominence,
        )
    except Exception as e:
        st.error(f"Erro ao processar: {e}")
        st.stop()

    x_raw, y_raw = res["x_raw"], res["y_raw"]
    x_proc, y_proc = res["x_proc"], res["y_proc"]
    peaks, diseases = res["peaks"], res["diseases"]

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Espectro")
        fig, ax = plt.subplots(figsize=(9,4))
        ax.plot(x_raw, y_raw, label="Bruto", alpha=0.6)
        ax.plot(x_proc, y_proc, label="Pré-processado", color="black")
        if peaks:
            ax.scatter([p.position_cm1 for p in peaks], [p.intensity for p in peaks], color="red", marker="x")
            for p in peaks:
                if p.group:
                    ax.annotate(p.group, xy=(p.position_cm1, p.intensity), xytext=(5,5), fontsize=8)
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Intensidade (u.a.)")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)
    with col2:
        st.subheader("Picos detectados")
        if not peaks:
            st.info("Nenhum pico detectado.")
        else:
            dfp = pd.DataFrame([{"posição (cm⁻¹)": round(p.position_cm1,2), "intensidade": round(p.intensity,4), "grupo": p.group or "-"} for p in peaks])
            st.dataframe(dfp, use_container_width=True)

    st.markdown("---")
    st.subheader("Padrões / Doenças (regras simples)")
    if not diseases:
        st.info("Nenhum padrão detectado.")
    else:
        dfd = pd.DataFrame([{"padrão": d.name, "score": d.score, "descricao": d.description} for d in diseases])
        st.dataframe(dfd, use_container_width=True)

    # Save results
    if st.button("Salvar resultados (CSV)"):
        out_df = pd.DataFrame({"x_raw": x_raw, "y_raw": y_raw, "x_proc": x_proc, "y_proc": y_proc})
        out_df.to_csv("raman_results.csv", index=False)
        st.success("Salvo como raman_results.csv")

else:  # calibration mode (3 uploads)
    st.info("Envie os 3 arquivos: silício (padrão), amostra, blank (porta-amostra)")
    si_file = st.file_uploader("Padrão de silício", type=["txt","csv","xls","xlsx"], key="si")
    sample_file = st.file_uploader("Amostra", type=["txt","csv","xls","xlsx"], key="sample")
    blank_file = st.file_uploader("Blank (porta-amostra)", type=["txt","csv","xls","xlsx"], key="blank")

    st.subheader("Coeficientes do polinômio base (ex.: [a_n,...,a0] para np.polyval)")
    base_coeffs_text = st.text_area("Coeficientes separados por vírgula", value="0,1")  # default identity
    try:
        base_coeffs = [float(s.strip()) for s in base_coeffs_text.split(",") if s.strip() != ""]
    except Exception:
        st.error("Coeficientes inválidos.")
        st.stop()

    if st.button("Rodar calibração"):
        try:
            out = rp.calibrate_with_fixed_pattern_and_silicon(
                silicon_file=si_file,
                sample_file=sample_file,
                blank_file=blank_file,
                base_poly_coeffs=np.array(base_coeffs, dtype=float),
                silicon_ref_position=520.7,
                preprocess_kwargs={"use_despike": use_despike, "smooth": smooth, "window_length": window_length, "polyorder": polyorder, "baseline_method": baseline_method, "normalize": normalize},
            )
        except Exception as e:
            st.error(f"Erro na calibração: {e}")
            st.stop()

        # Show calibrated sample vs blank-corrected
        fig, ax = plt.subplots(figsize=(9,4))
        ax.plot(out["x_sample_proc"], out["y_sample_proc"], label="Amostra (pré-processado)")
        ax.plot(out["x_sample_calibrated"], out["y_sample_blank_corrected"], label="Amostra (calibrada + blank subtraído)", color="black")
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Intensidade (u.a.)")
        ax.legend()
        st.pyplot(fig)

        # detect peaks on blank-corrected spectrum
        peaks = rp.detect_peaks(out["x_sample_calibrated"], out["y_sample_blank_corrected"], height=peak_height, distance=peak_distance, prominence=peak_prominence)
        peaks = rp.map_peaks_to_molecular_groups(peaks)
        diseases = rp.infer_diseases(peaks)

        st.subheader("Picos detectados (amostra calibrada)")
        if not peaks:
            st.info("Nenhum pico detectado.")
        else:
            dfp = pd.DataFrame([{"posição (cm⁻¹)": round(p.position_cm1,2), "intensidade": round(p.intensity,4), "grupo": p.group or "-"} for p in peaks])
            st.dataframe(dfp, use_container_width=True)

        st.subheader("Padrões / Doenças (regras simples)")
        if not diseases:
            st.info("Nenhum padrão detectado.")
        else:
            dfd = pd.DataFrame([{"padrão": d.name, "score": d.score, "descricao": d.description} for d in diseases])
            st.dataframe(dfd, use_container_width=True)
