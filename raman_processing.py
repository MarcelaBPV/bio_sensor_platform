# app.py
# -*- coding: utf-8 -*-

"""
BioRaman — Plataforma integrada
- Processamento Raman
- Detecção de picos e grupos moleculares
- Correlação com padrões (regras)
- Questionário / pacientes
⚠ Uso em pesquisa. NÃO é diagnóstico médico.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Optional
import io

import raman_processing as rp

# =========================================================
# CONFIGURAÇÃO GERAL
# =========================================================
st.set_page_config(page_title="BioRaman", layout="wide")
st.title("🧬 BioRaman — Plataforma de Processamento Raman")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
})

# =========================================================
# SESSION STATE (ESSENCIAL)
# =========================================================
if "raman_results" not in st.session_state:
    st.session_state.raman_results = None

# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def df_to_csv_bytes(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8")

# =========================================================
# SIDEBAR — PARÂMETROS
# =========================================================
with st.sidebar:
    st.header("Parâmetros Raman")

    use_despike = st.checkbox("Remover spikes", True)
    smooth = st.checkbox("Suavizar (Savitzky–Golay)", True)
    window_length = st.slider("Janela SG", 5, 201, 9, step=2)
    polyorder = st.slider("Ordem SG", 2, 5, 3)

    baseline_method = st.selectbox("Linha de base", ["als", "none"])
    normalize = st.checkbox("Normalizar 0–1", True)

    st.markdown("---")
    st.subheader("Detecção de picos")
    peak_height = st.slider("Altura mínima", 0.0, 1.0, 0.03, 0.01)
    peak_prominence = st.slider("Proeminência", 0.0, 1.0, 0.03, 0.01)
    peak_distance = st.slider("Distância mínima", 1, 500, 5)

# =========================================================
# ABAS
# =========================================================
tab1, tab2 = st.tabs(["Raman", "Questionário / Pacientes"])

# =========================================================
# ABA 1 — RAMAN
# =========================================================
with tab1:
    st.header("Processamento Raman")

    uploaded = st.file_uploader(
        "Upload do espectro (.txt, .csv, .xlsx)",
        type=["txt", "csv", "xls", "xlsx"]
    )

    if uploaded:
        if st.button("▶ Processar espectro"):
            try:
                preprocess_kwargs = dict(
                    despike_method="median" if use_despike else None,
                    smooth=smooth,
                    window_length=window_length,
                    polyorder=polyorder,
                    baseline_method=baseline_method,
                    normalize=normalize,
                )

                res = rp.process_raman_spectrum_with_groups(
                    uploaded,
                    preprocess_kwargs=preprocess_kwargs,
                    peak_height=peak_height,
                    peak_distance=peak_distance,
                    peak_prominence=peak_prominence,
                )

                st.session_state.raman_results = res
                st.success("Processamento concluído.")

            except Exception as e:
                st.error(f"Erro no processamento: {e}")

    # -----------------------------------------------------
    # VISUALIZAÇÃO (FORA DO BOTÃO!)
    # -----------------------------------------------------
    if st.session_state.raman_results is not None:
        data = st.session_state.raman_results

        x_raw, y_raw = data["x_raw"], data["y_raw"]
        x_proc, y_proc = data["x_proc"], data["y_proc"]
        peaks = data["peaks"]
        diseases = data["diseases"]

        # ---------------- GRÁFICO ----------------
        st.subheader("Espectro processado")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x_raw, y_raw, label="Bruto", alpha=0.6)
        ax.plot(x_proc, y_proc, label="Processado", linewidth=1.6)
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Intensidade (u.a.)")
        ax.legend()
        st.pyplot(fig)

        st.download_button(
            "Baixar gráfico (PNG)",
            fig_to_png_bytes(fig),
            "raman_spectrum.png",
            "image/png"
        )

        # ---------------- TABELA DE PICOS ----------------
        st.subheader("Tabela — Picos detectados")

        if peaks:
            df_peaks = pd.DataFrame([
                {
                    "Raman shift (cm⁻¹)": round(p.position_cm1, 2),
                    "Intensidade": round(p.intensity, 5),
                    "Grupo molecular": p.group or "Não classificado",
                }
                for p in peaks
            ])
            st.dataframe(df_peaks, use_container_width=True)

            st.download_button(
                "📥 Baixar picos (CSV)",
                df_to_csv_bytes(df_peaks),
                "peaks.csv",
                "text/csv"
            )
        else:
            st.info("Nenhum pico detectado.")

        # ---------------- AGRUPAMENTO ----------------
        st.subheader("Tabela — Agrupamento molecular")

        if peaks:
            df_groups = (
                df_peaks
                .groupby("Grupo molecular")
                .size()
                .reset_index(name="Número de picos")
            )
            st.dataframe(df_groups, use_container_width=True)
        else:
            st.info("Sem grupos para agrupar.")

        # ---------------- CORRELAÇÃO COM DOENÇAS ----------------
        st.subheader("Correlação com padrões (regras)")
        st.caption("⚠ Interpretação exploratória — não diagnóstico")

        rows = []
        present_groups = set(df_peaks["Grupo molecular"]) if peaks else set()

        for rule in rp.DISEASE_RULES:
            required = set(rule["groups_required"])
            score = len(required & present_groups) / max(len(required), 1) * 100
            rows.append({
                "Condição": rule["name"],
                "Grupos requeridos": ", ".join(required),
                "Correlação (%)": round(score, 1),
                "Descrição": rule["description"],
            })

        df_disease = pd.DataFrame(rows).sort_values("Correlação (%)", ascending=False)
        st.dataframe(df_disease, use_container_width=True)

        st.download_button(
            "Baixar correlação (CSV)",
            df_to_csv_bytes(df_disease),
            "disease_correlation.csv",
            "text/csv"
        )

# =========================================================
# ABA 2 — QUESTIONÁRIO / PACIENTES
# =========================================================
with tab2:
    st.header("Questionário / Pacientes")

    q_file = st.file_uploader("Upload CSV do questionário", type=["csv"])

    if q_file:
        df = pd.read_csv(q_file)
        st.subheader("Pré-visualização")
        st.dataframe(df.head(), use_container_width=True)

        st.download_button(
            "Baixar dados (CSV)",
            df_to_csv_bytes(df),
            "questionario.csv",
            "text/csv"
        )

# =========================================================
# RODAPÉ
# =========================================================
st.markdown("---")
st.caption(
    "BioRaman • Processamento Raman harmonizado • "
    "Uso em pesquisa"
)
