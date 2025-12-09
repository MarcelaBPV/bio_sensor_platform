# app.py
# -*- coding: utf-8 -*-
"""
BioRaman - Plataforma experimental para análise de espectros Raman,
mapeamento de grupos moleculares e correlação com padrões associados a doenças.

⚠ Uso exclusivo em pesquisa. Não utilizar para diagnóstico clínico.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from raman_processing import (
    load_spectrum,
    preprocess_spectrum,
    detect_peaks,
    map_peaks_to_molecular_groups,
    infer_diseases,
)

# ---------------------------------------------------------------------
# Configuração básica da página
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="BioRaman - Mapeamento Molecular e Doenças",
    layout="wide",
)

st.title("🧬 BioRaman – Espectrometria Raman + Grupos Moleculares + Doenças (Pesquisa)")
st.caption(
    "Ferramenta experimental para visualização de espectros Raman, "
    "identificação de grupos moleculares e correlação com padrões associados a doenças. "
    "**Não utilizar para diagnóstico clínico.**"
)

# ---------------------------------------------------------------------
# Sidebar: upload de arquivo e parâmetros de processamento
# ---------------------------------------------------------------------
st.sidebar.header("1. Upload do espectro")
uploaded_file = st.sidebar.file_uploader(
    "Selecione um arquivo de espectro (.csv, .xlsx, .txt)",
    type=["csv", "xls", "xlsx", "txt"],
)

st.sidebar.header("2. Pré-processamento")
smooth = st.sidebar.checkbox("Suavizar (Savitzky-Golay)", value=True)

window_length = st.sidebar.slider(
    "Janela de suavização",
    min_value=5,
    max_value=51,
    step=2,
    value=9,
    help="Tamanho da janela do filtro Savitzky-Golay (precisa ser ímpar).",
)

polyorder = st.sidebar.slider(
    "Ordem do polinômio",
    min_value=2,
    max_value=5,
    value=3,
    help="Ordem do polinômio usado na suavização.",
)

normalize = st.sidebar.checkbox(
    "Normalizar intensidade (0–1)",
    value=True,
)

st.sidebar.header("3. Detecção de picos")
height = st.sidebar.slider(
    "Altura mínima (intensidade normalizada)",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.01,
)

prominence = st.sidebar.slider(
    "Proeminência mínima",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.01,
)

distance = st.sidebar.slider(
    "Distância mínima entre picos (em pontos)",
    min_value=1,
    max_value=50,
    value=5,
)

# ---------------------------------------------------------------------
# Corpo principal
# ---------------------------------------------------------------------
if uploaded_file is None:
    st.info("📂 Faça o upload de um espectro para começar.")
    st.stop()

# 1) Carregamento do espectro
try:
    x, y = load_spectrum(uploaded_file)
except Exception as e:
    st.error(f"Erro ao ler espectro: {e}")
    st.stop()

# 2) Pré-processamento
x_proc, y_proc = preprocess_spectrum(
    x,
    y,
    smooth=smooth,
    window_length=window_length,
    polyorder=polyorder,
    normalize=normalize,
)

# 3) Detecção de picos
peaks = detect_peaks(
    x_proc,
    y_proc,
    height=height,
    distance=distance,
    prominence=prominence,
)

# 4) Mapeamento para grupos moleculares e correlação com doenças
peaks = map_peaks_to_molecular_groups(peaks)
disease_matches = infer_diseases(peaks)

# ---------------------------------------------------------------------
# Layout: gráfico + tabela de picos
# ---------------------------------------------------------------------
col_plot, col_table = st.columns([2, 1])

with col_plot:
    st.subheader("Espectro Raman (pré-processado)")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_proc, y_proc, label="Espectro (pré-processado)")

    # Marca os picos no gráfico
    if len(peaks) > 0:
        peak_positions = [p.position_cm1 for p in peaks]
        peak_intensities = [p.intensity for p in peaks]
        ax.scatter(peak_positions, peak_intensities, marker="x")

    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("Intensidade (u.a.)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    st.pyplot(fig)

with col_table:
    st.subheader("Picos detectados")
    if len(peaks) == 0:
        st.warning("Nenhum pico detectado com os parâmetros atuais.")
    else:
        df_peaks = pd.DataFrame(
            [
                {
                    "posição (cm⁻¹)": round(p.position_cm1, 2),
                    "intensidade": round(p.intensity, 4),
                    "grupo molecular": p.group if p.group else "-",
                }
                for p in peaks
            ]
        )
        st.dataframe(df_peaks, use_container_width=True)

# ---------------------------------------------------------------------
# Tabela de padrões associados a doenças
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("Padrões associados a doenças (pesquisa, não diagnóstico)")

if len(disease_matches) == 0:
    st.info("Nenhum padrão relevante encontrado com as regras atuais.")
else:
    df_dis = pd.DataFrame(
        [
            {
                "padrão / doença": d.name,
                "score": d.score,
                "descrição": d.description,
            }
            for d in disease_matches
        ]
    )
    st.dataframe(df_dis, use_container_width=True)

    st.markdown(
        "> ⚠️ **Aviso importante**: Estes padrões são apenas indicativos para fins de pesquisa e "
        "desenvolvimento. Não substituem exame clínico, nem laudo médico."
    )
