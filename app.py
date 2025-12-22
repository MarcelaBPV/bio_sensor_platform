# app.py
# -*- coding: utf-8 -*-

"""
BioRaman — Plataforma Integrada
Processamento Raman + Machine Learning + Persistência em Supabase

⚠ Uso exclusivo em pesquisa. NÃO é diagnóstico médico.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import uuid

import raman_processing as rp
from ml_otimizador import (
    train_random_forest_from_features,
    MLConfig,
)

from supabase_repository import (
    insert_sample,
    insert_spectrum,
    insert_peaks,
    insert_ml_features,
)

# =========================================================
# CONFIGURAÇÃO GERAL
# =========================================================
st.set_page_config(page_title="BioRaman", layout="wide")
st.title("🧬 BioRaman — Plataforma Integrada")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
})

# =========================================================
# SESSION STATE
# =========================================================
if "raman_results" not in st.session_state:
    st.session_state.raman_results = None

if "ml_dataset" not in st.session_state:
    st.session_state.ml_dataset = pd.DataFrame()

if "last_sample_id" not in st.session_state:
    st.session_state.last_sample_id = None

if "last_spectrum_id" not in st.session_state:
    st.session_state.last_spectrum_id = None

# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

# =========================================================
# SIDEBAR — PARÂMETROS
# =========================================================
with st.sidebar:
    st.header("Parâmetros Raman")

    use_substrate = st.checkbox("Subtrair substrato", False)

    fit_model = st.selectbox(
        "Ajuste de picos",
        [None, "gauss", "lorentz", "voigt"],
        index=0,
    )

    st.markdown("---")
    st.subheader("Detecção de picos")
    peak_height = st.slider("Altura mínima", 0.0, 1.0, 0.03, 0.01)
    peak_prominence = st.slider("Proeminência", 0.0, 1.0, 0.03, 0.01)
    peak_distance = st.slider("Distância mínima", 1, 500, 5)

# =========================================================
# ABAS
# =========================================================
tab1, tab2, tab3 = st.tabs(
    ["Raman", "Questionário / Pacientes", "Machine Learning"]
)

# =========================================================
# ABA 1 — RAMAN
# =========================================================
with tab1:
    st.header("Processamento Raman")

    sample_file = st.file_uploader(
        "Upload do espectro da amostra",
        type=["txt", "csv", "xls", "xlsx"],
        key="sample",
    )

    substrate_file = None
    if use_substrate:
        substrate_file = st.file_uploader(
            "Upload do espectro do substrato",
            type=["txt", "csv", "xls", "xlsx"],
            key="substrate",
        )

    if sample_file and st.button("▶ Processar espectro"):
        res = rp.process_raman_spectrum_with_groups(
            sample_file,
            substrate_file_like=substrate_file,
            peak_height=peak_height,
            peak_distance=peak_distance,
            peak_prominence=peak_prominence,
            fit_model=fit_model,
        )
        st.session_state.raman_results = res
        st.success("Processamento concluído.")

    # ---------------- VISUALIZAÇÃO ----------------
    if st.session_state.raman_results:
        data = st.session_state.raman_results

        st.subheader("Espectro processado")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data["x_proc"], data["y_proc"], lw=1.6)
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Intensidade (u.a.)")
        st.pyplot(fig)

        # ---------------- PICOS ----------------
        peaks = data["peaks"]
        if peaks:
            df_peaks = pd.DataFrame(
                [{
                    "Raman shift (cm⁻¹)": round(p.position_cm1, 2),
                    "Intensidade": round(p.intensity, 5),
                    "Grupo molecular": p.group,
                    "FWHM": p.width,
                } for p in peaks]
            )
            st.subheader("Picos detectados")
            st.dataframe(df_peaks, use_container_width=True)

        # ---------------- SALVAR NO SUPABASE ----------------
        st.markdown("---")
        st.subheader("Persistência")

        sample_code = st.text_input(
            "Código da amostra",
            value=f"AMOSTRA_{uuid.uuid4().hex[:6].upper()}",
        )

        sample_type = st.selectbox(
            "Tipo de amostra",
            ["sangue", "controle", "substrato", "outro"],
        )

        if st.button("💾 Salvar espectro no Supabase"):
            sample_id = insert_sample(
                sample_code=sample_code,
                sample_type=sample_type,
                metadata={"origem": "BioRaman"},
            )

            spectrum_id = insert_spectrum(
                sample_id=sample_id,
                spectrum_type="processed",
                wavenumber=data["x_proc"].tolist(),
                intensity=data["y_proc"].tolist(),
                preprocessing_params=data["meta"],
            )

            insert_peaks(spectrum_id, data["peaks"])

            st.session_state.last_sample_id = sample_id
            st.session_state.last_spectrum_id = spectrum_id

            st.success("Espectro e picos salvos no Supabase.")

# =========================================================
# ABA 2 — QUESTIONÁRIO
# =========================================================
with tab2:
    st.header("Questionário / Pacientes")

    q_file = st.file_uploader("Upload CSV do questionário", type=["csv"])
    if q_file:
        df_q = pd.read_csv(q_file)
        st.dataframe(df_q.head(), use_container_width=True)

# =========================================================
# ABA 3 — MACHINE LEARNING
# =========================================================
with tab3:
    st.header("Machine Learning — Random Forest")

    if st.session_state.raman_results is None:
        st.info("Processe um espectro na Aba Raman primeiro.")
    else:
        label = st.text_input(
            "Rótulo da amostra (classe)",
            help="Ex.: controle, diabetes, asma",
        )

        if st.button("➕ Adicionar amostra ao dataset ML"):
            features = st.session_state.raman_results["features"]
            row = {**features, "label": label}

            st.session_state.ml_dataset = pd.concat(
                [st.session_state.ml_dataset, pd.DataFrame([row])],
                ignore_index=True,
            )
            st.success("Amostra adicionada ao dataset ML.")

        if not st.session_state.ml_dataset.empty:
            st.subheader("Dataset ML acumulado")
            st.dataframe(st.session_state.ml_dataset, use_container_width=True)

            if st.button("🚀 Treinar Random Forest"):
                result = train_random_forest_from_features(
                    st.session_state.ml_dataset,
                    label_col="label",
                    config=MLConfig(),
                )

                st.subheader("Desempenho do modelo")
                st.metric("Acurácia", f"{result.accuracy:.2f}")
                st.text(result.report_text)

                st.subheader("Importância das features")
                st.dataframe(result.feature_importances.head(15))

                fig, ax = plt.subplots(figsize=(6, 4))
                result.feature_importances.head(10).plot(
                    kind="barh",
                    x="feature",
                    y="importance",
                    ax=ax,
                )
                ax.invert_yaxis()
                st.pyplot(fig)

                # -------- SALVAR FEATURES NO SUPABASE --------
                if (
                    st.session_state.last_sample_id
                    and st.session_state.last_spectrum_id
                ):
                    if st.button("💾 Salvar features ML no Supabase"):
                        insert_ml_features(
                            sample_id=st.session_state.last_sample_id,
                            spectrum_id=st.session_state.last_spectrum_id,
                            features=features,
                            label=label,
                        )
                        st.success("Features ML salvas no Supabase.")

# =========================================================
# RODAPÉ
# =========================================================
st.markdown("---")
st.caption("BioRaman • Plataforma científica • Marcela Veiga")
