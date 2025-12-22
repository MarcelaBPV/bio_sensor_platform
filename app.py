# app.py
# -*- coding: utf-8 -*-

"""
BioRaman — Plataforma integrada
- Processamento Raman
- Detecção de picos e grupos moleculares
- Correlação com padrões (regras)
- Otimizador com Random Forest
- Persistência no Supabase (opcional)
⚠ Uso em pesquisa. NÃO é diagnóstico médico.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

import raman_processing as rp

from ml_otimizador import (
    run_ml_pipeline_from_peaks_table,
    peaks_list_to_dataframe,
    MLConfig,
)

# Supabase (opcional)
try:
    from supabase_repository import (
        upsert_sample,
        insert_raman_spectrum,
        insert_raman_peaks,
        insert_ml_run,
        insert_ml_feature_importance,
    )
    SUPABASE_AVAILABLE = True
except Exception:
    SUPABASE_AVAILABLE = False


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
# SESSION STATE
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

    st.markdown("---")
    save_to_supabase = st.checkbox(
        "Salvar no Supabase",
        value=False,
        disabled=not SUPABASE_AVAILABLE
    )


# =========================================================
# ABAS
# =========================================================
tab1, tab2, tab3 = st.tabs(
    ["Raman", "Questionário / Pacientes", "Otimizador ML"]
)

# =========================================================
# ABA 1 — RAMAN
# =========================================================
with tab1:
    st.header("Processamento Raman")

    uploaded = st.file_uploader(
        "Upload do espectro (.txt, .csv, .xlsx)",
        type=["txt", "csv", "xls", "xlsx"]
    )

    sample_code = st.text_input("ID da amostra", value="SAMPLE_001")
    label = st.text_input("Classe / rótulo (ex.: controle, diabetes, asma)", value="")

    if uploaded and st.button("▶ Processar espectro"):
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

            # --------- SUPABASE ---------
            if save_to_supabase:
                sample_id = upsert_sample(sample_code, label)
                spectrum_id = insert_raman_spectrum(
                    sample_id=sample_id,
                    filename=uploaded.name,
                    preprocess_params=preprocess_kwargs,
                    meta=res["meta"],
                )
                insert_raman_peaks(spectrum_id, res["peaks"])
                st.info("Dados salvos no Supabase.")

        except Exception as e:
            st.error(f"Erro no processamento: {e}")

    # ---------------- VISUALIZAÇÃO ----------------
    if st.session_state.raman_results is not None:
        data = st.session_state.raman_results
        x_raw, y_raw = data["x_raw"], data["y_raw"]
        x_proc, y_proc = data["x_proc"], data["y_proc"]
        peaks = data["peaks"]

        st.subheader("Espectro processado")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x_raw, y_raw, label="Bruto", alpha=0.6)
        ax.plot(x_proc, y_proc, label="Processado", linewidth=1.6)
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Intensidade (u.a.)")
        ax.legend()
        st.pyplot(fig)

        st.download_button(
            "📥 Baixar gráfico (PNG)",
            fig_to_png_bytes(fig),
            "raman_spectrum.png",
            "image/png"
        )

# =========================================================
# ABA 2 — QUESTIONÁRIO
# =========================================================
with tab2:
    st.header("Questionário / Pacientes")

    q_file = st.file_uploader("Upload CSV do questionário", type=["csv"])

    if q_file:
        df = pd.read_csv(q_file)
        st.dataframe(df.head(), use_container_width=True)

        st.download_button(
            "📥 Baixar dados (CSV)",
            df_to_csv_bytes(df),
            "questionario.csv",
            "text/csv"
        )

# =========================================================
# ABA 3 — OTIMIZADOR ML
# =========================================================
with tab3:
    st.header("🧠 Otimizador — Random Forest")
    st.caption("Aprendizado supervisionado exploratório • Não diagnóstico")

    if st.session_state.raman_results is None:
        st.info("Processe ao menos um espectro na Aba Raman.")
    else:
        data = st.session_state.raman_results
        peaks = data["peaks"]

        df_peaks_ml = peaks_list_to_dataframe(
            peaks=peaks,
            sample_id=sample_code,
            label=label if label else None,
        )

        st.subheader("Tabela de picos (ML-ready)")
        st.dataframe(df_peaks_ml, use_container_width=True)

        st.markdown("### Parâmetros do modelo")
        test_size = st.slider("Test size", 0.1, 0.5, 0.3)
        n_estimators = st.slider("Número de árvores", 50, 500, 200, 50)
        max_depth = st.slider("Profundidade máxima", 2, 30, 10)

        if st.button("🚀 Treinar Random Forest"):
            if df_peaks_ml["label"].isna().all():
                st.error("Informe o rótulo (label) da amostra.")
            else:
                config = MLConfig(
                    test_size=test_size,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                )

                out = run_ml_pipeline_from_peaks_table(
                    peaks_df=df_peaks_ml,
                    config=config,
                )

                ml = out["ml_result"]

                st.subheader("Relatório de classificação")
                st.text(ml.report_text)

                st.subheader("Importância dos grupos moleculares")
                st.dataframe(ml.feature_importances_, use_container_width=True)

                if save_to_supabase:
                    ml_run_id = insert_ml_run(
                        model_type="RandomForest",
                        target_label="label",
                        parameters=config.__dict__,
                        metrics={"report": ml.report_text},
                    )
                    insert_ml_feature_importance(
                        ml_run_id,
                        ml.feature_importances_,
                    )
                    st.info("Resultado do ML salvo no Supabase.")

# =========================================================
# RODAPÉ
# =========================================================
st.markdown("---")
st.caption(
    "BioSensor Plattaorm• Marcela Veiga"
    "Rastreabilidade científica • Uso em pesquisa"
)
