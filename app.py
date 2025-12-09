# app.py
# -*- coding: utf-8 -*-
"""
BioRaman / BioSensor - Plataforma experimental para análise de espectros Raman,
mapeamento de grupos moleculares e correlação com padrões associados a doenças.

⚠ Uso exclusivo em pesquisa. Não utilizar para diagnóstico clínico.
"""

import uuid

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

st.title("BioSensor – Plataforma Integrada")
st.caption(
    "Ferramenta experimental para integração de dados clínicos, espectros Raman, "
    "identificação de grupos moleculares e modelos de aprendizado de máquina. "
    "**Não utilizar para diagnóstico clínico.**"
)

# =====================================================================
# ABA 1 – CADASTRO (GOOGLE FORMS)
# =====================================================================

def aba_cadastro():
    st.header("Aba 1 — Cadastro de participantes (Google Forms)")

    uploaded = st.file_uploader(
        "Upload do CSV/XLSX exportado do Google Forms",
        type=["csv", "xls", "xlsx"],
        key="forms_uploader",
    )

    if uploaded is None:
        st.info("📂 Faça upload do arquivo de respostas do Google Forms para iniciar o cadastro.")
        return

    # Leitura robusta
    if uploaded.name.lower().endswith(".csv"):
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, sep=";")
    else:
        df = pd.read_excel(uploaded)

    st.subheader("Pré-visualização dos dados do formulário")
    st.dataframe(df, use_container_width=True)

    # Cria ID único se não existir
    if "patient_id" not in df.columns:
        df["patient_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

    st.subheader("Participantes com ID gerado")
    cols_preview = ["patient_id"] + [c for c in df.columns if c != "patient_id"]
    st.dataframe(df[cols_preview], use_container_width=True)

    # Salva em sessão
    st.session_state["patients_df"] = df

    st.success(
        "✅ Participantes carregados e IDs gerados. "
        "Esses IDs poderão ser associados aos espectros na Aba 2."
    )


# =====================================================================
# ABA 2 – ESPECTROMETRIA RAMAN
# =====================================================================

def aba_raman():
    st.header("Aba 2 — Espectrometria Raman e mapeamento molecular")

    patients_df = st.session_state.get("patients_df", None)

    # Seleção de participante (se já houver cadastro)
    if patients_df is not None:
        st.subheader("Associação do espectro a um participante")
        selected_patient = st.selectbox(
            "Selecione o participante (patient_id)",
            options=patients_df["patient_id"],
        )
        st.caption("Este espectro será associado ao patient_id selecionado.")
    else:
        st.warning(
            "Nenhum participante cadastrado ainda (Aba 1). "
            "Você pode continuar, mas os espectros não terão associação a pessoas."
        )
        selected_patient = None

    # ID da amostra (para ML)
    st.subheader("Identificação da amostra")
    sample_id = st.text_input(
        "ID da amostra (ex.: S001, P1_T0). Este ID será usado depois na Aba 3 (ML).",
        value="amostra_1",
    )

    # ---------------- Sidebar: upload e parâmetros ----------------
    st.sidebar.header("1. Upload do espectro")
    uploaded_file = st.sidebar.file_uploader(
        "Selecione um arquivo de espectro (.csv, .xlsx, .txt)",
        type=["csv", "xls", "xlsx", "txt"],
        key="raman_uploader",
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

    # ---------------- Corpo principal ----------------
    if uploaded_file is None:
        st.info("📂 Faça o upload de um espectro para começar a análise Raman.")
        return

    # 1) Carregamento do espectro
    try:
        x, y = load_spectrum(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler espectro: {e}")
        return

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

    # ---------------- Layout: gráfico + tabela de picos ----------------
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
            # DataFrame interno com colunas amigáveis + colunas para ML
            df_peaks = pd.DataFrame(
                [
                    {
                        "sample_id": sample_id,
                        "patient_id": selected_patient,
                        "pos_cm1": round(p.position_cm1, 2),
                        "intensidade": float(p.intensity),
                        "grupo_molecular": p.group if p.group else None,
                    }
                    for p in peaks
                ]
            )

            # Exibição mais amigável
            df_display = df_peaks[["pos_cm1", "intensidade", "grupo_molecular"]].copy()
            df_display.rename(
                columns={
                    "pos_cm1": "posição (cm⁻¹)",
                    "grupo_molecular": "grupo molecular",
                },
                inplace=True,
            )

            st.dataframe(df_display, use_container_width=True)

            # Botão para salvar a tabela de picos para uso na Aba 3 (ML)
            if st.button("💾 Salvar picos desta amostra para ML (Aba 3)"):
                if "peaks_table" in st.session_state and st.session_state["peaks_table"] is not None:
                    st.session_state["peaks_table"] = pd.concat(
                        [st.session_state["peaks_table"], df_peaks],
                        ignore_index=True,
                    )
                else:
                    st.session_state["peaks_table"] = df_peaks

                st.success("Picos desta amostra foram adicionados ao dataset de ML (Aba 3).")

    # ---------------- Padrões associados a doenças (regra simples) ----------------
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


# =====================================================================
# ABA 3 – OTIMIZAÇÃO / ML (Random Forest)
# =====================================================================

def aba_otimizacao():
    st.header("Aba 3 — Otimização (Random Forest)")

    # Import atrasado: não quebra a app se o arquivo não existir
    try:
        from ml_otimizador import (
            run_ml_pipeline_from_peaks_table,
            MLConfig,
        )
    except ModuleNotFoundError:
        st.error(
            "❌ O módulo 'ml_otimizador.py' não foi encontrado.\n\n"
            "Crie o arquivo 'ml_otimizador.py' na mesma pasta do app.py "
            "com o código do otimizador de ML."
        )
        return

    peaks_df = st.session_state.get("peaks_table", None)
    patients_df = st.session_state.get("patients_df", None)

    if peaks_df is None or len(peaks_df) == 0:
        st.info(
            "⚠ Ainda não há picos salvos. "
            "Use a Aba 2 para detectar picos e clicar em "
            "'Salvar picos desta amostra para ML'."
        )
        return

    if patients_df is None:
        st.info(
            "⚠ Ainda não há participantes cadastrados. "
            "Carregue o formulário na Aba 1 para associar labels clínicos."
        )
        return

    st.subheader("Dataset atual de picos (para ML)")
    st.dataframe(peaks_df, use_container_width=True)

    # Escolher coluna do formulário que será usada como label (classe)
    st.subheader("Configuração do rótulo (label) para o modelo")

    candidate_label_cols = [c for c in patients_df.columns if c not in ["patient_id"]]

    if not candidate_label_cols:
        st.error("Não foram encontradas colunas no formulário para usar como label.")
        return

    label_col = st.selectbox(
        "Selecione a coluna do formulário que representa o 'rótulo' (ex.: doença, grupo clínico, etc.):",
        options=candidate_label_cols,
    )

    st.caption(
        f"O modelo irá aprender a partir dos grupos moleculares dos picos "
        f"para prever a coluna **{label_col}**."
    )

    # Junta picos + label vindo do formulário (via patient_id)
    if "patient_id" not in peaks_df.columns:
        st.error(
            "A tabela de picos não possui a coluna 'patient_id'. "
            "Verifique a Aba 2."
        )
        return

    labels_slice = patients_df[["patient_id", label_col]].drop_duplicates(subset=["patient_id"])
    merged = peaks_df.merge(labels_slice, on="patient_id", how="inner")

    if merged.empty:
        st.error(
            "Não foi possível associar picos a labels. "
            "Verifique se os patient_id da Aba 2 correspondem aos da Aba 1."
        )
        return

    st.subheader("Tabela de picos com rótulos associados")
    st.dataframe(merged, use_container_width=True)

    # Botão para rodar modelo
    if st.button("▶ Rodar Random Forest com esses dados"):
        with st.spinner("Treinando modelo..."):
            config = MLConfig(
                n_estimators=200,
                test_size=0.3,
                random_state=42,
            )

            result = run_ml_pipeline_from_peaks_table(
                peaks_df=merged,
                config=config,
                id_col="sample_id",
                label_col=label_col,
                group_col="grupo_molecular",
                intensity_col="intensidade",
            )

        ml_result = result["ml_result"]

        st.subheader("Relatório de classificação (train/test)")
        st.text(ml_result.report_text)

        st.subheader("Importância das features (grupos moleculares)")
        st.dataframe(ml_result.feature_importances_, use_container_width=True)

        st.bar_chart(
            ml_result.feature_importances_.set_index("feature")["importance"]
        )


# =====================================================================
# LAYOUT PRINCIPAL — TABS
# =====================================================================

tab1, tab2, tab3 = st.tabs(
    ["1️⃣ Cadastro (Forms)", "2️⃣ Raman", "3️⃣ Otimização / ML"]
)

with tab1:
    aba_cadastro()

with tab2:
    aba_raman()

with tab3:
    aba_otimizacao()
