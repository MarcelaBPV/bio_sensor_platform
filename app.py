# app.py
# -*- coding: utf-8 -*-
"""
Streamlit front-end unificado:
- Processamento Raman (1 arquivo)
- Calibração (3 arquivos: Si + sample + blank)
- Aba adicional: Questionário / Pacientes (tabela + 3 gráficos solicitados)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict

import raman_processing as rp  # seu módulo de processamento já criado

st.set_page_config(page_title="BioRaman", layout="wide")
st.title("🧬 BioRaman — Processamento Raman + Pacientes")

# ---- Sidebar: parâmetros comuns ----
with st.sidebar:
    st.header("Parâmetros gerais")
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

# ---- Top-level tabs ----
tab1, tab2, tab3 = st.tabs(["Processar 1 arquivo", "Calibração (3 uploads)", "Questionário / Pacientes"])

# --------------------
# TAB 1: Processar 1 arquivo
# --------------------
with tab1:
    st.header("Processar 1 espectro")
    uploaded = st.file_uploader("Faça upload do espectro (.txt/.csv/.xlsx)", type=["txt","csv","xls","xlsx"])
    if uploaded:
        preprocess_kwargs = {
            "use_despike": use_despike,
            "smooth": smooth,
            "window_length": window_length,
            "polyorder": polyorder,
            "baseline_method": baseline_method,
            "normalize": normalize
        }
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
        else:
            x_raw, y_raw = res["x_raw"], res["y_raw"]
            x_proc, y_proc = res["x_proc"], res["y_proc"]
            peaks, diseases = res["peaks"], res["diseases"]

            col1, col2 = st.columns([2,1])
            with col1:
                st.subheader("Espectro")
                fig, ax = plt.subplots(figsize=(9,4))
                ax.plot(x_raw, y_raw, label="Bruto", alpha=0.6)
                ax.plot(x_proc, y_proc, label="Pré-processado", linewidth=1.5)
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

# --------------------
# TAB 2: Calibração (3 uploads)
# --------------------
with tab2:
    st.header("Calibração com padrão fixo + silício (upload conjunto)")
    st.info("Envie os 3 arquivos: silício (padrão), amostra, blank (porta-amostra)")
    si_file = st.file_uploader("Padrão de silício", type=["txt","csv","xls","xlsx"], key="si")
    sample_file = st.file_uploader("Amostra", type=["txt","csv","xls","xlsx"], key="sample")
    blank_file = st.file_uploader("Blank (porta-amostra)", type=["txt","csv","xls","xlsx"], key="blank")

    st.subheader("Coeficientes do polinômio base (ex.: 0,1 para identidade)")
    base_coeffs_text = st.text_area("Coeficientes separados por vírgula", value="0,1")
    try:
        base_coeffs = [float(s.strip()) for s in base_coeffs_text.split(",") if s.strip() != ""]
    except Exception:
        st.error("Coeficientes inválidos.")
        base_coeffs = None

    if st.button("Rodar calibração"):
        if si_file is None or sample_file is None or blank_file is None:
            st.error("Envie os três arquivos antes de rodar a calibração.")
        elif base_coeffs is None:
            st.error("Coeficientes inválidos.")
        else:
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
            else:
                st.success("Calibração executada.")
                fig, ax = plt.subplots(figsize=(9,4))
                ax.plot(out["x_sample_proc"], out["y_sample_proc"], label="Amostra (pré-processado)")
                ax.plot(out["x_sample_calibrated"], out["y_sample_blank_corrected"], label="Amostra (calibrada + blank subtraído)", linewidth=1.5)
                ax.set_xlabel("Raman shift (cm⁻¹)")
                ax.set_ylabel("Intensidade (u.a.)")
                ax.legend()
                st.pyplot(fig)

                # Detecção de picos na amostra calibrada
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

# --------------------
# TAB 3: Questionário / Pacientes
# --------------------
with tab3:
    st.header("Questionário / Pacientes")
    st.markdown("Carregue um CSV com os dados do questionário. O sistema tenta mapear automaticamente as colunas mais comuns (sexo/gender, fumante/tabagista, doença).")
    q_file = st.file_uploader("Upload do arquivo do questionário (CSV)", type=["csv"], key="questionario")

    st.markdown("Opcional: upload de um CSV de mapeamento de amostras para pacientes (colunas: sample_filename, patient_id).")
    mapping_file = st.file_uploader("Upload do mapeamento amostra→paciente (CSV opcional)", type=["csv"], key="mapping")

    def _find_column(df: pd.DataFrame, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        # try case-insensitive match
        cols_lower = {col.lower(): col for col in df.columns}
        for c in candidates:
            if c.lower() in cols_lower:
                return cols_lower[c.lower()]
        return None

    def _normalize_gender(val):
        if pd.isna(val):
            return None
        v = str(val).strip().lower()
        if v in ("m","male","masculino","masc","homem","h"):
            return "M"
        if v in ("f","female","feminino","fem","mulher","m"):
            return "F"
        return None

    def _normalize_bool(val):
        if pd.isna(val):
            return None
        v = str(val).strip().lower()
        if v in ("1","true","t","yes","y","sim","s"):
            return True
        if v in ("0","false","f","no","n","nao","não","não"):
            return False
        return None

    if q_file is not None:
        try:
            dfq = pd.read_csv(q_file)
        except Exception:
            q_file.seek(0)
            dfq = pd.read_csv(q_file, sep=";")
        st.subheader("Pré-visualização dos dados")
        st.dataframe(dfq.head(), use_container_width=True)

        # Detectar colunas
        gender_col = _find_column(dfq, ["gender","sexo","sex","genero"])
        smoker_col = _find_column(dfq, ["smoker","fumante","tabagista","smokes"])
        disease_col = _find_column(dfq, ["disease","doenca","has_disease","possui_doenca","illness","doenças","tem_doenca"])

        st.markdown("**Colunas detectadas (tentativa automática):**")
        st.text(f"gender: {gender_col}   |   smoker: {smoker_col}   |   disease: {disease_col}")

        # Normalize and build working DataFrame
        df_work = dfq.copy()
        # create patient_id if available or use index
        id_col = _find_column(dfq, ["patient_id","id","codigo","paciente"])
        if id_col is None:
            df_work["patient_id"] = df_work.index.astype(str)
            id_col = "patient_id"
        else:
            df_work["patient_id"] = df_work[id_col].astype(str)

        # Normalize gender, smoker, disease
        df_work["gender_norm"] = df_work[gender_col].apply(_normalize_gender) if gender_col else None
        if smoker_col:
            df_work["smoker_bool"] = df_work[smoker_col].apply(_normalize_bool)
        else:
            df_work["smoker_bool"] = None
        if disease_col:
            df_work["disease_bool"] = df_work[disease_col].apply(_normalize_bool)
        else:
            df_work["disease_bool"] = None

        st.subheader("Tabela processada (normalizada)")
        st.dataframe(df_work[[id_col, "gender_norm", "smoker_bool", "disease_bool"]].head(200), use_container_width=True)

        # Optional mapping file
        mapping_df = None
        if mapping_file is not None:
            try:
                mapping_df = pd.read_csv(mapping_file)
                st.subheader("Mapeamento amostras → pacientes (pré-visualização)")
                st.dataframe(mapping_df.head(), use_container_width=True)
            except Exception:
                st.warning("Não foi possível ler o arquivo de mapeamento (verifique formato).")

        # Now compute the three charts
        st.markdown("---")
        st.subheader("Gráficos solicitados")

        # Prepare counts
        # Filter to rows with known gender
        df_gender = df_work[df_work["gender_norm"].notna()].copy()
        total_count = len(df_gender)
        counts_gender = df_gender["gender_norm"].value_counts().to_dict()

        # Chart 1: percentage of men and women (of total participants with known gender)
        fig1, ax1 = plt.subplots(figsize=(4,3))
        genders = ["M","F"]
        values = [counts_gender.get(g,0) for g in genders]
        if total_count > 0:
            pct = [v/total_count*100 for v in values]
        else:
            pct = [0,0]
        ax1.bar(genders, pct)
        ax1.set_ylabel("Percentual (%) do total")
        ax1.set_title("Porcentagem de Homens e Mulheres (do total)")
        st.pyplot(fig1)

        # Chart 2: percentage of men and women smokers (percent within each gender)
        fig2, ax2 = plt.subplots(figsize=(4,3))
        pct_smokers = []
        for g in genders:
            sub = df_gender[df_gender["gender_norm"]==g]
            n = len(sub)
            if n == 0:
                pct_smokers.append(0.0)
            else:
                n_smoke = sub["smoker_bool"].sum() if sub["smoker_bool"].notna().any() else 0
                # if smoker_bool has Nones, treat them as False in sum above; better is count only known
                known_smoke = sub["smoker_bool"].notna().sum()
                if known_smoke == 0:
                    pct_smokers.append(0.0)
                else:
                    pct_smokers.append((sub["smoker_bool"].sum() / known_smoke) * 100)
        ax2.bar(genders, pct_smokers)
        ax2.set_ylabel("Percentual (%) de fumantes (por gênero)")
        ax2.set_title("Porcentagem de Homens e Mulheres Tabagistas (por gênero)")
        st.pyplot(fig2)

        # Chart 3: percentage of men and women smokers who reported diseases
        fig3, ax3 = plt.subplots(figsize=(4,3))
        pct_smokers_with_disease = []
        for g in genders:
            sub = df_gender[df_gender["gender_norm"]==g]
            # consider only rows where smoker_bool == True
            smokers = sub[sub["smoker_bool"]==True]
            known_smokers = len(smokers)
            if known_smokers == 0:
                pct_smokers_with_disease.append(0.0)
            else:
                # among these smokers, count those with disease_bool == True (only known)
                known_disease = smokers["disease_bool"].notna().sum()
                if known_disease == 0:
                    # if no disease info, give 0 (or could be NaN)
                    pct_smokers_with_disease.append(0.0)
                else:
                    pct_smokers_with_disease.append((smokers["disease_bool"].sum() / known_disease) * 100)
        ax3.bar(genders, pct_smokers_with_disease)
        ax3.set_ylabel("Percentual (%) de fumantes com doença (por gênero)")
        ax3.set_title("Fumantes que relataram ter doenças (por gênero)")
        st.pyplot(fig3)

        st.markdown("---")
        st.subheader("Observações / notas")
        st.markdown("""
        - A detecção automática tenta mapear colunas comuns; verifique as colunas detectadas mostradas acima.  
        - Valores ausentes (NA) são ignorados nas porcentagens específicas (por exemplo, % de fumantes por gênero considera apenas linhas com informação sobre tabagismo).  
        - Se quiser associar amostras às linhas do questionário, suba um CSV de mapeamento com colunas `sample_filename` e `patient_id`.  
        """)

    else:
        st.info("Faça upload do arquivo CSV do questionário para visualizar tabelas e gráficos.")
