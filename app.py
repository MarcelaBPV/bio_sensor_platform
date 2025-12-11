# app.py
# -*- coding: utf-8 -*-
"""
Streamlit front-end com estilo e export:
- Processamento Raman (1 arquivo)
- Calibração (3 arquivos)
- Questionário / Pacientes
- Estilo dos gráficos ajustado (fontes, tamanhos, cores)
- Botões para exportar CSVs e PNGs
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List
import io
import base64

import raman_processing as rp  # must be present in same folder

# ---------------------------
# Matplotlib global style to match your examples
# ---------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# Colors used consistently
COLOR_RAW = "#5DA5E8"        # blue-ish (raw)
COLOR_PREPROC = "#FF8C00"    # orange-ish (preprocessed)
COLOR_NORMALIZED = "#222222" # dark for normalized
COLOR_PEAK = "#1f77b4"       # blue marker for peaks
ANNOT_COLOR = "#d62728"      # red annotation text

st.set_page_config(page_title="BioRaman", layout="wide")
st.title("🧬 BioRaman — Processamento Raman + Pacientes (com export)")

# ---- Sidebar parameters ----
with st.sidebar:
    st.header("Parâmetros gerais")
    use_despike = st.checkbox("Remover spikes (mediana)", value=True)
    smooth = st.checkbox("Suavizar (Savitzky-Golay)", value=True)
    window_length = st.slider("Janela SG (ímpar)", 5, 201, 9, 2)
    polyorder = st.slider("Ordem SG", 2, 5, 3)
    baseline_method = st.selectbox("Linha de base", ["als", "none"], index=0)
    normalize = st.checkbox("Normalizar 0–1 (pré-processamento)", value=True)
    st.markdown("---")
    st.subheader("Detecção de picos (valores para sinal normalizado)")
    peak_height = st.slider("Altura mínima (0-1)", 0.0, 1.0, 0.03, 0.01)
    peak_prominence = st.slider("Proeminência mínima (0-1)", 0.0, 1.0, 0.03, 0.01)
    peak_distance = st.slider("Distância mínima entre picos (pontos)", 1, 500, 5)

# Tabs
tab1, tab2, tab3 = st.tabs(["Processar 1 arquivo", "Calibração (3 uploads)", "Questionário / Pacientes"])

# Utility: download helpers
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    buf.seek(0)
    return buf.getvalue()

def send_download_button(bytes_obj: bytes, label: str, file_name: str, mime: str):
    st.download_button(label=label, data=bytes_obj, file_name=file_name, mime=mime)

# --------------------
# TAB 1: Processar 1 arquivo
# --------------------
with tab1:
    st.header("Processar 1 espectro")
    uploaded = st.file_uploader("Faça upload do espectro (.txt/.csv/.xlsx)", type=["txt","csv","xls","xlsx"])

    if uploaded is None:
        st.info("Envie um arquivo para processar.")
    else:
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
            st.stop()

        x_raw, y_raw = res["x_raw"], res["y_raw"]
        x_proc, y_proc = res["x_proc"], res["y_proc"]
        peaks, diseases = res["peaks"], res["diseases"]

        col1, col2 = st.columns([2, 1])

        # Left: two-panel spectrum plot
        with col1:
            st.subheader("Espectro — bruto e pré-processado (painel separado)")

            fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
            # Top: raw + preprocessed (original scale)
            ax_top.plot(x_raw, y_raw, label="Bruto", color=COLOR_RAW, linewidth=1.2, alpha=0.9)
            ax_top.plot(x_proc, y_proc, label="Pré-processado (escala original)", color=COLOR_PREPROC, linewidth=1.6)
            ax_top.set_ylabel("Intensidade (u.a.)", fontsize=13)
            ax_top.legend(frameon=True, fancybox=True)
            ax_top.tick_params(axis="both", which="major")

            # Bottom: normalized preprocessed (0-1)
            y_norm = y_proc.astype(float).copy()
            ymin, ymax = np.nanmin(y_norm), np.nanmax(y_norm)
            if ymax > ymin + 1e-12:
                y_norm = (y_norm - ymin) / (ymax - ymin)
            else:
                y_norm = np.zeros_like(y_norm)

            ax_bot.plot(x_proc, y_norm, label="Pré-processado (normalizado 0–1)", color=COLOR_NORMALIZED, linewidth=1.3)

            # Detect peaks on normalized signal
            try:
                peaks_norm = rp.detect_peaks(x_proc, y_norm, height=peak_height, distance=peak_distance, prominence=peak_prominence)
            except Exception:
                peaks_norm = []

            # If detect_peaks returns intensities corresponding to provided y, annotate accordingly
            if peaks_norm:
                xs = [p.position_cm1 for p in peaks_norm]
                ys = [float(np.interp(xi, x_proc, y_norm)) for xi in xs]
                ax_bot.scatter(xs, ys, marker="x", color=COLOR_PEAK, s=50, zorder=10, label="Picos (normalizado)")
                peaks_norm = rp.map_peaks_to_molecular_groups(peaks_norm)
                for p in peaks_norm:
                    if p.group:
                        yv = float(np.interp(p.position_cm1, x_proc, y_norm))
                        ax_bot.annotate(p.group, (p.position_cm1, yv),
                                        textcoords="offset points", xytext=(5,5),
                                        fontsize=10, color=ANNOT_COLOR, alpha=0.95)

            ax_bot.set_xlabel("Raman shift (cm⁻¹)", fontsize=13)
            ax_bot.set_ylabel("Int. norm.", fontsize=12)
            ax_bot.legend(frameon=True)
            ax_bot.set_xlim(min(x_proc), max(x_proc))

            st.pyplot(fig)

            # export this figure as PNG
            png_bytes = fig_to_png_bytes(fig)
            send_download_button(png_bytes, "📥 Baixar figura (espectro)", "espectro_preprocessado.png", "image/png")

        # Right: peaks table and export
        with col2:
            st.subheader("Picos detectados")
            if not peaks:
                st.info("Nenhum pico detectado (no sinal pré-processado com os parâmetros atuais).")
            else:
                # prepare dataframe for peaks (use normalized intensities for clarity)
                peaks_norm_table = []
                for p in peaks:
                    # map group from earlier mapping if available
                    peaks_norm_table.append({
                        "posição (cm⁻¹)": round(p.position_cm1, 2),
                        "intensidade": float(np.interp(p.position_cm1, x_proc, y_proc)),
                        "grupo": p.group or "-"
                    })
                df_peaks = pd.DataFrame(peaks_norm_table).sort_values("posição (cm⁻¹)").reset_index(drop=True)
                st.dataframe(df_peaks, use_container_width=True)

                # CSV download
                csv_bytes = df_to_csv_bytes(df_peaks)
                send_download_button(csv_bytes, "📥 Baixar tabela de picos (CSV)", "peaks_table.csv", "text/csv")

        st.markdown("---")
        st.subheader("Padrões / Doenças (regras simples)")
        if not diseases:
            st.info("Nenhum padrão detectado.")
        else:
            dfd = pd.DataFrame([{"padrão": d.name, "score": d.score, "descricao": d.description} for d in diseases])
            st.dataframe(dfd, use_container_width=True)
            # allow download of disease matches
            send_download_button(df_to_csv_bytes(dfd), "📥 Baixar padrões detectados (CSV)", "disease_matches.csv", "text/csv")

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
                st.stop()

            st.success("Calibração executada.")

            # Two-panel plot
            figc, (axc_top, axc_bot) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
            axc_top.plot(out["x_sample_proc"], out["y_sample_proc"], label="Amostra (pré-processado)", color=COLOR_PREPROC, linewidth=1.2)
            axc_top.plot(out["x_sample_calibrated"], out["y_sample_blank_corrected"], label="Amostra (calibrada + blank subtraído)", color=COLOR_NORMALIZED, linewidth=1.4)
            axc_top.set_ylabel("Intensidade (u.a.)")
            axc_top.legend()
            axc_top.grid(alpha=0.3)

            # bottom: normalized blank-corrected
            x_cal = np.asarray(out["x_sample_calibrated"])
            y_cal = np.asarray(out["y_sample_blank_corrected"], dtype=float)
            ymin, ymax = np.min(y_cal), np.max(y_cal)
            if ymax > ymin + 1e-12:
                y_norm_cal = (y_cal - ymin) / (ymax - ymin)
            else:
                y_norm_cal = np.zeros_like(y_cal)
            axc_bot.plot(x_cal, y_norm_cal, label="Amostra (calibrada) — normalizado 0–1", color=COLOR_NORMALIZED, linewidth=1.3)

            try:
                peaks_norm_c = rp.detect_peaks(x_cal, y_norm_cal, height=peak_height, distance=peak_distance, prominence=peak_prominence)
            except Exception:
                peaks_norm_c = []

            if peaks_norm_c:
                xs = [p.position_cm1 for p in peaks_norm_c]
                ys = [float(np.interp(xi, x_cal, y_norm_cal)) for xi in xs]
                axc_bot.scatter(xs, ys, marker="x", color=COLOR_PEAK, s=50, zorder=10, label="Picos (normalizado)")
                peaks_norm_c = rp.map_peaks_to_molecular_groups(peaks_norm_c)
                for p in peaks_norm_c:
                    if p.group:
                        yv = float(np.interp(p.position_cm1, x_cal, y_norm_cal))
                        axc_bot.annotate(p.group, (p.position_cm1, yv), textcoords="offset points", xytext=(5,5), fontsize=10, color=ANNOT_COLOR)

            axc_bot.set_xlabel("Raman shift (cm⁻¹)")
            axc_bot.set_ylabel("Int. norm.")
            axc_bot.legend()
            st.pyplot(figc)

            # Export calibrated figure as PNG
            png_cal = fig_to_png_bytes(figc)
            send_download_button(png_cal, "📥 Baixar figura (calibração)", "calibration_figure.png", "image/png")

            # Peaks table for calibrated sample
            peaks_cal = rp.detect_peaks(out["x_sample_calibrated"], out["y_sample_blank_corrected"], height=peak_height, distance=peak_distance, prominence=peak_prominence)
            peaks_cal = rp.map_peaks_to_molecular_groups(peaks_cal)
            if peaks_cal:
                df_peaks_cal = pd.DataFrame([{"posição (cm⁻¹)": round(p.position_cm1, 2), "intensidade": float(np.interp(p.position_cm1, out["x_sample_calibrated"], out["y_sample_blank_corrected"])), "grupo": p.group or "-"} for p in peaks_cal])
                st.subheader("Picos detectados (amostra calibrada)")
                st.dataframe(df_peaks_cal, use_container_width=True)
                send_download_button(df_to_csv_bytes(df_peaks_cal), "📥 Baixar picos calibrados (CSV)", "peaks_calibrated.csv", "text/csv")
            else:
                st.info("Nenhum pico detectado na amostra calibrada com os parâmetros atuais.")

# --------------------
# TAB 3: Questionário / Pacientes
# --------------------
with tab3:
    st.header("Questionário / Pacientes")
    st.markdown("Carregue um CSV com os dados do questionário. O sistema tenta mapear automaticamente as colunas mais comuns (sexo/gender, fumante/tabagista, doença).")
    q_file = st.file_uploader("Upload do arquivo do questionário (CSV)", type=["csv"], key="questionario")

    st.markdown("Opcional: upload de um CSV de mapeamento de amostras para pacientes (colunas: sample_filename, patient_id).")
    mapping_file = st.file_uploader("Upload do mapeamento amostra→paciente (CSV opcional)", type=["csv"], key="mapping")

    def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        cols = list(df.columns)
        for c in candidates:
            if c in cols:
                return c
        lower_map = {col.lower(): col for col in cols}
        for c in candidates:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        for col in cols:
            for c in candidates:
                if c.lower() in col.lower():
                    return col
        return None

    def _normalize_gender(val: Any) -> Optional[str]:
        if pd.isna(val):
            return None
        v = str(val).strip().lower()
        if v in ("m","male","masculino","masc","homem","h"):
            return "M"
        if v in ("f","female","feminino","fem","mulher","w"):
            return "F"
        return None

    def _normalize_bool(val: Any) -> Optional[bool]:
        if pd.isna(val):
            return None
        v = str(val).strip().lower()
        if v in ("1","true","t","yes","y","sim","s"):
            return True
        if v in ("0","false","f","no","n","nao","não","naõ"):
            return False
        return None

    if q_file is None:
        st.info("Faça upload do arquivo CSV do questionário para visualizar tabelas e gráficos.")
    else:
        try:
            dfq = pd.read_csv(q_file)
        except Exception:
            q_file.seek(0)
            dfq = pd.read_csv(q_file, sep=";")

        st.subheader("Pré-visualização dos dados")
        st.dataframe(dfq.head(), use_container_width=True)

        # detect columns
        gender_col = _find_column(dfq, ["gender","sexo","sex","genero"])
        smoker_col = _find_column(dfq, ["smoker","fumante","tabagista","smokes","fuma"])
        disease_col = _find_column(dfq, ["disease","doenca","has_disease","possui_doenca","illness","doenças","tem_doenca","doente"])
        id_col = _find_column(dfq, ["patient_id","id","codigo","paciente","registro","matricula"])

        st.markdown("**Colunas detectadas (tentativa automática):**")
        st.text(f"gender: {gender_col}   |   smoker: {smoker_col}   |   disease: {disease_col}   |   id: {id_col}")

        # prepare working DF
        df_work = dfq.copy()
        if id_col is None:
            df_work["patient_id"] = df_work.index.astype(str)
            id_col_used = "patient_id"
        else:
            df_work["patient_id"] = df_work[id_col].astype(str)
            id_col_used = id_col

        df_work["gender_norm"] = df_work[gender_col].apply(_normalize_gender) if gender_col else None
        df_work["smoker_bool"] = df_work[smoker_col].apply(_normalize_bool) if smoker_col else None
        df_work["disease_bool"] = df_work[disease_col].apply(_normalize_bool) if disease_col else None

        st.subheader("Tabela processada (normalizada)")
        st.dataframe(df_work[[id_col_used, "gender_norm", "smoker_bool", "disease_bool"]].head(200), use_container_width=True)

        # allow download of processed patient table
        send_download_button(df_to_csv_bytes(df_work[[id_col_used, "gender_norm", "smoker_bool", "disease_bool"]].reset_index(drop=True)), "📥 Baixar tabela de pacientes (CSV)", "patients_processed.csv", "text/csv")

        # preview mapping file
        mapping_df = None
        if mapping_file is not None:
            try:
                mapping_df = pd.read_csv(mapping_file)
                st.subheader("Mapeamento amostras → pacientes (pré-visualização)")
                st.dataframe(mapping_df.head(), use_container_width=True)
            except Exception:
                st.warning("Não foi possível ler o arquivo de mapeamento (verifique formato).")

        # Charts corrected
        st.markdown("---")
        st.subheader("Gráficos solicitados (estilizados)")

        df_gender = df_work[df_work["gender_norm"].notna()].copy()
        total_known_gender = len(df_gender)
        genders = ["M","F"]
        counts_gender = df_gender["gender_norm"].value_counts().to_dict()
        pct_gender = [(counts_gender.get(g, 0) / total_known_gender * 100) if total_known_gender > 0 else 0 for g in genders]

        # Chart 1
        fig1, ax1 = plt.subplots(figsize=(5,3.2))
        colors = [COLOR_PEAK, COLOR_PREPROC]
        ax1.bar(genders, pct_gender, color=colors, edgecolor="#333333")
        ax1.set_ylim(0, 100)
        ax1.set_ylabel("Percentual (%)")
        ax1.set_title("Porcentagem de Homens e Mulheres")
        for i, v in enumerate(pct_gender):
            ax1.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=11)
        st.pyplot(fig1)
        png1 = fig_to_png_bytes(fig1)
        send_download_button(png1, "📥 Baixar figura (Gênero)", "chart_gender.png", "image/png")

        # Chart 2: % smokers within each gender
        pct_smokers_within_gender = []
        for g in genders:
            sub = df_gender[df_gender["gender_norm"] == g]
            known_smoker = sub["smoker_bool"].notna().sum()
            if known_smoker == 0:
                pct_smokers_within_gender.append(0.0)
            else:
                n_smoke = int(sub["smoker_bool"].sum(skipna=True))
                pct_smokers_within_gender.append(n_smoke / known_smoker * 100)
        fig2, ax2 = plt.subplots(figsize=(5,3.2))
        ax2.bar(genders, pct_smokers_within_gender, color=colors, edgecolor="#333333")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("Percentual (%) de fumantes")
        ax2.set_title("Porcentagem de Homens e Mulheres Tabagistas (por gênero)")
        for i, v in enumerate(pct_smokers_within_gender):
            ax2.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=11)
        st.pyplot(fig2)
        png2 = fig_to_png_bytes(fig2)
        send_download_button(png2, "📥 Baixar figura (Fumantes)", "chart_smokers.png", "image/png")

        # Chart 3: % of smokers who reported disease within each gender
        pct_smokers_with_disease = []
        for g in genders:
            sub = df_gender[df_gender["gender_norm"] == g]
            smokers = sub[sub["smoker_bool"] == True]
            if len(smokers) == 0:
                pct_smokers_with_disease.append(0.0)
            else:
                known_disease = smokers["disease_bool"].notna().sum()
                if known_disease == 0:
                    pct_smokers_with_disease.append(0.0)
                else:
                    n_disease = int(smokers["disease_bool"].sum(skipna=True))
                    pct_smokers_with_disease.append(n_disease / known_disease * 100)
        fig3, ax3 = plt.subplots(figsize=(5,3.2))
        ax3.bar(genders, pct_smokers_with_disease, color=colors, edgecolor="#333333")
        ax3.set_ylim(0, 100)
        ax3.set_ylabel("Percentual (%)")
        ax3.set_title("Fumantes que relataram ter doenças (por gênero)")
        for i, v in enumerate(pct_smokers_with_disease):
            ax3.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=11)
        st.pyplot(fig3)
        png3 = fig_to_png_bytes(fig3)
        send_download_button(png3, "📥 Baixar figura (Fumantes + Doença)", "chart_smokers_disease.png", "image/png")

        st.markdown("---")
        st.subheader("Notas sobre os cálculos")
        st.markdown(
            "- Gráfico 1: porcentagem considerando apenas registros com gênero conhecido.\n"
            "- Gráfico 2: por gênero, porcentagem de fumantes considerando apenas registros com informação de tabagismo conhecida.\n"
            "- Gráfico 3: entre fumantes (True), porcentagem que relatou doença considerando apenas fumantes com informação de doença conhecida.\n"
            "- Você pode baixar a tabela processada de pacientes (CSV) acima."
        )

st.sidebar.markdown("---")
st.sidebar.caption("Export disponível: tabelas de picos, pacientes processados e figuras PNG. Ajuste parâmetros para melhorar detecção de picos em sinais normalizados.")
