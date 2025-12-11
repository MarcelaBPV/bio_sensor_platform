# app.py
# -*- coding: utf-8 -*-
"""
Plataforma Bio-Raman com cadastro de pacientes e pipeline harmonizado.
Aba 1: Pacientes & Formulários
Aba 2: Raman & Correlação (pipeline tipo Figura 1: despike, baseline,
       smoothing, peak fitting, calibração com Si e gráfico multi-painel)
"""

from typing import List, Tuple, Dict, Optional, Any
import json
import io
import inspect

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------
# CONFIG STREAMLIT
# ---------------------------------------------------------------------
st.set_page_config(page_title="Plataforma Bio-Raman", layout="wide")

# ---------------------------------------------------------------------
# IMPORT SEGURO DO MÓDULO RAMAN
# ---------------------------------------------------------------------
try:
    import raman_processing as rp
except Exception as e:
    st.title("Plataforma Bio-Raman")
    st.error("Erro ao importar o arquivo **raman_processing.py**.")
    st.caption(
        "Verifique se o arquivo está no mesmo diretório do app, se não há erros de sintaxe "
        "e se todas as dependências estão instaladas."
    )
    st.exception(e)
    st.stop()

# ---------------------------------------------------------------------
# FUNÇÕES ABA 1 (PACIENTES)
# ---------------------------------------------------------------------
def load_patient_table(file) -> pd.DataFrame:
    name = getattr(file, "name", "").lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    elif name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file)
    else:
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, sep=r"\s+", engine="python", header=None)
    return df


def guess_gender_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in ["sexo", "gênero", "genero", "sex", "gender"])
    ]
    return candidates[0] if candidates else None


def guess_smoker_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in ["fuma", "fumante", "smoker"])
    ]
    return candidates[0] if candidates else None


def guess_disease_column(df: pd.DataFrame) -> Optional[str]:
    keys = ["doença", "doenca", "doencas", "comorb", "diagnostico", "diagnóstico", "disease"]
    candidates = [c for c in df.columns if any(k in c.lower() for k in keys)]
    return candidates[0] if candidates else None


def normalize_gender(value) -> str:
    if pd.isna(value):
        return "Não informado"
    v = str(value).strip().lower()
    if v in ["f", "fem", "feminino", "female", "woman", "mulher"]:
        return "Feminino"
    if v in ["m", "masc", "masculino", "male", "man", "homem"]:
        return "Masculino"
    return "Não informado"


def normalize_yesno(value) -> str:
    if pd.isna(value):
        return "Não informado"
    v = str(value).strip().lower()
    if v in ["sim", "s", "yes", "y", "1", "true", "verdadeiro"]:
        return "Sim"
    if v in ["não", "nao", "n", "no", "0", "false", "falso"]:
        return "Não"
    return "Não informado"


def compute_patient_stats(df: pd.DataFrame, col_gender, col_smoker, col_disease):
    stats = {}
    if col_gender:
        g = df[col_gender].map(normalize_gender)
        stats["sexo"] = (g.value_counts(normalize=True) * 100).round(1).rename("percentual").to_frame()
    if col_smoker:
        s = df[col_smoker].map(normalize_yesno)
        stats["fumante"] = (s.value_counts(normalize=True) * 100).round(1).rename("percentual").to_frame()
    if col_disease:
        d = df[col_disease].map(normalize_yesno)
        stats["doenca"] = (d.value_counts(normalize=True) * 100).round(1).rename("percentual").to_frame()
    return stats


def compute_association_gender_smoker_disease(df, col_gender, col_smoker, col_disease):
    if not (col_gender and col_smoker and col_disease):
        return None

    df2 = pd.DataFrame({
        "Sexo": df[col_gender].map(normalize_gender),
        "Fumante": df[col_smoker].map(normalize_yesno),
        "Doenca": df[col_disease].map(normalize_yesno),
    })

    df_pos = df2[df2["Doenca"] == "Sim"]
    if df_pos.empty:
        return None

    tab = pd.crosstab(df_pos["Sexo"], df_pos["Fumante"], normalize="index") * 100
    return tab.round(1)
# ---------------------------------------------------------------------
# GRÁFICOS (ABA 1)
# ---------------------------------------------------------------------
def plot_percentage_bar(series: pd.Series, title: str, ylabel: str = "% de pacientes"):
    fig, ax = plt.subplots(figsize=(3, 2.5))
    labels = [str(i) for i in series.index]
    values = series.values
    ax.bar(labels, values)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, max(100, values.max() * 1.1))

    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.03, f"{v:.1f}%", ha="center", fontsize=7)

    plt.tight_layout()
    return fig


def plot_association_bar(tab: pd.DataFrame, title: str = ""):
    fig, ax = plt.subplots(figsize=(4, 3))
    index = np.arange(len(tab.index))
    cols = list(tab.columns)
    width = 0.8 / len(cols)

    for i, col in enumerate(cols):
        ax.bar(index + i * width, tab[col].values, width, label=str(col))

    ax.set_xticks(index + width)
    ax.set_xticklabels(tab.index)
    ax.set_ylabel("% de pacientes com doença", fontsize=9)
    ax.set_title(title or "Associação sexo × fumante")
    ax.legend(fontsize=7)
    plt.tight_layout()
    return fig

# ---------------------------------------------------------------------
# FIGURA TIPO (a–g)
# ---------------------------------------------------------------------
def plot_pipeline_panels(
    x_raw, y_raw, x_proc, y_proc, x_cal, y_cal, peaks, despike_metrics
):
    fig = plt.figure(figsize=(11, 10))
    gs = GridSpec(4, 3, figure=fig, wspace=0.7, hspace=0.9)

    # a) Synthetic / Raw
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(x_raw, y_raw, lw=0.8)
    ax_a.set_title("a) Synthetic / Raw")

    # b) Spike removal
    ax_b = fig.add_subplot(gs[0, 1])
    y_med, _, _ = rp.compare_despike_algorithms(y_raw)
    ax_b.plot(x_raw, y_raw, lw=0.5, label="raw")
    ax_b.plot(x_raw, y_med, lw=0.9, label="despiked")
    ax_b.set_title("b) Spike removal")
    ax_b.legend(fontsize=7)

    # b-metric
    ax_bm = fig.add_subplot(gs[0, 2])
    methods = list(despike_metrics.keys())
    vals = [despike_metrics[m] for m in methods]
    ax_bm.bar(np.arange(len(methods)), vals)
    ax_bm.set_xticks(np.arange(len(methods)))
    ax_bm.set_xticklabels(methods, rotation=45)
    ax_bm.set_title("Metrics vs algorithms")

    # c) Baseline
    ax_c = fig.add_subplot(gs[1, 0])
    base = rp.baseline_als(y_med)
    ax_c.plot(x_raw, y_med, lw=0.8)
    ax_c.plot(x_raw, base, lw=0.8)
    ax_c.set_title("c) Baseline")

    # d) Smoothing
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.plot(x_raw, y_med, lw=0.5, label="before")
    ax_d.plot(x_proc, y_proc, lw=0.9, label="after")
    ax_d.set_title("d) Smoothing")
    ax_d.legend(fontsize=7)

    # e) Peak fitting
    ax_e = fig.add_subplot(gs[1, 2])
    ax_e.plot(x_cal, y_cal, lw=0.9)
    for p in peaks:
        ax_e.axvline(p.position_cm1, color="r", lw=0.6)
        ax_e.plot(p.position_cm1, p.intensity, "ro", ms=3)
    ax_e.set_title("e) Peaks")

    # f) Calibration
    ax_f = fig.add_subplot(gs[2, :])
    ax_f.plot(x_proc, y_proc, lw=0.7, label="raw-axis")
    ax_f.plot(x_cal, y_proc, lw=0.9, label="calibrated")
    ax_f.set_title("f) Calibration")
    ax_f.legend()

    # g) Example code
    ax_g = fig.add_subplot(gs[3, :])
    ax_g.axis("off")
    ax_g.text(
        0.01, 0.9,
        "spec = rp.load_spectrum('file.txt')\n"
        "x, y = spec\n"
        "x, y_proc, meta = rp.preprocess_spectrum(x, y)\n"
        "peaks = rp.detect_peaks(x, y_proc)\n"
        "peaks = rp.fit_peaks(x, y_proc, peaks)",
        fontsize=9, family="monospace"
    )

    fig.suptitle("Pipeline Raman – estilo Figura 1", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# ---------------------------------------------------------------------
# GRÁFICO FINAL MULTI-PEAK
# ---------------------------------------------------------------------
def plot_final_multipeak_fit(x_cal, y_proc, peaks, title="Ajuste multi-pico (1000–1700 cm⁻¹)"):

    mask = (x_cal >= 990) & (x_cal <= 1700)
    if np.sum(mask) >= 5:
        x_r = x_cal[mask]
        y_r = y_proc[mask]
    else:
        x_r = x_cal
        y_r = y_proc

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_r, y_r, color="0.4", lw=1.2)

    y_fit_tot = np.zeros_like(x_r)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(peaks))))

    for i, p in enumerate(peaks):
        if not p.fit_params:
            continue

        if "cen" in p.fit_params:
            cen = p.fit_params["cen"]
            amp = p.fit_params["amp"]
            wid = p.fit_params["wid"]
        else:
            cen = p.fit_params.get("center", p.position_cm1)
            amp = p.fit_params.get("amplitude", p.intensity)
            wid = p.fit_params.get("sigma", 3.0)

        y_comp = rp.gaussian(x_r, amp, cen, wid)
        y_fit_tot += y_comp

        ax.plot(x_r, y_comp, "--", lw=1.0, alpha=0.8, color=colors[i])

    ax.plot(x_r, y_fit_tot, color="red", lw=2, label="Soma ajustada")

    # marcação dos picos
    xs = [p.position_cm1 for p in peaks]
    ys = np.interp(xs, x_cal, y_proc)
    ax.plot(xs, ys, "bx", ms=7)

    # anotações
    for p in peaks:
        if not p.group:
            continue
        yv = float(np.interp(p.position_cm1, x_cal, y_proc))
        ax.annotate(
            f"{p.group}\n~{p.position_cm1:.0f} cm⁻¹",
            xy=(p.position_cm1, yv),
            xytext=(p.position_cm1 + 10, yv + 0.03),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=8, color="red"
        )

    ax.set_title(title)
    ax.set_xlabel("Deslocamento Raman (cm⁻¹)")
    ax.set_ylabel("Intensidade normalizada")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    plt.tight_layout()
    return fig
# ---------------------------------------------------------------------
# INTERFACE – ABAS
# ---------------------------------------------------------------------
st.title("Plataforma Bio-Raman")
tab_pacientes, tab_raman = st.tabs(["1 Pacientes & Formulários", "2 Raman & Correlação"])

# =====================================================================
# =======================  ABA 1 – PACIENTES  ==========================
# =====================================================================
with tab_pacientes:
    st.header("Cadastro de pacientes via planilha")

    patient_file = st.file_uploader(
        "Carregar planilha de pacientes (XLS, XLSX ou CSV)",
        type=["xls", "xlsx", "csv"],
    )

    if patient_file:
        df_pac = load_patient_table(patient_file)
        st.subheader("Pré-visualização")
        st.dataframe(df_pac.head())

        cols = list(df_pac.columns)

        col_g, col_s, col_d = st.columns(3)
        with col_g:
            col_gender = st.selectbox(
                "Coluna de sexo/gênero",
                ["(nenhuma)"] + cols,
                index=(cols.index(guess_gender_column(df_pac)) + 1
                       if guess_gender_column(df_pac) in cols
                       else 0)
            )
            col_gender = None if col_gender == "(nenhuma)" else col_gender

        with col_s:
            col_smoker = st.selectbox(
                "Coluna de fumante (sim/não)",
                ["(nenhuma)"] + cols,
                index=(cols.index(guess_smoker_column(df_pac)) + 1
                       if guess_smoker_column(df_pac) in cols
                       else 0)
            )
            col_smoker = None if col_smoker == "(nenhuma)" else col_smoker

        with col_d:
            col_disease = st.selectbox(
                "Coluna 'tem alguma doença?'",
                ["(nenhuma)"] + cols,
                index=(cols.index(guess_disease_column(df_pac)) + 1
                       if guess_disease_column(df_pac) in cols
                       else 0)
            )
            col_disease = None if col_disease == "(nenhuma)" else col_disease

        if st.button("Gerar estatísticas"):
            stats = compute_patient_stats(df_pac, col_gender, col_smoker, col_disease)
            st.subheader("Resumo estatístico")
            st.write(f"**Total de pacientes:** {len(df_pac)}")

            if "sexo" in stats:
                c1, c2 = st.columns(2)
                with c1: st.dataframe(stats["sexo"])
                with c2: st.pyplot(plot_percentage_bar(stats["sexo"]["percentual"], "Distribuição por sexo"))

            if "fumante" in stats:
                c1, c2 = st.columns(2)
                with c1: st.dataframe(stats["fumante"])
                with c2: st.pyplot(plot_percentage_bar(stats["fumante"]["percentual"], "Fumantes"))

            if "doenca" in stats:
                c1, c2 = st.columns(2)
                with c1: st.dataframe(stats["doenca"])
                with c2: st.pyplot(plot_percentage_bar(stats["doenca"]["percentual"], "Doença declarada"))

            tab_assoc = compute_association_gender_smoker_disease(df_pac, col_gender, col_smoker, col_disease)
            if tab_assoc is not None:
                st.subheader("Associação entre sexo × fumante (pacientes com doença)")
                c1, c2 = st.columns(2)
                with c1: st.dataframe(tab_assoc)
                with c2: st.pyplot(plot_association_bar(tab_assoc))

# =====================================================================
# =======================  ABA 2 – RAMAN  =============================
# =====================================================================
with tab_raman:
    st.header("Pipeline Raman – calibração + picos + correlação")

    col1, col2 = st.columns(2)

    with col1:
        sample_file = st.file_uploader("Amostra", type=["txt", "csv", "xlsx"])
        paper_file = st.file_uploader("Papel / substrato", type=["txt", "csv", "xlsx"])
        si_file = st.file_uploader("Silício (padrão)", type=["txt", "csv", "xlsx"])

    with col2:
        use_lmfit = st.checkbox("Usar lmfit (Voigt)", value=rp.LMFIT_AVAILABLE)
        silicon_ref_value = st.number_input("Pico do Si (cm⁻¹)", value=520.7)
        coeffs_str = st.text_input(
            "Coeficientes do polinômio base (ex.: 1.2e-7, -0.03, 550)",
        )

    st.markdown("---")
    colA, colB = st.columns(2)
    run_raman = colA.button("Executar pipeline Raman completo")
    run_fig = colB.button("Mostrar figura tipo (a–g)")

    # ================================================================
    #  PIPELINE PRINCIPAL
    # ================================================================
    if run_raman:

        if not sample_file:
            st.error("Carregue o espectro da amostra.")
            st.stop()

        if not si_file:
            st.error("Carregue o espectro de Silício.")
            st.stop()

        if not coeffs_str.strip():
            st.error("Informe os coeficientes do polinômio.")
            st.stop()

        try:
            base_poly_coeffs = np.fromstring(coeffs_str, sep=",")
            if base_poly_coeffs.size == 0:
                raise ValueError("Nenhum coeficiente numérico encontrado.")
        except Exception as e:
            st.error(f"Erro: {e}")
            st.stop()

        progress = st.progress(0)

        def update(p, text=""):
            progress.progress(int(p), text=text)

        try:
            res = rp.calibrate_with_fixed_pattern_and_silicon(
                silicon_file=si_file,
                sample_file=sample_file,
                paper_file=paper_file,
                base_poly_coeffs=base_poly_coeffs,
                silicon_ref_position=float(silicon_ref_value),
                progress_cb=update,
            )
        except Exception as e:
            st.error("Erro no pipeline Raman.")
            st.exception(e)
            st.stop()

        progress.empty()
        st.success("Pipeline concluído.")

        # Recuperação dos dados
        x_raw = res["x_sample_raw"]
        y_raw = res["y_sample_raw"]
        x_proc = res["x_sample_proc"]
        y_proc = res["y_sample_proc"]
        x_cal = res["x_sample_calibrated"]

        # --- Gráfico Raw vs Processado / Calibrado ---
        fig, axs = plt.subplots(1, 2, figsize=(13, 4))
        axs[0].plot(x_raw, y_raw, lw=0.7)
        axs[0].plot(x_proc, y_proc, lw=0.9)
        axs[0].set_title("Raw vs Processado")

        axs[1].plot(x_cal, y_proc, lw=0.9)
        axs[1].set_title("Processado no eixo calibrado")
        st.pyplot(fig)

        # --- Picos ---
        peaks = rp.detect_peaks(x_cal, y_proc, height=0.05, distance=5, prominence=0.02)
        peaks = rp.fit_peaks(x_cal, y_proc, peaks, use_lmfit=use_lmfit)
        peaks = rp.map_peaks_to_molecular_groups(peaks)

        # =====================================================================
        # TABELA 1 — PICOS DETECTADOS
        # =====================================================================
        st.subheader("Tabela: Picos detectados")

        if peaks:
            df_peaks = pd.DataFrame([
                {
                    "position_cm-1": float(p.position_cm1),
                    "intensity": float(p.intensity),
                    "width": float(p.width) if p.width else "",
                    "group": p.group or "Sem classificação",
                }
                for p in peaks
            ])
        else:
            df_peaks = pd.DataFrame(columns=["position_cm-1", "intensity", "width", "group"])

        st.dataframe(df_peaks)

        # =====================================================================
        # TABELA 2 — AGRUPAMENTO POR GRUPO MOLECULAR
        # =====================================================================
        st.subheader("Tabela: Agregação por grupo molecular")

        total_peaks = max(1, len(df_peaks))
        group_counts = (
            df_peaks["group"]
            .value_counts()
            .rename_axis("group")
            .reset_index(name="n_peaks")
        )
        group_counts["pct"] = (group_counts["n_peaks"] / total_peaks * 100).round(1)

        def diseases_for_group(g):
            return [
                rule["name"]
                for rule in rp.DISEASE_RULES
                if g in rule["groups_required"]
            ]

        group_counts["linked_diseases"] = group_counts["group"].apply(diseases_for_group)
        st.dataframe(group_counts)

        # =====================================================================
        # TABELA 3 — CORRELAÇÃO GRUPOS ↔ DOENÇAS
        # =====================================================================
        st.subheader("Tabela: Correlação grupo → condições possíveis")

        detected_groups = set(df_peaks["group"]) - {"Sem classificação"}

        rows = []
        for rule in rp.DISEASE_RULES:
            req = set(rule["groups_required"])
            present = len(req.intersection(detected_groups))
            total = len(req)
            score = round((present / total) * 100, 1) if total > 0 else 0

            rows.append({
                "disease": rule["name"],
                "required_groups": ", ".join(req),
                "present_groups": present,
                "total_required": total,
                "correlation_%": score,
                "description": rule["description"],
            })

        df_corr = pd.DataFrame(rows).sort_values("correlation_%", ascending=False)
        st.dataframe(df_corr)

        # Gráfico multi-pico final
        st.subheader("Ajuste multi-pico (1000–1700 cm⁻¹)")
        st.pyplot(plot_final_multipeak_fit(x_cal, y_proc, peaks))

# =====================================================================
# FIGURA TIPO (a–g)
# =====================================================================
    if run_fig:
        if not sample_file:
            st.error("Carregue um espectro para gerar a figura.")
            st.stop()

        x_raw, y_raw = rp.load_spectrum(sample_file)
        x_proc, y_proc, meta = rp.preprocess_spectrum(x_raw, y_raw)

        despike_metrics = meta.get("despike_metrics", {})

        # Tenta calibrar
        if coeffs_str.strip() and si_file:
            try:
                base_poly = np.fromstring(coeffs_str, sep=",")
                res_tmp = rp.calibrate_with_fixed_pattern_and_silicon(
                    silicon_file=si_file,
                    sample_file=sample_file,
                    paper_file=paper_file,
                    base_poly_coeffs=base_poly,
                    silicon_ref_position=float(silicon_ref_value),
                )
                x_cal = res_tmp["x_sample_calibrated"]
                y_cal = res_tmp["y_sample_proc"]
            except Exception:
                x_cal, y_cal = x_proc, y_proc
        else:
            x_cal, y_cal = x_proc, y_proc

        peaks = rp.detect_peaks(x_cal, y_cal)
        peaks = rp.fit_peaks(x_cal, y_cal, peaks)

        fig = plot_pipeline_panels(
            x_raw, y_raw, x_proc, y_proc, x_cal, y_cal, peaks, despike_metrics
        )
        st.pyplot(fig)

# Rodapé
st.markdown("---")
st.caption("Plataforma Bio-Raman – processamento completo de espectros + análise exploratória.")
