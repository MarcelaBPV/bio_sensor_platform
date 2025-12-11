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
        # tenta leitura genérica separada por whitespace
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

def compute_patient_stats(
    df: pd.DataFrame,
    col_gender: Optional[str],
    col_smoker: Optional[str],
    col_disease: Optional[str],
) -> Dict[str, pd.DataFrame]:
    stats: Dict[str, pd.DataFrame] = {}
    if col_gender:
        g_norm = df[col_gender].map(normalize_gender)
        stats["sexo"] = (
            g_norm.value_counts(normalize=True) * 100
        ).round(1).rename("percentual").to_frame()
    if col_smoker:
        s_norm = df[col_smoker].map(normalize_yesno)
        stats["fumante"] = (
            s_norm.value_counts(normalize=True) * 100
        ).round(1).rename("percentual").to_frame()
    if col_disease:
        d_norm = df[col_disease].map(normalize_yesno)
        stats["doenca"] = (
            d_norm.value_counts(normalize=True) * 100
        ).round(1).rename("percentual").to_frame()
    return stats

def compute_association_gender_smoker_disease(
    df: pd.DataFrame,
    col_gender: Optional[str],
    col_smoker: Optional[str],
    col_disease: Optional[str],
) -> Optional[pd.DataFrame]:
    if not (col_gender and col_smoker and col_disease):
        return None

    g_norm = df[col_gender].map(normalize_gender)
    s_norm = df[col_smoker].map(normalize_yesno)
    d_norm = df[col_disease].map(normalize_yesno)

    df_norm = pd.DataFrame({"Sexo": g_norm, "Fumante": s_norm, "Doenca": d_norm})
    df_pos = df_norm[df_norm["Doenca"] == "Sim"].copy()
    if df_pos.empty:
        return None

    tab = pd.crosstab(
        df_pos["Sexo"],
        df_pos["Fumante"],
        normalize="index",
    ) * 100.0
    return tab.round(1)

def plot_percentage_bar(series: pd.Series, title: str, ylabel: str = "% de pacientes"):
    fig, ax = plt.subplots(figsize=(3, 2.5))
    labels = [str(i) for i in series.index]
    values = series.values
    ax.bar(labels, values)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, max(100, values.max() * 1.1))
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.02, f"{v:.1f}%", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    return fig

def plot_association_bar(tab: pd.DataFrame, title: str = ""):
    fig, ax = plt.subplots(figsize=(4, 3))
    index = np.arange(len(tab.index))
    cols = list(tab.columns)
    ncols = len(cols)
    width = 0.8 / max(ncols, 1)

    for i, col in enumerate(cols):
        ax.bar(index + i * width, tab[col].values, width, label=str(col))

    ax.set_xticks(index + width * (ncols - 1) / 2 if ncols > 1 else index)
    ax.set_xticklabels(tab.index)
    ax.set_ylabel("% de pacientes com doença", fontsize=9)
    ax.set_title(title or "Associação sexo × fumante (com doença)", fontsize=10)
    ax.set_ylim(0, 100)

    for i, sexo in enumerate(tab.index):
        for j, col in enumerate(cols):
            v = tab.loc[sexo, col]
            ax.text(i + j * width, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=7)

    ax.legend(title="Fumante", fontsize=8, title_fontsize=9)
    plt.tight_layout()
    return fig

# ---------------------------------------------------------------------
# FIGURA TIPO (a–g) USANDO raman_processing
# ---------------------------------------------------------------------
def plot_pipeline_panels(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    x_proc: np.ndarray,
    y_proc: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    peaks: List[rp.Peak],
    despike_metrics: Dict[str, float],
):
    fig = plt.figure(figsize=(11, 10))
    gs = GridSpec(4, 3, figure=fig, wspace=0.7, hspace=0.9)

    # a) Synthetic / Raw
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(x_raw, y_raw, lw=0.8)
    ax_a.set_title("a) Synthetic / Raw")
    ax_a.set_xlabel("Wavenumber / cm⁻¹")
    ax_a.set_ylabel("Intensity")

    # b) Spike removal + métricas
    ax_b = fig.add_subplot(gs[0, 1])
    y_med, _, _ = rp.compare_despike_algorithms(y_raw)
    ax_b.plot(x_raw, y_raw, lw=0.5, label="raw")
    ax_b.plot(x_raw, y_med, lw=0.9, label="despiked")
    ax_b.set_title("b) Spike removal")
    ax_b.set_xlabel("Wavenumber / cm⁻¹")
    ax_b.legend(fontsize=7)

    ax_bm = fig.add_subplot(gs[0, 2])
    methods = list(despike_metrics.keys())
    vals = [despike_metrics[m] for m in methods] if methods else []
    ax_bm.bar(np.arange(len(methods)), vals)
    ax_bm.set_xticks(np.arange(len(methods)))
    ax_bm.set_xticklabels(methods, rotation=45, fontsize=7)
    ax_bm.set_title("Metric vs algorithms")

    # c) Baseline
    ax_c = fig.add_subplot(gs[1, 0])
    base = rp.baseline_als(y_med)
    ax_c.plot(x_raw, y_med, lw=0.8, label="despiked")
    ax_c.plot(x_raw, base, lw=0.8, label="baseline")
    ax_c.set_title("c) Baseline calculation")
    ax_c.set_xlabel("Wavenumber / cm⁻¹")
    ax_c.legend(fontsize=7)

    # d) Smoothing
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.plot(x_raw, y_med, lw=0.5, label="before")
    ax_d.plot(x_proc, y_proc, lw=0.9, label="after SG")
    ax_d.set_title("d) Smoothing")
    ax_d.set_xlabel("Wavenumber / cm⁻¹")
    ax_d.legend(fontsize=7)

    # e) Peak fitting
    ax_e = fig.add_subplot(gs[1, 2])
    ax_e.plot(x_cal, y_cal, lw=0.9)
    for p in peaks:
        ax_e.axvline(p.position_cm1, color="r", lw=0.6, alpha=0.8)
        ax_e.plot(p.position_cm1, p.intensity, "ro", ms=3)
    ax_e.set_title("e) Peak fitting")
    ax_e.set_xlabel("Raman shift / cm⁻¹")

    # f) Wavenumber calibration
    ax_f = fig.add_subplot(gs[2, :])
    ax_f.plot(x_proc, y_proc, lw=0.7, label="processado (eixo bruto)")
    ax_f.plot(x_cal, y_proc, lw=0.9, label="processado (eixo calibrado)")
    ax_f.set_title("f) Wavenumber calibration (Si + polinômio)")
    ax_f.set_xlabel("Raman shift / cm⁻¹")
    ax_f.legend(fontsize=8)

    # g) “Coding example”
    ax_g = fig.add_subplot(gs[3, :])
    ax_g.axis("off")
    code = (
        "spec = rp.load_spectrum('spectrum.txt')\n"
        "x, y = spec\n"
        "x, y_proc, meta = rp.preprocess_spectrum(x, y)\n"
        "peaks = rp.detect_peaks(x, y_proc)\n"
        "peaks = rp.fit_peaks(x, y_proc, peaks)"
    )
    ax_g.text(0.01, 0.9, code, family="monospace", fontsize=9, va="top")

    fig.suptitle("Pipeline Raman – visão estilo Figura 1", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# ---------------------------------------------------------------------
# FUNÇÃO: GRÁFICO FINAL MULTI-PIKO (estilo figura exemplo)
# ---------------------------------------------------------------------
def plot_final_multipeak_fit(
    x_cal: np.ndarray,
    y_proc: np.ndarray,
    peaks: List[rp.Peak],
    title: str = "Ajuste multi-pico – região 1000–1700 cm⁻¹",
):
    # seleção de região (fingerprint)
    mask = (x_cal >= 990) & (x_cal <= 1700)
    if np.sum(mask) < 5:
        # se não houver pontos nesta região usa todo eixo
        x_reg = x_cal
        y_reg = y_proc
    else:
        x_reg = x_cal[mask]
        y_reg = y_proc[mask]

    fig, ax = plt.subplots(figsize=(10, 5))

    # espectro processado (cinza)
    ax.plot(x_reg, y_reg, color="0.45", lw=1.2, label="Espectro processado")

    # soma dos componentes
    y_fit_tot = np.zeros_like(x_reg)

    # cores para componentes
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(peaks))))

    for i, p in enumerate(peaks):
        if not p.fit_params:
            continue

        # extrai parâmetros de forma robusta
        if "cen" in p.fit_params:
            cen = p.fit_params.get("cen", p.position_cm1)
            amp = p.fit_params.get("amp", p.intensity)
            wid = p.fit_params.get("wid", p.width or 3.0)
        else:
            cen = p.fit_params.get("center", p.position_cm1)
            amp = p.fit_params.get("amplitude", p.intensity)
            wid = p.fit_params.get("sigma", p.width or 3.0)

        # componente gaussiana para visualização
        y_comp = rp.gaussian(x_reg, amp, cen, wid)
        y_fit_tot += y_comp

        ax.plot(x_reg, y_comp, linestyle="--", linewidth=1.0, alpha=0.9, color=colors[i % len(colors)])

    # soma ajustada (vermelho grosso)
    ax.plot(x_reg, y_fit_tot, color="red", lw=2.0, label="Soma dos ajustes")

    # marcar picos com 'x' azul
    px = [p.position_cm1 for p in peaks]
    if len(px) > 0:
        py = np.interp(px, x_cal, y_proc)
        ax.plot(px, py, "bx", ms=7, mew=1.7, label="Picos")

    # anotações (usa group se disponível)
    for idx, p in enumerate(peaks):
        if not p.group:
            continue
        y_peak = float(np.interp(p.position_cm1, x_cal, y_proc))
        dy = 0.02 + 0.02 * (idx % 3)
        ax.annotate(
            f"{p.group}\n(~{p.position_cm1:.0f} cm⁻¹)",
            xy=(p.position_cm1, y_peak),
            xytext=(p.position_cm1 + 12, y_peak + dy),
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
            fontsize=8,
            color="red",
        )

    ax.set_xlabel("Wave(cm⁻¹)")
    ax.set_ylabel("Int. Norm.")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    return fig

# ---------------------------------------------------------------------
# INTERFACE – ABAS
# ---------------------------------------------------------------------
st.title("Plataforma Bio-Raman")
tab_pacientes, tab_raman = st.tabs(["1 Pacientes & Formulários", "2 Raman & Correlação"])

# ======================= ABA 1 – PACIENTES ===========================
with tab_pacientes:
    st.header("Cadastro de pacientes via planilha")
    patient_file = st.file_uploader(
        "Carregar planilha de pacientes (XLS, XLSX ou CSV)",
        type=["xls", "xlsx", "csv"],
    )
    if patient_file:
        df_pac = load_patient_table(patient_file)
        st.subheader("Pré-visualização da planilha")
        st.dataframe(df_pac.head())

        st.subheader("Mapeamento de colunas")
        cols = list(df_pac.columns)
        default_gender = guess_gender_column(df_pac)
        default_smoker = guess_smoker_column(df_pac)
        default_disease = guess_disease_column(df_pac)

        col_g, col_s, col_d = st.columns(3)
        with col_g:
            col_gender = st.selectbox(
                "Coluna de sexo/gênero",
                options=["(nenhuma)"] + cols,
                index=(cols.index(default_gender) + 1) if default_gender in cols else 0,
            )
            col_gender = None if col_gender == "(nenhuma)" else col_gender
        with col_s:
            col_smoker = st.selectbox(
                "Coluna de fumante (sim/não)",
                options=["(nenhuma)"] + cols,
                index=(cols.index(default_smoker) + 1) if default_smoker in cols else 0,
            )
            col_smoker = None if col_smoker == "(nenhuma)" else col_smoker
        with col_d:
            col_disease = st.selectbox(
                "Coluna de 'tem alguma doença?' (sim/não)",
                options=["(nenhuma)"] + cols,
                index=(cols.index(default_disease) + 1) if default_disease in cols else 0,
            )
            col_disease = None if col_disease == "(nenhuma)" else col_disease

        if st.button("Calcular estatísticas dos pacientes"):
            stats = compute_patient_stats(df_pac, col_gender, col_smoker, col_disease)
            st.subheader("Resumo estatístico")
            st.markdown(f"**Total de registros:** {len(df_pac)}")

            if "sexo" in stats:
                st.markdown("### Distribuição de sexo/gênero (%)")
                c1, c2 = st.columns(2)
                with c1:
                    st.dataframe(stats["sexo"])
                with c2:
                    fig_sexo_bar = plot_percentage_bar(stats["sexo"]["percentual"], "Sexo/gênero – barras")
                    st.pyplot(fig_sexo_bar)

            if "fumante" in stats:
                st.markdown("### Fumante (%)")
                c1, c2 = st.columns(2)
                with c1:
                    st.dataframe(stats["fumante"])
                with c2:
                    fig_fum_bar = plot_percentage_bar(stats["fumante"]["percentual"], "Fumante – barras")
                    st.pyplot(fig_fum_bar)

            if "doenca" in stats:
                st.markdown("### Alguma doença declarada (%)")
                c1, c2 = st.columns(2)
                with c1:
                    st.dataframe(stats["doenca"])
                with c2:
                    fig_doenc_bar = plot_percentage_bar(stats["doenca"]["percentual"], "Doença declarada – barras")
                    st.pyplot(fig_doenc_bar)

            assoc_tab = compute_association_gender_smoker_disease(df_pac, col_gender, col_smoker, col_disease)
            st.markdown("### Associação entre sexo, tabagismo e presença de doença")
            if assoc_tab is not None:
                st.caption(
                    "Tabela e gráfico mostram, entre os pacientes COM doença declarada, "
                    "a distribuição percentual por sexo e status de fumante."
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.dataframe(assoc_tab)
                with c2:
                    fig_assoc = plot_association_bar(assoc_tab, "Pacientes com doença – % por sexo × fumante")
                    st.pyplot(fig_assoc)
            else:
                st.info("Não foi possível calcular a associação (faltam colunas mapeadas ou não há pacientes com doença = 'Sim').")

            st.caption("Obs.: 'Não informado' inclui valores vazios, nulos ou não reconhecidos.")

# ======================= ABA 2 – RAMAN ===============================
with tab_raman:
    st.header("Pipeline Raman – calibração, picos e correlação")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Arquivos de espectros")
        sample_file = st.file_uploader("Amostra (sangue em papel, etc.)", type=["txt", "csv", "xlsx"])
        paper_file = st.file_uploader("Papel / substrato (background)", type=["txt", "csv", "xlsx"])
        si_file = st.file_uploader("Silício (padrão para ajuste fino)", type=["txt", "csv", "xlsx"])

    with col2:
        st.subheader("Configurações do processamento")
        use_lmfit = st.checkbox("Usar lmfit (ajuste multi-peak Voigt)", value=rp.LMFIT_AVAILABLE)
        silicon_ref_value = st.number_input("Posição de referência do pico do Silício (cm⁻¹)", value=520.7, format="%.2f")
        coeffs_str = st.text_input("Coeficientes do polinômio base (np.polyfit, separados por vírgula)", help="Ex.: 1.2e-7, -0.03, 550.0")

    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_raman = st.button("Executar pipeline Raman completo")
    with col_btn2:
        run_fig = st.button("Mostrar figura tipo (a–g)")

    # -------- Pipeline principal ----------
    if run_raman:
        if not sample_file:
            st.error("Carregue o espectro da amostra.")
        elif not si_file:
            st.error("Carregue o espectro de Silício.")
        elif not coeffs_str.strip():
            st.error("Informe os coeficientes do polinômio base.")
        else:
            try:
                base_poly_coeffs = np.fromstring(coeffs_str, sep=",")
                if base_poly_coeffs.size == 0:
                    raise ValueError("Nenhum coeficiente encontrado.")
            except Exception as e:
                st.error(f"Erro ao interpretar os coeficientes do polinômio base: {e}")
                st.stop()

            progress = st.progress(0, text="Iniciando pipeline...")
            def set_progress(p, text=""):
                progress.progress(int(p), text=text)

            with st.spinner("Processando espectros..."):
                try:
                    # chamando de forma robusta: tenta passar paper_file se função aceitar
                    cal_fn = rp.calibrate_with_fixed_pattern_and_silicon
                    sig = inspect.signature(cal_fn)
                    kwargs = dict(
                        silicon_file=si_file,
                        sample_file=sample_file,
                        base_poly_coeffs=base_poly_coeffs,
                        silicon_ref_position=float(silicon_ref_value),
                        progress_cb=set_progress,
                    )
                    if "paper_file" in sig.parameters:
                        kwargs["paper_file"] = paper_file
                    res = cal_fn(**kwargs)
                except Exception as e:
                    progress.empty()
                    st.error(f"Erro no pipeline Raman: {e}")
                    st.exception(e)
                    st.stop()

            progress.empty()
            st.success("Pipeline Raman concluído.")

            calib_warning = res.get("calibration", {}).get("warning")
            if calib_warning:
                st.warning(f"Aviso na calibração: {calib_warning}")

            x_raw = res["x_sample_raw"]
            y_raw = res["y_sample_raw"]
            x_proc = res["x_sample_proc"]
            y_proc = res["y_sample_proc"]
            x_cal = res["x_sample_calibrated"]

            fig, axs = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
            axs[0].plot(x_raw, y_raw, lw=0.6, label="Raw")
            axs[0].plot(x_proc, y_proc, lw=0.9, label="Processado")
            axs[0].set_xlabel("Eixo bruto (unidades do equipamento)")
            axs[0].set_title("Raw vs Processado")
            axs[0].legend()

            axs[1].plot(x_cal, y_proc, lw=0.9)
            axs[1].set_xlabel("Deslocamento Raman (cm⁻¹, calibrado)")
            axs[1].set_title("Processado no eixo calibrado (padrão fixo + Si)")

            st.pyplot(fig)

            peaks = rp.detect_peaks(x_cal, y_proc, height=0.05, distance=5, prominence=0.02)
            peaks = rp.fit_peaks(x_cal, y_proc, peaks, use_lmfit=use_lmfit)
            peaks = rp.map_peaks_to_molecular_groups(peaks)
            diseases = rp.infer_diseases(peaks)

            # --- inserção: tabelas detalhadas de picos, grupos e correlação com doenças ---
            # 1) Tabela detalhada de picos
            if peaks:
                df_peaks = pd.DataFrame([
                    {
                        "position_cm-1": float(p.position_cm1),
                        "intensity": float(p.intensity),
                        "width": float(p.width) if p.width is not None else "",
                        "group": p.group or "Sem classificação",
                    }
                    for p in peaks
                ])
            else:
                df_peaks = pd.DataFrame(columns=["position_cm-1", "intensity", "width", "group"])

            st.subheader("Tabela: picos detectados (e grupo molecular associado)")
            st.dataframe(df_peaks)

            # 2) Agregação por grupo molecular
            total_peaks = max(1, len(peaks))  # evita divisão por zero
            group_counts = df_peaks["group"].value_counts(dropna=False).rename_axis("group").reset_index(name="n_peaks")
            group_counts["pct_of_peaks"] = (group_counts["n_peaks"] / total_peaks * 100).round(1)

            def diseases_for_group(group_name: str) -> List[str]:
                if not group_name or group_name == "Sem classificação":
                    return []
                return [rule["name"] for rule in rp.DISEASE_RULES if group_name in rule.get("groups_required", [])]

            group_counts["linked_diseases"] = group_counts["group"].apply(diseases_for_group)

            st.subheader("Tabela: agregação por grupo molecular")
            st.dataframe(group_counts)

            # 3) Tabela de correlação grupo → doença (por regras simples)
            disease_rows = []
            present_groups = set(g for g in df_peaks["group"].unique() if pd.notna(g) and g != "")

            for rule in rp.DISEASE_RULES:
                required = set(rule.get("groups_required", []))
                present = len(required.intersection(present_groups))
                total_required = len(required) if len(required) > 0 else 1
                score_pct = round((present / total_required) * 100, 1)
                disease_rows.append({
                    "disease": rule["name"],
                    "required_groups": ", ".join(required) if required else "",
                    "n_required_present": present,
                    "n_required_total": total_required,
                    "correlation_%": score_pct,
                    "description": rule.get("description", "")
                })

            df_disease_corr = pd.DataFrame(disease_rows).sort_values("correlation_%", ascending=False)

            st.subheader("Tabela: correlação entre grupos detectados e condições (regras)")
            st.caption("Interpretação: % indica quantos dos grupos exigidos pela regra foram detectados no espectro.")
            st.dataframe(df_disease_corr)

            st.markdown("**Regras com pelo menos um grupo detectado:**")
            df_disease_some = df_disease_corr[df_disease_corr["n_required_present"] > 0]
            if not df_disease_some.empty:
                st.table(df_disease_some)
            else:
                st.write("Nenhuma regra tem grupos detectados no espectro atual.")
            # --- fim da inserção ---

            st.subheader("Picos detectados e ajustados (eixo calibrado)")
            if peaks:
                df_peaks_fit = pd.DataFrame(
                    [
                        {
                            "position_cm-1": p.position_cm1,
                            "intensity": p.intensity,
                            "width": p.width or "",
                            "group": p.group,
                            "fit_params": json.dumps(p.fit_params) if p.fit_params else "",
                        }
                        for p in peaks
                    ]
                )
                st.dataframe(df_peaks_fit)

                # GRÁFICO FINAL MULTI-PIKO
                st.subheader("Ajuste multi-pico (região 1000–1700 cm⁻¹)")
                fig_final = plot_final_multipeak_fit(x_cal, y_proc, peaks)
                st.pyplot(fig_final)

            else:
                st.info("Nenhum pico detectado com os parâmetros atuais.")

            st.subheader("Correlação com padrões de 'doenças' (modo pesquisa)")
            st.caption("⚠ Uso exploratório / pesquisa. NÃO é diagnóstico médico.")
            if diseases:
                st.table(pd.DataFrame(diseases))
            else:
                st.write("Nenhum padrão identificado com as regras atuais.")

            if rp.H5PY_AVAILABLE:
                bytes_h5 = rp.save_to_nexus_bytes(x_cal, y_proc, {"calibration": json.dumps(res["calibration"])})
                st.download_button("Baixar espectro calibrado (NeXus-like .h5)", data=bytes_h5, file_name="sample_calibrated.h5", mime="application/octet-stream")
            else:
                st.info("Instale 'h5py' para habilitar export HDF5: pip install h5py")

    # -------- Figura tipo a–g ----------
    if run_fig:
        if not sample_file:
            st.error("Carregue pelo menos o espectro da amostra.")
        else:
            # 1) Ler amostra uma vez
            try:
                x_raw, y_raw = rp.load_spectrum(sample_file)
            except Exception as e:
                st.error("Erro ao ler espectro da amostra para figura (a–g).")
                st.exception(e)
                st.stop()

            x_proc, y_proc, meta = rp.preprocess_spectrum(x_raw, y_raw)
            despike_metrics = meta.get("despike_metrics", {})
            if not despike_metrics:
                _, _, despike_metrics = rp.compare_despike_algorithms(y_raw)

            # 2) Tentar usar calibração, mas com cópias (BytesIO) dos arquivos
            if coeffs_str.strip() and si_file is not None:
                try:
                    base_poly_coeffs = np.fromstring(coeffs_str, sep=",")
                    if base_poly_coeffs.size == 0:
                        raise ValueError("Nenhum coeficiente encontrado.")

                    # criar cópias em memória com ponteiro no início
                    si_buf = io.BytesIO(si_file.getvalue())
                    si_buf.name = si_file.name

                    sample_buf = io.BytesIO(sample_file.getvalue())
                    sample_buf.name = sample_file.name

                    paper_buf = None
                    if paper_file is not None:
                        paper_buf = io.BytesIO(paper_file.getvalue())
                        paper_buf.name = paper_file.name

                    # tenta chamar calibracao com ou sem paper_file dependendo da assinatura
                    cal_fn = rp.calibrate_with_fixed_pattern_and_silicon
                    sig = inspect.signature(cal_fn)
                    kwargs = dict(
                        silicon_file=si_buf,
                        sample_file=sample_buf,
                        base_poly_coeffs=base_poly_coeffs,
                        silicon_ref_position=float(silicon_ref_value),
                    )
                    if "paper_file" in sig.parameters:
                        kwargs["paper_file"] = paper_buf

                    res_tmp = cal_fn(**kwargs)
                    x_cal = res_tmp["x_sample_calibrated"]
                    y_cal = res_tmp["y_sample_proc"]
                except Exception as e:
                    st.error("Erro ao aplicar calibração para figura (a–g). Usando eixo bruto.")
                    st.exception(e)
                    x_cal, y_cal = x_proc, y_proc
            else:
                x_cal, y_cal = x_proc, y_proc

            peaks = rp.detect_peaks(x_cal, y_cal, height=0.05, distance=5, prominence=0.02)
            peaks = rp.fit_peaks(x_cal, y_cal, peaks, use_lmfit=use_lmfit)

            fig_pipeline = plot_pipeline_panels(x_raw, y_raw, x_proc, y_proc, x_cal, y_cal, peaks, despike_metrics)
            st.pyplot(fig_pipeline)

# Rodapé
st.markdown("---")
st.caption("Aba 1: cadastro e estatísticas de pacientes • Aba 2: Raman harmonizado + calibração fixa + Si + visualização estilo Figura 1.")
