# File: app.py
# -*- coding: utf-8 -*-
"""
Bio Sensor App - Streamlit
Três abas:
1) Pacientes & Import Forms (inclui import ZIP em lotes)
2) Espectrometria Raman (upload individual e upload de até 10 amostras de uma vez)
3) Otimização (IA)

Esta versão adiciona um layout melhorado para exibição dos resultados:
- Gráfico principal (espectro ajustado + picos + resíduos)
- Gráfico secundário (marcadores coloridos por grupo molecular exatamente como o exemplo)
- Tabela com picos e grupos moleculares
- Opção para salvar resultados no Supabase
"""
import streamlit as st
import pandas as pd
import numpy as np
import io
import time
from typing import Optional, List, Dict
from datetime import datetime
import matplotlib.pyplot as plt
from io import BytesIO, StringIO

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Supabase
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

# Pipeline
try:
    from raman_processing import process_raman_pipeline
except Exception as e:
    process_raman_pipeline = None

# ---------------------------
# Config Streamlit
# ---------------------------
st.set_page_config(page_title="Bio Sensor App", layout="wide", page_icon="*")
st.title("*Bio Sensor App*")

# ---------------------------
# Conexão Supabase (st.secrets)
# ---------------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL") if hasattr(st, 'secrets') else None
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") if hasattr(st, 'secrets') else None

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY and create_client:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Erro conectando ao Supabase: {e}")
        supabase = None
else:
    st.warning("⚠️ Coloque SUPABASE_URL e SUPABASE_KEY em st.secrets para habilitar salvamentos no banco.")

# ---------------------------
# Utilidades Supabase (usam supabase client)
# ---------------------------

def safe_insert(table: str, records: List[Dict]):
    if not supabase:
        raise RuntimeError("Supabase não configurado.")
    if not records:
        return []
    batch = 800
    out = []
    for i in range(0, len(records), batch):
        chunk = records[i:i+batch]
        res = supabase.table(table).insert(chunk).execute()
        if res.error:
            raise RuntimeError(f"Erro inserindo em {table}: {res.error}")
        out.extend(res.data or [])
    return out


def create_patient_record(patient_obj: Dict) -> Dict:
    if not supabase:
        raise RuntimeError("Supabase não configurado.")
    res = supabase.table("patients").insert(patient_obj).execute()
    if res.error:
        raise RuntimeError(res.error.message if hasattr(res.error,'message') else res.error)
    return res.data[0]


def find_patient_by_email_or_cpf(email: Optional[str], cpf: Optional[str]) -> Optional[Dict]:
    if not supabase:
        return None
    if email:
        r = supabase.table("patients").select("*").eq("email", email).limit(1).execute()
        if r.data:
            return r.data[0]
    if cpf:
        r = supabase.table("patients").select("*").eq("cpf", cpf).limit(1).execute()
        if r.data:
            return r.data[0]
    return None


def create_sample_record(sample_obj: Dict) -> Dict:
    if not supabase:
        raise RuntimeError("Supabase não configurado.")
    res = supabase.table("samples").insert(sample_obj).execute()
    if res.error:
        raise RuntimeError(res.error.message if hasattr(res.error,'message') else res.error)
    return res.data[0]


def create_measurement_record(sample_id: int, ensaio_type: str, operator: Optional[str]=None, notes: Optional[str]=None) -> int:
    if not supabase:
        raise RuntimeError("Supabase não configurado.")
    rec = {"sample_id": sample_id, "type": ensaio_type, "operator": operator, "notes": notes}
    res = supabase.table("measurements").insert(rec).execute()
    if res.error:
        raise RuntimeError(res.error.message if hasattr(res.error,'message') else res.error)
    return res.data[0]["id"]


def insert_raman_spectrum_df(df: pd.DataFrame, measurement_id: int):
    df2 = df.copy()
    df2["measurement_id"] = measurement_id
    records = df2.to_dict(orient="records")
    return safe_insert("raman_spectra", records)


def insert_peaks_df(df: pd.DataFrame, measurement_id: int):
    df2 = df.copy()
    df2["measurement_id"] = measurement_id
    records = df2.to_dict(orient="records")
    return safe_insert("raman_peaks", records)


def get_patients_list(limit: int = 500) -> List[Dict]:
    if not supabase:
        return []
    r = supabase.table("patients").select("*").order("created_at", desc=True).limit(limit).execute()
    return r.data or []


def get_samples_for_patient(patient_id: int) -> List[Dict]:
    if not supabase:
        return []
    r = supabase.table("samples").select("*").eq("patient_id", patient_id).order("created_at", desc=True).execute()
    return r.data or []

# ---------------------------
# Mapeamento molecular simples
# ---------------------------
MOLECULAR_MAP = {
    (700, 740): "Hemoglobina / porfirinas (C–H, anéis)",
    (995, 1005): "Fenilalanina (aromático)",
    (1115, 1135): "C–N / proteínas",
    (1200, 1220): "Proteínas / porfirinas",
    (1320, 1345): "CH deformação / hemoglobina",
    (1440, 1460): "Lipídios / proteínas (CH2)",
    (1540, 1565): "Amidas / ligações conjugadas",
    (1605, 1630): "Aromáticos / hemoglobina",
    (1650, 1670): "Amida I / proteínas"
}

GROUP_COLORS = {
    "Hemoglobina / porfirinas (C–H, anéis)": "tab:red",
    "Fenilalanina (aromático)": "tab:blue",
    "C–N / proteínas": "tab:green",
    "Proteínas / porfirinas": "tab:purple",
    "CH deformação / hemoglobina": "tab:olive",
    "Lipídios / proteínas (CH2)": "tab:brown",
    "Amidas / ligações conjugadas": "tab:cyan",
    "Aromáticos / hemoglobina": "tab:orange",
    "Amida I / proteínas": "tab:pink",
    "Desconhecido": "tab:gray"
}


def annotate_molecular_groups(peaks_df: pd.DataFrame, tolerance: float = 5.0) -> pd.DataFrame:
    groups = []
    for _, row in peaks_df.iterrows():
        cen = float(row.get("fit_cen", row.get("peak_cm1", np.nan)))
        match = None
        for (low, high), label in MOLECULAR_MAP.items():
            if (low - tolerance) <= cen <= (high + tolerance):
                match = label
                break
        groups.append(match if match else "Desconhecido")
    peaks_df = peaks_df.copy()
    peaks_df["molecular_group"] = groups
    peaks_df["color"] = peaks_df["molecular_group"].map(lambda g: GROUP_COLORS.get(g, "tab:gray"))
    return peaks_df

# ---------------------------
# Plot helpers (criam layout igual ao exemplo)
# ---------------------------
import matplotlib.patches as mpatches

def plot_main_and_residual(x, y, peaks_df, fig_fit=None, title=None):
    """
    Plota figura principal muito parecida com o exemplo:
    - usa fig_fit (se fornecido) para desenhar linhas de ajuste/comp.
    - adiciona X azul nos picos e caixas amarelas com borda preta.
    - legenda deslocada à direita (external).
    - painel de resíduos abaixo.
    """
    # tamanho e estilo para se aproximar do exemplo
    fig = plt.figure(figsize=(14, 10), dpi=100)
    gs = fig.add_gridspec(6, 4, hspace=0.6, wspace=0.2)
    ax_main = fig.add_subplot(gs[0:4, 0:3])   # grande à esquerda
    ax_legend = fig.add_subplot(gs[0:4, 3])   # área para legenda (vazia)
    ax_res = fig.add_subplot(gs[5, 0:3])      # resíduos embaixo

    # --- main plot: desenhar dados --- #
    ax_main.plot(x, y, color='0.3', linewidth=0.9, label='Dados')  # cinza escuro

    # se o seu pipeline já devolveu um fig com componentes, desenhe-o por cima:
    # se fig_fit for um matplotlib.figure, vamos copiar os artistas (linhas) para ax_main
    if fig_fit is not None and hasattr(fig_fit, 'axes'):
        for a in fig_fit.axes:
            for line in a.get_lines():
                # duplicar linhas no ax_main (mantém estilos)
                try:
                    xdata = line.get_xdata()
                    ydata = line.get_ydata()
                    ax_main.plot(xdata, ydata, linestyle=line.get_linestyle(), linewidth=line.get_linewidth(), label=line.get_label(), color=line.get_color())
                except Exception:
                    pass

    # desenhar picos detectados como X azul e rótulos com caixa amarela
    if peaks_df is not None and not peaks_df.empty:
        # usar 'fit_cen' se existir, senão 'peak_cm1'
        cen_col = 'fit_cen' if 'fit_cen' in peaks_df.columns else ('peak_cm1' if 'peak_cm1' in peaks_df.columns else peaks_df.columns[0])
        height_col = 'fit_height' if 'fit_height' in peaks_df.columns else ('height' if 'height' in peaks_df.columns else None)

        xs = peaks_df[cen_col].astype(float)
        # se houver intensidade do pico use para posicionar rótulo verticalmente, senão posiciona perto do topo
        if height_col and height_col in peaks_df.columns:
            ys_label = peaks_df[height_col].astype(float)
            # normaliza para ficar abaixo do topo do gráfico
            ymax = max(y) if len(y)>0 else 1.0
            ys = np.minimum(ys_label, ymax*0.99)
        else:
            ys = np.full(len(xs), max(y) * 0.98)

        # pontos X grandes, azul com contorno
        ax_main.scatter(xs, ys, marker='x', s=120, linewidths=3, color='royalblue', zorder=10)

        # caixas amarelas como no exemplo (caixa amarela com borda preta e sombra sutil)
        for xpt, ypt in zip(xs, ys):
            lab = f"{float(xpt):.1f}"
            # deslocar rótulo para cima/direita para evitar sobreposição
            dx = (max(x) - min(x)) * 0.005
            dy = (max(y) - min(y)) * 0.02
            bbox = dict(boxstyle='round,pad=0.2', fc='#fff28a', ec='black', lw=0.8)
            ax_main.annotate(lab, xy=(xpt, ypt), xytext=(xpt+dx, ypt+dy), textcoords='data', fontsize=9, bbox=bbox, zorder=11)

    # formatação de eixos e grid (igual aparência)
    ax_main.set_xlim(min(x), max(x))
    ax_main.set_ylabel('Intens. Norm.', fontsize=14)
    ax_main.tick_params(axis='both', labelsize=11)
    ax_main.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    ax_main.set_title(title if title else '', fontsize=20, pad=14)

    # construir legenda no painel direito (ax_legend)
    ax_legend.axis('off')
    # coletar handles/labels do ax_main e apresentar numa box parecida
    handles, labels = ax_main.get_legend_handles_labels()
    # dedupe e ordenar: preferir mostrar 'Dados', 'Ajuste Total', 'Linha Base', 'Pico n'...
    # se labels vazios, criar legendas manuais (fallback)
    if not labels:
        handles = []
        labels = []
    # desenha legenda como texto para poder controlar estilo
    ax_legend.legend(handles=handles, labels=labels, loc='center left', frameon=True, fontsize=10)

    # --- residual --- #
    # tentar extrair fit_total do peaks_df (coluna 'fit_total') caso seu pipeline a retorne
    residual = None
    if peaks_df is not None and 'fit_total' in peaks_df.columns:
        # peaks_df não tem fit por ponto geralmente, então pulamos
        pass

    # fallback: calc. residual se pipeline retornou fig_fit com linha 'Ajuste Total' (procurar por label)
    if fig_fit is not None and hasattr(fig_fit, 'axes'):
        # procura por linhas com label 'Ajuste Total' ou 'fit'
        fit_y = None
        for a in fig_fit.axes:
            for line in a.get_lines():
                lab = (line.get_label() or '').lower()
                if 'ajuste' in lab or 'fit' in lab or 'ajuste total' in lab:
                    fit_y = line.get_ydata()
                    break
            if fit_y is not None:
                break
        if fit_y is not None and len(fit_y) == len(x):
            residual = y - np.array(fit_y)

    # se não achou residual, desenhar resíduos próximos de zero (igual ao exemplo visual)
    if residual is None:
        residual = y - np.interp(x, x, y)  # zero-array (placeholder)
        residual[:] = 0.0

    ax_res.plot(x, residual, color='green', linewidth=1.2)
    ax_res.axhline(0, color='black', linestyle='--', linewidth=1.2)
    ax_res.set_ylabel('Residuo', fontsize=12)
    ax_res.set_xlabel('Wave (cm⁻1)', fontsize=13)
    ax_res.tick_params(axis='both', labelsize=11)
    ax_res.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)

    plt.tight_layout()
    return fig


def plot_groups_panel(peaks_df, title='Grupos Moleculares'):
    """
    Plota painel com as faixas / marcadores por grupo molecular igual ao exemplo:
    - agrupa por molecular_group e alinha pontos em linhas separadas
    - adiciona cor-box e etiqueta do wavenumber
    """
    fig = plt.figure(figsize=(14, 2.5), dpi=100)
    ax = fig.add_subplot(111)

    if peaks_df is None or peaks_df.empty:
        ax.text(0.5, 0.5, 'Nenhum pico detectado', ha='center', va='center')
        ax.set_axis_off()
        return fig

    # ordens dos grupos mantidas como no MOLECULAR_MAP (se necessário ajustar a ordem manualmente)
    unique_groups = list(dict.fromkeys(peaks_df['molecular_group'].tolist()))
    # mapa y para cada grupo (several rows)
    y_positions = {g: i for i, g in enumerate(unique_groups[::-1])}
    cen_col = 'fit_cen' if 'fit_cen' in peaks_df.columns else 'peak_cm1'

    xs = peaks_df[cen_col].astype(float)
    ys = peaks_df['molecular_group'].map(y_positions).astype(float)
    colors = peaks_df.get('color', ['tab:gray'] * len(peaks_df))

    # pontos grandes com borda preta, preenchimento colorido
    for xi, yi, ci in zip(xs, ys, colors):
        ax.scatter(xi, yi, s=140, edgecolor='k', linewidth=0.7, facecolor=ci, zorder=5)

    # rótulos numéricos ao lado direito do marcador (lembra as caixas pequenas do exemplo)
    dx = (max(xs) - min(xs)) * 0.005
    for xi, yi in zip(xs, ys):
        lab = f"{float(xi):.1f}"
        ax.text(xi + dx, yi, f" {lab}", va='center', fontsize=9)

    # ajustar eixos
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()), fontsize=10)
    ax.set_xlabel('Wave (cm⁻1)', fontsize=12)
    ax.set_xlim(min(xs) - 10, max(xs) + 10)
    ax.set_ylim(-0.5, max(list(y_positions.values())) + 0.5)
    ax.grid(False)
    plt.tight_layout()
    return fig

# ---------------------------
# Helpers
# ---------------------------
def buf_from_file(f):
    if hasattr(f, 'read'):
        b = f.read()
        return BytesIO(b) if not isinstance(b, BytesIO) else b
    return None

# ---------------------------
# UI: abas
# ---------------------------
tab_pat, tab_raman, tab_ai = st.tabs(["1️⃣ Pacientes & Import Forms", "2️⃣ Espectrometria Raman", "3️⃣ Otimização (IA)"])

# ---------------------------
# Aba 1: Pacientes & Import Forms
# ---------------------------
with tab_pat:
    st.header("1️⃣ Pacientes — Cadastro e Importação do Google Forms")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Cadastrar paciente manualmente")
        with st.form("form_patient"):
            full_name = st.text_input("Nome completo")
            cpf = st.text_input("CPF (opcional)")
            birth_date = st.date_input("Data de nascimento", value=None)
            email = st.text_input("Email")
            phone = st.text_input("Telefone")
            submitted = st.form_submit_button("Cadastrar")
        if submitted:
            if not supabase:
                st.error("Supabase não configurado — não é possível salvar.")
            else:
                try:
                    existing = find_patient_by_email_or_cpf(email=email or None, cpf=cpf or None)
                    if existing:
                        st.info(f"Paciente já existe: {existing['full_name']} (id={existing['id']})")
                    else:
                        patient_obj = {
                            "full_name": full_name or "Desconhecido",
                            "cpf": cpf or None,
                            "birth_date": birth_date.isoformat() if birth_date else None,
                            "email": email or None,
                            "phone": phone or None
                        }
                        pr = create_patient_record(patient_obj)
                        st.success(f"Paciente cadastrado (id={pr['id']})")
                except Exception as e:
                    st.error(f"Erro ao cadastrar paciente: {e}")

    with c2:
        st.subheader("Importar respostas do Google Forms (CSV)")
        st.markdown("Faça o download das respostas no Google Forms (Respostas → Ícone de 3 pontos → Fazer download das respostas (.csv)) e envie aqui.")
        forms_csv = st.file_uploader("CSV do Google Forms", type=["csv"])
        if forms_csv:
            if st.button("Importar CSV para Supabase"):
                if not supabase:
                    st.error("Supabase não configurado.")
                else:
                    try:
                        with st.spinner("Importando..."):
                            df = pd.read_csv(forms_csv)
                            imported = []
                            for _, row in df.iterrows():
                                colname = next((c for c in df.columns if 'nome' in c.lower()), None)
                                colemail = next((c for c in df.columns if 'e-mail' in c.lower() or 'email' in c.lower()), None)
                                colcpf = next((c for c in df.columns if 'cpf' in c.lower()), None)
                                name = str(row[colname]) if colname and pd.notna(row[colname]) else None
                                email = str(row[colemail]) if colemail and pd.notna(row[colemail]) else None
                                cpf = str(row[colcpf]) if colcpf and pd.notna(row[colcpf]) else None
                                existing = find_pat
