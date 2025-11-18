# app.py
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
from supabase import create_client, Client

# Pipeline
try:
    from raman_processing_v2 import process_raman_pipeline
except Exception as e:
    process_raman_pipeline = None

# ---------------------------
# Config Streamlit
# ---------------------------
st.set_page_config(page_title="Plataforma Raman — Pacientes & Ensaios", layout="wide", page_icon="🧬")
st.title("🧬 Plataforma Raman — Análise Molecular do Sangue")

# ---------------------------
# Conexão Supabase (st.secrets)
# ---------------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
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

def plot_main_and_residual(x, y, peaks_df, title=None):
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 0.05, 1], hspace=0.35)

    ax_main = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[2, 0])

    # main: plot data (gray) and synthetic fit if available in peaks_df -> assume peaks_df may contain 'fit_total' column
    ax_main.plot(x, y, color='gray', linewidth=0.8, label='Dados')

    # If peaks_df contains components, they will be plotted elsewhere by caller. Here we plot markers for detected peaks
    if peaks_df is not None and not peaks_df.empty:
        # markers
        ax_main.scatter(peaks_df['fit_cen'] if 'fit_cen' in peaks_df.columns else peaks_df['peak_cm1'],
                        peaks_df.get('fit_height', np.ones(len(peaks_df))) * 0.98,
                        marker='x', s=70, linewidths=2, color='blue', label='Picos Detectados')

        # annotate labels (small boxes) near markers
        for idx, r in peaks_df.iterrows():
            cen = r.get('fit_cen', r.get('peak_cm1'))
            lab = f"{float(cen):.1f}"
            ax_main.annotate(lab, xy=(cen, 0.98), xytext=(5, 5), textcoords='offset points', bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7), fontsize=8)

    ax_main.set_ylabel('Intens. Norm.')
    ax_main.set_xlim(min(x), max(x))
    ax_main.grid(True, linestyle='--', linewidth=0.5)
    ax_main.set_title(title if title else '')

    # residual: plot zeros if not provided
    res = np.zeros_like(x)
    ax_res.plot(x, res, color='green')
    ax_res.set_ylabel('Residuo')
    ax_res.set_xlabel('Wave (cm⁻1)')
    ax_res.grid(True, linestyle='--', linewidth=0.5)

    return fig


def plot_groups_panel(peaks_df, title='Grupos Moleculares'):
    # create a separate figure similar to your example: points colored by group, y arranged for readability
    fig, ax = plt.subplots(figsize=(10, 2.0))
    if peaks_df is None or peaks_df.empty:
        ax.text(0.5, 0.5, 'Nenhum pico detectado', ha='center')
        return fig

    # choose y positions by group to separate rows
    unique_groups = peaks_df['molecular_group'].unique().tolist()
    y_map = {g: i for i, g in enumerate(unique_groups[::-1])}  # reverse for nicer ordering

    xs = peaks_df['fit_cen'] if 'fit_cen' in peaks_df.columns else peaks_df['peak_cm1']
    ys = peaks_df['molecular_group'].map(y_map)
    colors = peaks_df['color'] if 'color' in peaks_df.columns else ['tab:gray'] * len(peaks_df)

    ax.scatter(xs, ys, s=80, c=colors, marker='o', edgecolors='k')

    # annotate with label text to the right
    for i, r in peaks_df.iterrows():
        x = r.get('fit_cen', r.get('peak_cm1'))
        y = y_map[r['molecular_group']]
        lab = f"{float(x):.1f}"
        ax.text(x + (max(xs)-min(xs)) * 0.005, y, f" {lab}", va='center', fontsize=8)

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(list(y_map.keys()))
    ax.set_xlabel('Wave (cm⁻1)')
    ax.set_title(title)
    ax.grid(False)
    ax.set_xlim(min(xs) - 10, max(xs) + 10)
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
                                existing = find_patient_by_email_or_cpf(email=email, cpf=cpf)
                                if existing:
                                    patient_record = existing
                                else:
                                    patient_obj = {
                                        "full_name": name or "Desconhecido",
                                        "email": email,
                                        "cpf": cpf,
                                        "created_at": datetime.utcnow().isoformat()
                                    }
                                    patient_record = create_patient_record(patient_obj)
                                sample_obj = {
                                    "patient_id": patient_record["id"],
                                    "sample_name": f"FormResponse_{patient_record['id']}_{int(time.time())}",
                                    "description": "Importado via Google Forms",
                                    "collection_date": None,
                                    "metadata": {str(k): (v if pd.notna(v) else None) for k, v in row.items()},
                                    "substrate": None
                                }
                                create_sample_record(sample_obj)
                                imported.append(patient_record["id"])
                        st.success(f"Importadas {len(imported)} respostas.")
                    except Exception as e:
                        st.error(f"Erro na importação: {e}")

    st.markdown("---")
    st.subheader("Pacientes cadastrados (últimos 200)")
    if supabase:
        try:
            patients = get_patients_list(200)
            st.dataframe(pd.DataFrame(patients))
        except Exception as e:
            st.error(f"Erro listando pacientes: {e}")
    else:
        st.info("Conecte ao Supabase para ver a lista de pacientes.")

# ---------------------------
# Aba 2: Espectrometria Raman
# ---------------------------
with tab_raman:
    st.header("2️⃣ Espectrometria Raman — processamento e anotação")

    if process_raman_pipeline is None:
        st.error("Módulo raman_processing_v2.py não encontrado ou com erro. Coloque no mesmo diretório.")

    patients = get_patients_list(200) if supabase else []
    patient_map = {f"{p['id']} - {p['full_name']}": p["id"] for p in patients} if patients else {}

    st.subheader("Escolha paciente / amostra")
    col_pa, col_pb = st.columns([1, 2])

    with col_pa:
        if patient_map:
            sel_patient_label = st.selectbox("Paciente", list(patient_map.keys()))
            sel_patient_id = patient_map[sel_patient_label]
            if supabase:
                samp_res = supabase.table("samples").select("*").eq("patient_id", sel_patient_id).order("created_at", desc=True).execute()
                patient_samples = samp_res.data or []
            else:
                patient_samples = []
            samp_map = {f"{s['id']} - {s['sample_name']}": s["id"] for s in patient_samples} if patient_samples else {}
            sel_sample_label = st.selectbox("Amostra (opcional, para salvar)", [""] + list(samp_map.keys()))
            sel_sample_id = samp_map[sel_sample_label] if sel_sample_label else None
        else:
            st.info("Nenhum paciente encontrado — importe formulário ou cadastre um paciente.")
            sel_patient_id = None
            sel_sample_id = None

    with col_pb:
        st.subheader("Parâmetros de processamento")
        resample_points = st.number_input("Resample points", min_value=256, max_value=16384, value=2048, step=256)
        sg_window = st.number_input("Savitzky-Golay window", min_value=5, max_value=101, value=11, step=2)
        sg_poly = st.number_input("Savitzky-Golay poly", min_value=1, max_value=5, value=2)
        asls_lambda = st.number_input("ASLS lambda", min_value=1.0, value=1e5, format="%.0f")
        asls_p = st.number_input("ASLS p", min_value=0.0, max_value=1.0, value=0.01, format="%.3f")
        prominence = st.number_input("Peak prominence (fraction)", min_value=1e-6, max_value=10.0, value=0.05, format="%.6f")

    st.markdown("---")
    uploaded_substrate = st.file_uploader("Carregar espectro do substrato (branco)", type=["txt", "csv"], key="substrate")
    uploaded_sample_single = st.file_uploader("Upload único (um espectro)", type=["txt", "csv"], key="single")

    # batch
    st.markdown("### Upload em lote (até 10 arquivos) — criar 1 paciente/amostra por arquivo")
    batch_files = st.file_uploader("Selecione até 10 arquivos (.txt, .csv) — um arquivo por paciente", type=["txt", "csv"], accept_multiple_files=True, help="Cada arquivo será tratado como uma amostra de um paciente distinto.")
    create_patient_per_file = st.checkbox("Criar paciente novo para cada arquivo (nome baseado no filename)", value=True)
    batch_process_btn = st.button("🚀 Processar e (opcional) salvar lote como novos pacientes")

    # process single
    if uploaded_sample_single and process_raman_pipeline is not None:
        try:
            sample_buf = BytesIO(uploaded_sample_single.read())
            substrate_buf = BytesIO(uploaded_substrate.read()) if uploaded_substrate else None
            with st.spinner("Processando..."):
                (x, y), peaks_df, fig_fit = process_raman_pipeline(
                    sample_input=sample_buf,
                    substrate_input=substrate_buf,
                    resample_points=int(resample_points),
                    sg_window=int(sg_window),
                    sg_poly=int(sg_poly),
                    asls_lambda=float(asls_lambda),
                    asls_p=float(asls_p),
                    peak_prominence=float(prominence),
                    trim_frac=0.02
                )
            peaks_df = annotate_molecular_groups(peaks_df)

            # main + residual
            fig_main = plot_main_and_residual(x, y, peaks_df, title=uploaded_sample_single.name)
            st.pyplot(fig_main)

            # groups panel and table side-by-side
            gcol1, gcol2 = st.columns([2, 1])
            with gcol1:
                fig_groups = plot_groups_panel(peaks_df)
                st.pyplot(fig_groups)
            with gcol2:
                st.subheader('Tabela de picos e grupos')
                display_df = peaks_df[['fit_cen' if 'fit_cen' in peaks_df.columns else 'peak_cm1', 'fit_height' if 'fit_height' in peaks_df.columns else 'height', 'molecular_group']].copy()
                display_df.columns = ['wavenumber_cm1', 'intensity', 'molecular_group']
                st.dataframe(display_df)

            # downloads
            df_spec = pd.DataFrame({"wavenumber_cm1": x, "intensity_a": y})
            st.download_button("⬇️ Baixar espectro corrigido (CSV)", df_spec.to_csv(index=False).encode("utf-8"), file_name="spectrum_corrected.csv", mime="text/csv")
            st.download_button("⬇️ Baixar picos (CSV)", peaks_df.to_csv(index=False).encode("utf-8"), file_name="raman_peaks.csv", mime="text/csv")

            # save
            if st.button("💾 Salvar espectro e picos no Supabase"):
                if not supabase:
                    st.error("Supabase não configurado — não é possível salvar.")
                else:
                    try:
                        if sel_sample_id is None:
                            if sel_patient_id is None:
                                st.error("Selecione um paciente ou importe o Google Forms antes de salvar.")
                            else:
                                sample_obj = {
                                    "patient_id": sel_patient_id,
                                    "sample_name": f"Sample_auto_{int(time.time())}",
                                    "description": "Criada automaticamente a partir do upload do espectro",
                                    "collection_date": None,
                                    "metadata": None,
                                    "substrate": "paper_ag_blood"
                                }
                                sample_record = create_sample_record(sample_obj)
                                sample_id_to_use = sample_record["id"]
                        else:
                            sample_id_to_use = sel_sample_id

                        meas_id = create_measurement_record(sample_id_to_use, "raman", operator=None, notes="Process via app")
                        df_to_save = pd.DataFrame({"wavenumber_cm1": x, "intensity_a": y})
                        insert_raman_spectrum_df(df_to_save, meas_id)
                        insert_peaks_df(peaks_df, meas_id)
                        st.success(f"✅ Dados salvos. measurement_id = {meas_id}")
                    except Exception as e:
                        st.error(f"Erro ao salvar: {e}")

        except Exception as e:
            st.error(f"Erro no processamento: {e}")

    # batch processing block (similar to single but loops)
    if batch_files and batch_process_btn:
        if len(batch_files) > 10:
            st.warning("Você enviou mais de 10 arquivos — por favor selecione até 10 por vez.")
        elif process_raman_pipeline is None:
            st.error("Módulo raman_processing_v2.py não encontrado — não é possível processar.")
        else:
            substrate_bytes = BytesIO(uploaded_substrate.read()) if uploaded_substrate else None
            total = len(batch_files)
            progress = st.progress(0)
            results = []
            for i, f in enumerate(batch_files):
                try:
                    file_bytes = f.read()
                    sample_input = BytesIO(file_bytes)
                    substrate_input = substrate_bytes
                    with st.spinner(f"Processando {f.name} ({i+1}/{total})"):
                        (x, y), peaks_df, fig_fit = process_raman_pipeline(
                            sample_input=sample_input,
                            substrate_input=substrate_input,
                            resample_points=int(resample_points),
                            sg_window=int(sg_window),
                            sg_poly=int(sg_poly),
                            asls_lambda=float(asls_lambda),
                            asls_p=float(asls_p),
                            peak_prominence=float(prominence),
                            trim_frac=0.02
                        )
                    peaks_df = annotate_molecular_groups(peaks_df)

                    # show main plot
                    st.subheader(f"Resultado — {f.name}")
                    fig_main = plot_main_and_residual(x, y, peaks_df, title=f.name)
                    st.pyplot(fig_main)

                    # groups and table
                    gcol1, gcol2 = st.columns([2, 1])
                    with gcol1:
                        fig_groups = plot_groups_panel(peaks_df)
                        st.pyplot(fig_groups)
                    with gcol2:
                        st.subheader('Tabela de picos e grupos')
                        display_df = peaks_df[['fit_cen' if 'fit_cen' in peaks_df.columns else 'peak_cm1', 'fit_height' if 'fit_height' in peaks_df.columns else 'height', 'molecular_group']].copy()
                        display_df.columns = ['wavenumber_cm1', 'intensity', 'molecular_group']
                        st.dataframe(display_df)

                    # downloads
                    df_spec = pd.DataFrame({"wavenumber_cm1": x, "intensity_a": y})
                    st.download_button(f"⬇️ Baixar espectro corrigido ({f.name})", df_spec.to_csv(index=False).encode("utf-8"), file_name=f"{f.name}_corrected.csv", mime="text/csv")
                    st.download_button(f"⬇️ Baixar picos ({f.name})", peaks_df.to_csv(index=False).encode("utf-8"), file_name=f"{f.name}_peaks.csv", mime="text/csv")

                    # optional save per file
                    if supabase and st.checkbox(f"Salvar {f.name} no Supabase (criar paciente/amostra)", key=f"save_{i}"):
                        try:
                            if create_patient_per_file:
                                suggested_name = f.name.split()[0]
                                patient_obj = {"full_name": suggested_name, "email": None, "cpf": None, "created_at": datetime.utcnow().isoformat()}
                                patient_rec = create_patient_record(patient_obj)
                                patient_id = patient_rec["id"]
                            else:
                                patient_id = sel_patient_id if 'sel_patient_id' in globals() and sel_patient_id else None
                                if patient_id is None:
                                    st.error("Nenhum paciente selecionado para associar — selecione um paciente ou marque 'Criar paciente novo'.")
                                    raise RuntimeError("Paciente não especificado")

                            sample_obj = {"patient_id": patient_id, "sample_name": f"{patient_id}_{f.name}", "description": "Upload em lote — autom. criado", "collection_date": None, "metadata": {"source_file": f.name}, "substrate": "paper_ag_blood"}
                            sample_rec = create_sample_record(sample_obj)
                            sample_id_to_use = sample_rec["id"]

                            meas_id = create_measurement_record(sample_id_to_use, "raman", operator=None, notes="Lote 10 upload via app")
                            insert_raman_spectrum_df(pd.DataFrame({"wavenumber_cm1": x, "intensity_a": y}), meas_id)
                            insert_peaks_df(peaks_df, meas_id)
                            st.success(f"✅ {f.name} salvo. measurement_id = {meas_id}")
                            results.append({"file": f.name, "measurement_id": meas_id})
                        except Exception as e:
                            st.error(f"Erro ao salvar {f.name}: {e}")

                except Exception as e:
                    st.error(f"Erro processando {f.name}: {e}")
                progress.progress(int(((i+1)/total) * 100))

            if results:
                st.success(f"{len(results)} arquivos salvos no Supabase.")
                st.dataframe(pd.DataFrame(results))

    st.markdown("---")
    st.subheader("Ensaios cadastrados (amostra selecionada)")
    if sel_sample_id and supabase:
        try:
            df_meas = pd.DataFrame(supabase.table("measurements").select("*").eq("sample_id", sel_sample_id).order("created_at", desc=True).execute().data)
            st.dataframe(df_meas)
        except Exception as e:
            st.error(f"Erro ao listar medições: {e}")
    else:
        st.info("Selecione uma amostra para ver medições associadas.")

# ---------------------------
# Aba 3: Otimização (IA)
# ---------------------------
with tab_ai:
    st.header("3️⃣ Otimização (IA) — identificar possíveis doenças por picos")
    file_train = st.file_uploader("CSV de treino (contendo 'label')", type=["csv"], key="train_csv")
    if file_train:
        try:
            df_train = pd.read_csv(file_train)
            if "label" not in df_train.columns:
                st.error("Arquivo de treino deve conter coluna 'label'.")
            else:
                X = df_train.drop(columns=["label"])
                y = df_train["label"]
                scaler = StandardScaler().fit(X)
                Xs = scaler.transform(X)
                X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier(n_estimators=200, random_state=42)
                with st.spinner("Treinando modelo..."):
                    model.fit(X_train, y_train)
                acc = model.score(X_test, y_test)
                st.success(f"Modelo treinado — acurácia: {acc:.2%}")

                importances = pd.DataFrame({
                    "feature": X.columns,
                    "importance": model.feature_importances_
                }).sort_values("importance", ascending=False).reset_index(drop=True)
                st.subheader("Importância das features (top 20)")
                st.table(importances.head(20))

                st.markdown("### Fazer previsões em novo arquivo")
                file_pred = st.file_uploader("CSV para prever (mesmo formato sem coluna label)", type=["csv"], key="pred_csv")
                if file_pred:
                    df_pred = pd.read_csv(file_pred)
                    try:
                        preds = model.predict(scaler.transform(df_pred))
                        st.dataframe(pd.DataFrame({"prediction": preds}))
                    except Exception as e:
                        st.error(f"Erro ao prever: {e}")
        except Exception as e:
            st.error(f"Erro no treino: {e}")

# ---------------------------
# Footer: notas de segurança e propriedade intelectual
# ---------------------------
st.markdown("---")
st.caption("""
© 2025 Marcela Veiga — Todos os direitos reservados.  
Bio Sensor App — Plataforma Integrada para Análise Molecular via Espectroscopia Raman e Supabase.  
Desenvolvido com fins de pesquisa científica e validação experimental.  
O uso, cópia ou redistribuição deste código é proibido sem autorização expressa da autora.
""")
