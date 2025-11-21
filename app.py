# app.py
# -*- coding: utf-8 -*-
"""
Bio Sensor App - Streamlit
Três abas:
1) Pacientes & Import Forms (inclui import ZIP em lotes)
2) Espectrometria Raman (upload individual e upload de até 10 amostras de uma vez)
3) Otimização (IA)

Layout:
- Gráfico principal (espectro ajustado + picos + resíduos)
- Tabela simples com picos e grupos moleculares (sem painel de grupos)
- Calibração por silício e identificação de picos de substrato (paper / paper+silver)
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
from io import BytesIO

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Supabase (opcional)
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

# Pipeline (seu módulo local)
try:
    from raman_processing import process_raman_pipeline
except Exception:
    process_raman_pipeline = None

# ---------------------------
# Config Streamlit
# ---------------------------
st.set_page_config(page_title="Bio Sensor App", layout="wide", page_icon="*")
st.title("*Bio Sensor App*")

# ---------------------------
# Conexão Supabase (st.secrets)
# ---------------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL") if hasattr(st, "secrets") else None
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") if hasattr(st, "secrets") else None

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

# ---------------------------
# Função de anotação com substrato
# ---------------------------
def annotate_molecular_groups(peaks_df: pd.DataFrame, tolerance: float = 5.0, substrate_type: Optional[str]=None) -> pd.DataFrame:
    """
    Anota grupos moleculares por wavenumber.
    substrate_type: None | 'paper' | 'paper+silver' | 'paper+other'
    """
    PAPER_BANDS = [
        (380, 410, 'Papel - lignina/celulose'),
        (1050, 1110, 'Papel - celulose (C–O)')
    ]
    PAPER_SILVER_BANDS = [
        (1000, 1015, 'SERS: Fenilalanina (enriquecido)'),
        (1325, 1350, 'SERS: modos CH / hemoglobina (alterado)')
    ]

    groups = []
    for _, row in peaks_df.iterrows():
        cen = float(row.get("fit_cen", row.get("peak_cm1", np.nan)))
        match = None
        if substrate_type:
            stype = substrate_type.lower()
            if stype == 'paper':
                for lo, hi, label in PAPER_BANDS:
                    if (lo - tolerance) <= cen <= (hi + tolerance):
                        match = f"Substrato: {label}"
                        break
            elif stype == 'paper+silver':
                for lo, hi, label in PAPER_SILVER_BANDS:
                    if (lo - tolerance) <= cen <= (hi + tolerance):
                        match = f"Substrato: {label}"
                        break
                # fallback para bandas padrão de papel
                if match is None:
                    for lo, hi, label in PAPER_BANDS:
                        if (lo - tolerance) <= cen <= (hi + tolerance):
                            match = f"Substrato: {label}"
                            break
            elif stype == 'paper+other':
                for lo, hi, label in PAPER_BANDS:
                    if (lo - tolerance) <= cen <= (hi + tolerance):
                        match = f"Substrato: {label}"
                        break

        # se não for substrato, mapear molecular normal
        if match is None:
            for (low, high), label in MOLECULAR_MAP.items():
                if (low - tolerance) <= cen <= (high + tolerance):
                    match = label
                    break

        groups.append(match if match else "Desconhecido")

    peaks_df = peaks_df.copy()
    peaks_df["molecular_group"] = groups

    def _color_map(g):
        if isinstance(g, str) and g.startswith('Substrato:'):
            return GROUP_COLORS.get("Desconhecido", "tab:gray")
        return GROUP_COLORS.get(g, "tab:gray")

    peaks_df["color"] = peaks_df["molecular_group"].map(_color_map)
    peaks_df['is_substrate'] = peaks_df['molecular_group'].str.startswith('Substrato:')
    return peaks_df

# ---------------------------
# Plot helpers
# ---------------------------
def plot_main_and_residual(x, y, peaks_df, fig_fit=None, title=None):
    fig = plt.figure(figsize=(14, 10), dpi=100)
    gs = fig.add_gridspec(6, 4, hspace=0.6, wspace=0.2)
    ax_main = fig.add_subplot(gs[0:4, 0:3])
    ax_legend = fig.add_subplot(gs[0:4, 3])
    ax_res = fig.add_subplot(gs[5, 0:3])

    ax_main.plot(x, y, color='0.3', linewidth=0.9, label='Dados')

    # se fig_fit tiver linhas, copia (compatibilidade)
    if fig_fit is not None and hasattr(fig_fit, 'axes'):
        for a in fig_fit.axes:
            for line in a.get_lines():
                try:
                    ax_main.plot(line.get_xdata(), line.get_ydata(),
                                 linestyle=line.get_linestyle(), linewidth=line.get_linewidth(),
                                 label=line.get_label(), color=line.get_color())
                except Exception:
                    pass

    # marcar picos com X azul e caixa amarela
    if peaks_df is not None and not peaks_df.empty:
        cen_col = 'fit_cen' if 'fit_cen' in peaks_df.columns else ('peak_cm1' if 'peak_cm1' in peaks_df.columns else peaks_df.columns[0])
        height_col = 'fit_height' if 'fit_height' in peaks_df.columns else ('height' if 'height' in peaks_df.columns else None)

        xs = peaks_df[cen_col].astype(float)
        if height_col and height_col in peaks_df.columns:
            ys_label = peaks_df[height_col].astype(float)
            ymax = max(y) if len(y) > 0 else 1.0
            ys = np.minimum(ys_label, ymax * 0.99)
        else:
            ys = np.full(len(xs), max(y) * 0.98)

        ax_main.scatter(xs, ys, marker='x', s=120, linewidths=3, color='royalblue', zorder=10)

        for xpt, ypt in zip(xs, ys):
            lab = f"{float(xpt):.1f}"
            dx = (max(x) - min(x)) * 0.005
            dy = (max(y) - min(y)) * 0.02
            bbox = dict(boxstyle='round,pad=0.2', fc='#fff28a', ec='black', lw=0.8)
            ax_main.annotate(lab, xy=(xpt, ypt), xytext=(xpt + dx, ypt + dy), textcoords='data',
                              fontsize=9, bbox=bbox, zorder=11)

    ax_main.set_xlim(min(x), max(x))
    ax_main.set_ylabel('Intens. Norm.', fontsize=14)
    ax_main.tick_params(axis='both', labelsize=11)
    ax_main.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    ax_main.set_title(title if title else '', fontsize=20, pad=14)

    ax_legend.axis('off')
    handles, labels = ax_main.get_legend_handles_labels()
    if not labels:
        handles, labels = [], []
    ax_legend.legend(handles=handles, labels=labels, loc='center left', frameon=True, fontsize=10)

    # residual (placeholder se não houver fit)
    residual = np.zeros_like(y)
    ax_res.plot(x, residual, color='green', linewidth=1.2)
    ax_res.axhline(0, color='black', linestyle='--', linewidth=1.2)
    ax_res.set_ylabel('Residuo', fontsize=12)
    ax_res.set_xlabel('Wave (cm⁻1)', fontsize=13)
    ax_res.tick_params(axis='both', labelsize=11)
    ax_res.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)

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
    st.subheader("Importar respostas do Google Forms (XLSX ou CSV)")
    st.markdown("""
    Faça o download das respostas do formulário (no Google Sheets ou Google Forms)  
    e envie o arquivo **.xlsx** ou **.csv** aqui.
    """)

    forms_file = st.file_uploader(
        "Arquivo de respostas do formulário (.xlsx ou .csv)",
        type=["xlsx", "csv"]
    )

    if forms_file:
        if st.button("Importar arquivo para Supabase"):
            if not supabase:
                st.error("Supabase não configurado.")
            else:
                try:
                    with st.spinner("Importando..."):
                        # Detecta o tipo de arquivo
                        filename = forms_file.name.lower()
                        if filename.endswith(".csv"):
                            df = pd.read_csv(forms_file)
                        else:
                            df = pd.read_excel(forms_file)

                        imported = []

                        for _, row in df.iterrows():
                            # Detectar colunas principais
                            colname = next((c for c in df.columns if 'nome' in c.lower()), None)
                            colemail = next((c for c in df.columns if 'e-mail' in c.lower() or 'email' in c.lower()), None)
                            colcpf = next((c for c in df.columns if 'cpf' in c.lower()), None)
                            col_part = next((c for c in df.columns if 'particip' in c.lower()), None)

                            name = str(row[colname]) if colname and pd.notna(row[colname]) else None
                            email = str(row[colemail]) if colemail and pd.notna(row[colemail]) else None
                            cpf = str(row[colcpf]) if colcpf and pd.notna(row[colcpf]) else None

                            # Código do participante (P1, 1, etc.)
                            participant_code = None
                            if col_part and pd.notna(row[col_part]):
                                participant_raw = str(row[col_part]).strip()
                                # limpa coisas tipo "1.0"
                                if participant_raw.endswith(".0"):
                                    participant_raw = participant_raw[:-2]
                                participant_code = participant_raw

                            # Nome que vai aparecer na lista de pacientes
                            if name and participant_code:
                                full_name_field = f"{participant_code} - {name}"
                            elif name:
                                full_name_field = name
                            elif participant_code:
                                full_name_field = f"Participante {participant_code}"
                            else:
                                full_name_field = "Desconhecido"

                            # Verifica se já existe paciente com mesmo email ou CPF
                            existing = find_patient_by_email_or_cpf(email=email, cpf=cpf)
                            if existing:
                                patient_record = existing
                            else:
                                patient_obj = {
                                    "full_name": full_name_field,
                                    "email": email,
                                    "cpf": cpf,
                                    "created_at": datetime.utcnow().isoformat()
                                }
                                patient_record = create_patient_record(patient_obj)

                            # Nome da amostra já amarrado ao participante (para você usar depois na aba 2)
                            if participant_code:
                                sample_name = f"{participant_code}_Form"
                            else:
                                sample_name = f"FormResponse_{patient_record['id']}_{int(time.time())}"

                            # Salva toda a linha do formulário em metadata + participant_code
                            metadata_dict = {str(k): (v if pd.notna(v) else None) for k, v in row.items()}
                            if participant_code:
                                metadata_dict["participant_code"] = participant_code

                            sample_obj = {
                                "patient_id": patient_record["id"],
                                "sample_name": sample_name,
                                "description": "Importado via formulário (XLSX/CSV)",
                                "collection_date": None,
                                "metadata": metadata_dict,
                                "substrate": None
                            }
                            create_sample_record(sample_obj)
                            imported.append(patient_record["id"])

                    st.success(f"Importadas {len(imported)} respostas.")
                except Exception as e:
                    st.error(f"Erro na importação: {e}")

# ---------------------------
# Aba 2: Espectrometria Raman
# ---------------------------
with tab_raman:
    st.header("2️⃣ Espectrometria Raman — processamento e anotação")

    if process_raman_pipeline is None:
        st.error("Módulo raman_processing.py não encontrado ou com erro. Coloque no mesmo diretório.")

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

        # novos controles: tipo de substrato (detalhado) e calibração de silício
        substrate_type = st.selectbox(
            "Substrato usado",
            options=["Nenhum", "paper", "paper+silver", "paper+other"],
            index=1,
            help="Escolha 'paper' se amostra for papel; 'paper+silver' para papel com camada de Ag (SERS), ou 'paper+other' para outro recobrimento."
        )
        silicon_calib = st.file_uploader(
            "Arquivo de calibração (Silício) opcional",
            type=["txt", "csv"],
            help="Se enviado, será usado para calibrar/deslocar o eixo wavenumber para 520.7 cm^-1."
        )

    st.markdown("---")
    uploaded_substrate = st.file_uploader("Carregar espectro do substrato (branco)", type=["txt", "csv"], key="substrate")
    uploaded_sample_single = st.file_uploader("Upload único (um espectro)", type=["txt", "csv"], key="single")

    # batch
    st.markdown("### Upload em lote (até 10 arquivos) — criar 1 paciente/amostra por arquivo")
    batch_files = st.file_uploader("Selecione até 10 arquivos (.txt, .csv) — um arquivo por paciente", type=["txt", "csv"], accept_multiple_files=True, help="Cada arquivo será tratado como uma amostra de um paciente distinto.")
    create_patient_per_file = st.checkbox("Criar paciente novo para cada arquivo (nome baseado no filename)", value=True)
    batch_process_btn = st.button("Processar lote (até 10 arquivos)")

    # ler bytes do arquivo de calibração uma vez (reutilizável)
    silicon_calib_bytes = None
    if silicon_calib is not None:
        try:
            silicon_calib_bytes = silicon_calib.read()
        except Exception:
            silicon_calib_bytes = None

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

            # --- CALIBRAÇÃO POR SILÍCIO (se fornecida) --- #
            delta = 0.0
            if silicon_calib_bytes is not None:
                try:
                    df_si = pd.read_csv(BytesIO(silicon_calib_bytes), sep=None, engine='python', comment='#', header=None)
                    df_si = df_si.select_dtypes(include=[np.number])
                    xsi = np.asarray(df_si.iloc[:,0], dtype=float)
                    ysi = np.asarray(df_si.iloc[:,1], dtype=float)
                    mask_si = (xsi >= 510) & (xsi <= 530)
                    if mask_si.any():
                        idx_mask = np.where(mask_si)[0]
                        relative_idx = np.argmax(ysi[mask_si])
                        observed = xsi[idx_mask[relative_idx]]
                        delta = 520.7 - observed
                    else:
                        delta = 0.0
                except Exception:
                    delta = 0.0

            # aplicar deslocamento ao eixo x e aos centros de pico
            try:
                if float(delta) != 0.0:
                    x = (np.array(x) + float(delta)).tolist()
                    if 'fit_cen' in peaks_df.columns:
                        peaks_df['fit_cen'] = peaks_df['fit_cen'].astype(float) + float(delta)
                    elif 'peak_cm1' in peaks_df.columns:
                        peaks_df['peak_cm1'] = peaks_df['peak_cm1'].astype(float) + float(delta)
            except Exception:
                pass

            # anotar levando em conta substrato
            peaks_df = annotate_molecular_groups(peaks_df, substrate_type=substrate_type if substrate_type != 'Nenhum' else None)

            # main + residual
            fig_main = plot_main_and_residual(x, y, peaks_df, title=uploaded_sample_single.name)
            st.pyplot(fig_main)

            # tabela simples (sem painel de grupos)
            st.subheader('Tabela de picos e grupos')
            display_df = peaks_df[['fit_cen' if 'fit_cen' in peaks_df.columns else 'peak_cm1',
                                   'fit_height' if 'fit_height' in peaks_df.columns else ('height' if 'height' in peaks_df.columns else None),
                                   'molecular_group']].copy()
            # renomear de forma segura
            display_df.columns = ['wavenumber_cm1', 'intensity', 'molecular_group']
            st.dataframe(display_df)

            # downloads
            df_spec = pd.DataFrame({"wavenumber_cm1": x, "intensity_a": y})
            st.download_button("⬇️ Baixar espectro corrigido (CSV)", df_spec.to_csv(index=False).encode("utf-8"),
                               file_name="spectrum_corrected.csv", mime="text/csv")
            st.download_button("⬇️ Baixar picos (CSV)", peaks_df.to_csv(index=False).encode("utf-8"),
                               file_name="raman_peaks.csv", mime="text/csv")

            # save
            if st.button("Salvar espectro e picos no Supabase"):
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
                                    "substrate": substrate_type
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

    # batch processing block
    if batch_files and batch_process_btn:
        if len(batch_files) > 10:
            st.warning("Você enviou mais de 10 arquivos — por favor selecione até 10 por vez.")
        elif process_raman_pipeline is None:
            st.error("Módulo raman_processing.py não encontrado — não é possível processar.")
        else:
            substrate_bytes = BytesIO(uploaded_substrate.read()) if uploaded_substrate else None
            total = len(batch_files)
            progress = st.progress(0)
            results = []

            # se silicon_calib_bytes for fornecido, já lido no topo da aba
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

                    # --- CALIBRAÇÃO POR SILÍCIO (se fornecida) --- #
                    delta = 0.0
                    if silicon_calib_bytes is not None:
                        try:
                            df_si = pd.read_csv(BytesIO(silicon_calib_bytes), sep=None, engine='python', comment='#', header=None)
                            df_si = df_si.select_dtypes(include=[np.number])
                            xsi = np.asarray(df_si.iloc[:,0], dtype=float)
                            ysi = np.asarray(df_si.iloc[:,1], dtype=float)
                            mask_si = (xsi >= 510) & (xsi <= 530)
                            if mask_si.any():
                                idx_mask = np.where(mask_si)[0]
                                relative_idx = np.argmax(ysi[mask_si])
                                observed = xsi[idx_mask[relative_idx]]
                                delta = 520.7 - observed
                            else:
                                delta = 0.0
                        except Exception:
                            delta = 0.0

                    # aplicar deslocamento ao eixo x e aos centros de pico
                    try:
                        if float(delta) != 0.0:
                            x = (np.array(x) + float(delta)).tolist()
                            if 'fit_cen' in peaks_df.columns:
                                peaks_df['fit_cen'] = peaks_df['fit_cen'].astype(float) + float(delta)
                            elif 'peak_cm1' in peaks_df.columns:
                                peaks_df['peak_cm1'] = peaks_df['peak_cm1'].astype(float) + float(delta)
                    except Exception:
                        pass

                    peaks_df = annotate_molecular_groups(peaks_df, substrate_type=substrate_type if substrate_type != 'Nenhum' else None)

                    # show main plot
                    st.subheader(f"Resultado — {f.name}")
                    fig_main = plot_main_and_residual(x, y, peaks_df, title=f.name)
                    st.pyplot(fig_main)

                    # tabela simples
                    st.subheader('Tabela de picos e grupos')
                    display_df = peaks_df[['fit_cen' if 'fit_cen' in peaks_df.columns else 'peak_cm1',
                                           'fit_height' if 'fit_height' in peaks_df.columns else ('height' if 'height' in peaks_df.columns else None),
                                           'molecular_group']].copy()
                    display_df.columns = ['wavenumber_cm1', 'intensity', 'molecular_group']
                    st.dataframe(display_df)

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

                            sample_obj = {"patient_id": patient_id, "sample_name": f"{patient_id}_{f.name}", "description": "Upload em lote — autom. criado", "collection_date": None, "metadata": {"source_file": f.name}, "substrate": substrate_type}
                            sample_rec = create_sample_record(sample_obj)
                            sample_id_to_use = sample_rec["id"]

                            meas_id = create_measurement_record(sample_id_to_use, "raman", operator=None, notes="Lote upload via app")
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
    st.subheader("Visualizar ensaios Raman já salvos na plataforma")

    if not supabase:
        st.info("Conecte ao Supabase para visualizar os ensaios salvos.")
    else:
        # 1) Selecionar paciente
        patients_vis = get_patients_list(200)
        if not patients_vis:
            st.info("Nenhum paciente cadastrado ainda.")
        else:
            patient_map_vis = {
                f"{p['id']} - {p.get('full_name', 'Sem nome')}": p["id"]
                for p in patients_vis
            }
            sel_patient_label_vis = st.selectbox(
                "Paciente (para visualizar ensaios salvos)",
                list(patient_map_vis.keys()),
                key="vis_patient"
            )
            sel_patient_id_vis = patient_map_vis[sel_patient_label_vis]

            # 2) Selecionar amostra desse paciente
            samples_vis = get_samples_for_patient(sel_patient_id_vis)
            if not samples_vis:
                st.info("Este paciente ainda não possui amostras cadastradas.")
            else:
                sample_map_vis = {
                    f"{s['id']} - {s.get('sample_name', 'Sem nome de amostra')}": s["id"]
                    for s in samples_vis
                }
                sel_sample_label_vis = st.selectbox(
                    "Amostra",
                    list(sample_map_vis.keys()),
                    key="vis_sample"
                )
                sel_sample_id_vis = sample_map_vis[sel_sample_label_vis]

                # 3) Selecionar medição (ensaio) dessa amostra
                try:
                    meas_res_vis = supabase.table("measurements")\
                        .select("*")\
                        .eq("sample_id", sel_sample_id_vis)\
                        .order("created_at", desc=True)\
                        .execute()
                    meas_list_vis = meas_res_vis.data or []
                except Exception as e:
                    meas_list_vis = []
                    st.error(f"Erro ao buscar medições: {e}")

                if not meas_list_vis:
                    st.info("Nenhum ensaio Raman cadastrado para esta amostra.")
                else:
                    meas_map_vis = {}
                    for m in meas_list_vis:
                        tipo = m.get("type", "sem tipo")
                        created = m.get("created_at", "") or ""
                        created_short = created[:19]  # YYYY-MM-DDTHH:MM:SS
                        label = f"{m['id']} - {tipo} - {created_short}"
                        meas_map_vis[label] = m["id"]

                    sel_meas_label_vis = st.selectbox(
                        "Ensaio (measurement)",
                        list(meas_map_vis.keys()),
                        key="vis_meas"
                    )
                    sel_meas_id_vis = meas_map_vis[sel_meas_label_vis]

                    # 4) Buscar espectro e picos no Supabase
                    try:
                        spec_res = supabase.table("raman_spectra")\
                            .select("wavenumber_cm1,intensity_a")\
                            .eq("measurement_id", sel_meas_id_vis)\
                            .order("wavenumber_cm1", desc=False)\
                            .execute()
                        spec_data = spec_res.data or []
                    except Exception as e:
                        spec_data = []
                        st.error(f"Erro ao buscar espectro: {e}")

                    try:
                        peaks_res = supabase.table("raman_peaks")\
                            .select("*")\
                            .eq("measurement_id", sel_meas_id_vis)\
                            .execute()
                        peaks_data = peaks_res.data or []
                    except Exception as e:
                        peaks_data = []
                        st.error(f"Erro ao buscar picos: {e}")

                    if not spec_data:
                        st.warning("Não há espectro salvo para este ensaio.")
                    else:
                        df_spec_vis = pd.DataFrame(spec_data)
                        # garante nomes corretos
                        if "wavenumber_cm1" not in df_spec_vis.columns or "intensity_a" not in df_spec_vis.columns:
                            st.error("Espectro salvo em formato inesperado (faltam colunas wavenumber_cm1/intensity_a).")
                        else:
                            x_vis = df_spec_vis["wavenumber_cm1"].astype(float).values
                            y_vis = df_spec_vis["intensity_a"].astype(float).values

                            if peaks_data:
                                peaks_df_vis = pd.DataFrame(peaks_data)
                            else:
                                peaks_df_vis = pd.DataFrame()

                            # 5) Plotar gráfico + tabela de picos para esse ensaio
                            st.markdown("### 🔍 Visualização do ensaio selecionado")

                            fig_vis = plot_main_and_residual(
                                x_vis,
                                y_vis,
                                peaks_df_vis if not peaks_df_vis.empty else None,
                                title=f"Paciente: {sel_patient_label_vis} | {sel_sample_label_vis} | {sel_meas_label_vis}"
                            )
                            st.pyplot(fig_vis)

                            # 6) Tabela de picos correspondentes à amostra/ensaio escolhido
                            if peaks_df_vis is not None and not peaks_df_vis.empty:
                                st.subheader("Tabela de picos deste ensaio")

                                # detectar colunas de centro e altura
                                cen_col = None
                                if "fit_cen" in peaks_df_vis.columns:
                                    cen_col = "fit_cen"
                                elif "peak_cm1" in peaks_df_vis.columns:
                                    cen_col = "peak_cm1"

                                height_col = None
                                if "fit_height" in peaks_df_vis.columns:
                                    height_col = "fit_height"
                                elif "height" in peaks_df_vis.columns:
                                    height_col = "height"

                                cols_to_show = []
                                if cen_col:
                                    cols_to_show.append(cen_col)
                                if height_col:
                                    cols_to_show.append(height_col)
                                if "molecular_group" in peaks_df_vis.columns:
                                    cols_to_show.append("molecular_group")

                                if cols_to_show:
                                    display_df_vis = peaks_df_vis[cols_to_show].copy()
                                    rename_map = {}
                                    if cen_col:
                                        rename_map[cen_col] = "wavenumber_cm1"
                                    if height_col:
                                        rename_map[height_col] = "intensity"
                                    display_df_vis = display_df_vis.rename(columns=rename_map)
                                    st.dataframe(display_df_vis)
                                else:
                                    st.info("Picos salvos sem colunas padrão (fit_cen/peak_cm1, fit_height/height).")
                            else:
                                st.info("Nenhum pico salvo para este ensaio.")

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
    st.header("3️⃣ Otimização (IA) — associação de picos a possíveis doenças")

    st.markdown("### Upload de tabela de picos (CSV gerado na aba 2)")
    file_peaks = st.file_uploader("CSV contendo colunas 'wavenumber_cm1' ou 'fit_cen'", type=["csv"], key="clinical_csv")

    # mapa clínico simples baseado nas regiões típicas
    CLINICAL_MAP = [
        (735, 770, "Porfirina / Heme (Hb)", "Alterações podem indicar hipóxia, anemia, inflamação, talassemia, alterações do estado redox da hemoglobina."),
        (995, 1015, "Fenilalanina", "Associada a processos inflamatórios sistêmicos, cânceres sólidos, resposta imune ativada."),
        (1120, 1170, "Carotenoides", "Marcador de estresse oxidativo; alterações associadas a câncer, doenças cardiovasculares e inflamação crônica."),
        (1230, 1300, "Amida III", "Mudanças estruturais em proteínas: inflamação, doenças hepáticas, infecções, sepse e câncer."),
        (1320, 1380, "Modos ligados à hemoglobina", "Deslocamentos refletem hipóxia, diabetes, disfunção pulmonar e alterações metabólicas relativas à hemoglobina."),
        (1420, 1470, "Lipídeos (CH2/CH3)", "Alterações são fortes marcadores de diabetes, obesidade, doenças hepáticas e inflamação."),
        (1490, 1590, "Bandas porfirínicas (Hb)", "Indicador de oxigenação/desoxigenação anormal, metemoglobinemia, inflamação e doenças hematológicas."),
        (1590, 1620, "Aromáticos (Tyr/Trp)", "Marcadores de estresse oxidativo, apoptose e processos neoplásicos."),
        (1620, 1690, "Amida I", "Desordem estrutural proteica; associado a sepse, câncer, inflamação intensa e doenças neurodegenerativas."),
        (2800, 3000, "C–H (lipídeos/proteínas)", "Alterações indicam doenças metabólicas, diabetes e estados inflamatórios."),
    ]

    def find_clinical_association(peak):
        for lo, hi, group, disease in CLINICAL_MAP:
            if lo <= peak <= hi:
                return group, disease
        return "Desconhecido", "Nenhuma associação clínica conhecida para esta região."

    if file_peaks:
        try:
            df_peaks = pd.read_csv(file_peaks)
            peak_col = "fit_cen" if "fit_cen" in df_peaks.columns else ("wavenumber_cm1" if "wavenumber_cm1" in df_peaks.columns else None)
            if peak_col is None:
                st.error("Arquivo precisa ter coluna 'fit_cen' ou 'wavenumber_cm1'.")
            else:
                df_peaks["grupo_molecular"], df_peaks["possivel_doenca"] = zip(*df_peaks[peak_col].astype(float).apply(find_clinical_association))
                st.subheader("Associação clínica dos picos detectados")
                st.dataframe(df_peaks[[peak_col, "grupo_molecular", "possivel_doenca"]])

                st.download_button(
                    "⬇️ Baixar tabela com interpretações clínicas",
                    df_peaks.to_csv(index=False).encode("utf-8"),
                    file_name="raman_clinical_association.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Erro ao analisar arquivo: {e}")

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
