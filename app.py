# app.py
# -*- coding: utf-8 -*-
"""
Plataforma Raman — Final
3 abas:
1) Pacientes (cadastro & listagem)
2) Espectrometria Raman (import Google Forms -> sample metadata, upload espectros, processamento, salvar)
3) Otimização (IA) - treinar modelo para identificar possíveis doenças a partir dos espectros
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import time
from typing import Optional, List, Dict
from datetime import datetime
import matplotlib.pyplot as plt

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Supabase client
from supabase import create_client, Client

# Pipeline de Raman (coloque raman_processing.py no mesmo diretório)
try:
    from raman_processing import process_raman_pipeline
except Exception:
    process_raman_pipeline = None
    # We'll surface the error in UI

# ---------------------------
# Config Streamlit
# ---------------------------
st.set_page_config(page_title="Plataforma Raman — Pacientes & Ensaios", layout="wide", page_icon="🧬")
st.title("🧬 Plataforma Raman — Análise Molecular do Sangue")

# ============================================
# 🔌 Conexão com Supabase + Diagnóstico
# ============================================
def init_supabase():
    """Cria cliente Supabase e testa conexão."""
    try:
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
        SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    except Exception as e:
        st.sidebar.error(f"❌ Falta variável em st.secrets: {e}")
        return None

    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        # teste rápido: buscar primeira linha da tabela patients (não falha se tabela vazia)
        try:
            res = client.table("patients").select("id").limit(1).execute()
            # dependendo da versão do client, res pode ter .status_code ou apenas .data
            if hasattr(res, "status_code"):
                ok = res.status_code in (200, 201)
            else:
                ok = True
            if ok:
                st.sidebar.success("✅ Supabase conectado com sucesso!")
            else:
                st.sidebar.warning("⚠️ Supabase respondeu, mas sem status esperado.")
        except Exception as e:
            # Pode falhar se a tabela não existir — ainda assim retornamos o cliente
            st.sidebar.warning(f"⚠️ Não foi possível consultar tabela 'patients': {e}")
        return client
    except Exception as e:
        st.sidebar.error(f"❌ Erro ao conectar Supabase: {e}")
        return None

# inicializa o cliente global
supabase: Optional[Client] = init_supabase()
# Não interrompemos rigidamente: permitimos executar a UI mesmo sem supabase, mas operações de save avisarão.
# if not supabase:
#     st.stop()

# ---------------------------
# Utilidades Supabase (seguras)
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
        if hasattr(res, "error") and res.error:
            raise RuntimeError(f"Erro inserindo em {table}: {res.error.message if hasattr(res.error,'message') else res.error}")
        out.extend(res.data or [])
    return out

def create_patient_record(patient_obj: Dict) -> Dict:
    if not supabase:
        raise RuntimeError("Supabase não configurado.")
    res = supabase.table("patients").insert(patient_obj).execute()
    if hasattr(res, "error") and res.error:
        raise RuntimeError(res.error.message)
    return res.data[0]

def find_patient_by_email_or_cpf(email: Optional[str], cpf: Optional[str]) -> Optional[Dict]:
    # tenta encontrar paciente existente para evitar duplicatas
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
    if hasattr(res, "error") and res.error:
        raise RuntimeError(res.error.message)
    return res.data[0]

def create_measurement_record(sample_id: int, ensaio_type: str, operator: Optional[str]=None, notes: Optional[str]=None) -> int:
    if not supabase:
        raise RuntimeError("Supabase não configurado.")
    rec = {"sample_id": sample_id, "type": ensaio_type, "operator": operator, "notes": notes}
    res = supabase.table("measurements").insert(rec).execute()
    if hasattr(res, "error") and res.error:
        raise RuntimeError(res.error.message)
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

def get_samples_list(limit: int = 500) -> List[Dict]:
    if not supabase:
        return []
    r = supabase.table("samples").select("*").order("created_at", desc=True).limit(limit).execute()
    return r.data or []

# ---------------------------
# Mapeamento molecular (simples)
# ranges (cm^-1) -> annotations
# ajustar conforme literatura
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
    return peaks_df

# ---------------------------
# Importador Google Forms CSV -> cria paciente + sample + salva metadata (JSON)
# - tenta evitar duplicatas por email/CPF
# ---------------------------
def import_google_forms_csv(file_buf) -> List[Dict]:
    """
    Recebe arquivo CSV (uploaded do Google Forms) e cria pacientes + amostras no Supabase.
    Retorna lista de {"patient":..., "sample":...}
    """
    df = pd.read_csv(file_buf)
    results = []
    for _, row in df.iterrows():
        # monta metadata com TODOs from form row
        metadata = {str(k): (v if pd.notna(v) else None) for k, v in row.items()}
        # tenta mapear nome/email/cpf
        # heurística: acha coluna com 'nome', 'e-mail' ou 'email', 'cpf'
        colname = next((c for c in df.columns if 'nome' in c.lower()), None)
        colemail = next((c for c in df.columns if 'e-mail' in c.lower() or 'email' in c.lower()), None)
        colcpf = next((c for c in df.columns if 'cpf' in c.lower()), None)

        name = str(row[colname]) if colname and pd.notna(row[colname]) else None
        email = str(row[colemail]) if colemail and pd.notna(row[colemail]) else None
        cpf = str(row[colcpf]) if colcpf and pd.notna(row[colcpf]) else None

        # procura paciente existente
        existing = find_patient_by_email_or_cpf(email=email, cpf=cpf)
        if existing:
            patient_record = existing
        else:
            # cria paciente novo (campos mínimos)
            patient_obj = {
                "full_name": name or "Desconhecido",
                "email": email,
                "cpf": cpf,
                "created_at": datetime.utcnow().isoformat()
            }
            patient_record = create_patient_record(patient_obj)

        # cria amostra vinculada e salva metadata inteira
        sample_obj = {
            "patient_id": patient_record["id"],
            "sample_name": f"FormResponse_{patient_record['id']}_{int(time.time())}",
            "description": "Importado via Google Forms",
            "collection_date": None,
            "metadata": metadata,
            "substrate": metadata.get("substrato") or metadata.get("substrate") or None
        }
        sample_record = create_sample_record(sample_obj)
        results.append({"patient": patient_record, "sample": sample_record})
    return results

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
                            res = import_google_forms_csv(forms_csv)
                        st.success(f"Importadas {len(res)} respostas. Amostras criadas: {len(res)}")
                        st.write(pd.DataFrame([{
                            "patient_id": r["patient"]["id"],
                            "patient_name": r["patient"].get("full_name"),
                            "sample_id": r["sample"]["id"],
                            "sample_name": r["sample"].get("sample_name")
                        } for r in res]))
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
        st.error("Módulo raman_processing.py não encontrado ou com erro. Coloque no mesmo diretório.")
    # selecionar paciente e amostra
    patients = get_patients_list(200) if supabase else []
    patient_map = {f"{p['id']} - {p['full_name']}": p["id"] for p in patients} if patients else {}
    st.subheader("Escolha paciente / amostra")
    col_pa, col_pb = st.columns([1, 2])
    with col_pa:
        if patient_map:
            sel_patient_label = st.selectbox("Paciente", list(patient_map.keys()))
            sel_patient_id = patient_map[sel_patient_label]
            # listar amostras do paciente (busca simples no supabase)
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
        st.subheader("Upload dos espectros")
        uploaded_sample = st.file_uploader("Espectro da amostra (txt/csv)", type=["txt","csv"])
        uploaded_substrate = st.file_uploader("Espectro do substrato / branco (txt/csv) — obrigatório", type=["txt","csv"])
        st.markdown("Se já salvou o branco como amostra, pode baixar o CSV do storage e enviar aqui.")

    st.markdown("---")
    st.subheader("Parâmetros do pipeline")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        resample_points = st.number_input("Pontos reamostragem", value=3000, min_value=200, max_value=10000, step=100)
        sg_window = st.number_input("SG window (ímpar)", value=11, min_value=3, step=2)
    with c2:
        sg_poly = st.number_input("SG polyorder", value=3, min_value=1, max_value=5)
        asls_lambda = st.number_input("ASLS lambda", value=1e5, format="%.0f")
    with c3:
        asls_p = st.number_input("ASLS p", value=0.01, format="%.4f")
        prominence = st.number_input("Prominence (picos)", value=0.02, format="%.4f")

    if uploaded_sample and uploaded_substrate:
        try:
            # prepare buffers
            sample_buf = io.StringIO(uploaded_sample.getvalue().decode("utf-8"))
            substrate_buf = io.StringIO(uploaded_substrate.getvalue().decode("utf-8"))
            with st.spinner("Processando (remoção substrato + baseline + identificação de picos)..."):
                (x, y), peaks_df, fig = process_raman_pipeline(
                    sample_input=sample_buf,
                    substrate_input=substrate_buf,
                    resample_points=int(resample_points),
                    sg_window=int(sg_window),
                    sg_poly=int(sg_poly),
                    asls_lambda=float(asls_lambda),
                    asls_p=float(asls_p),
                    peak_prominence=float(prominence),
                    fit_profile='lorentz'
                )
            st.pyplot(fig)
            st.success(f"Picos detectados: {len(peaks_df)}")
            peaks_df = annotate_molecular_groups(peaks_df)
            st.subheader("Picos detectados e grupos moleculares")
            st.dataframe(peaks_df)

            # ações: download, salvar
            cold1, cold2 = st.columns([1,1])
            with cold1:
                df_spec = pd.DataFrame({"wavenumber_cm1": x, "intensity_a": y})
                csv_spec = df_spec.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Baixar espectro corrigido (CSV)", csv_spec, file_name="spectrum_corrected.csv", mime="text/csv")
                csv_peaks = peaks_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Baixar picos (CSV)", csv_peaks, file_name="raman_peaks.csv", mime="text/csv")
            with cold2:
                if st.button("💾 Salvar espectro e picos no Supabase"):
                    if not supabase:
                        st.error("Supabase não configurado — não é possível salvar.")
                    else:
                        try:
                            # se não tiver sample_id selecionado cria uma amostra temporária vinculada ao paciente
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
                            # salvar espectro em batches
                            df_to_save = pd.DataFrame({"wavenumber_cm1": x, "intensity_a": y})
                            insert_raman_spectrum_df(df_to_save, meas_id)
                            # salvar picos
                            insert_peaks_df(peaks_df, meas_id)
                            st.success(f"✅ Dados salvos. measurement_id = {meas_id}")
                        except Exception as e:
                            st.error(f"Erro ao salvar: {e}")

        except Exception as e:
            st.error(f"Erro no processamento: {e}")
    else:
        st.info("Envie o espectro da amostra e o espectro do substrato para processar.")

# ---------------------------
# Aba 3: Otimização (IA)
# ---------------------------
with tab_ai:
    st.header("3️⃣ Otimização (IA) — identificar possíveis doenças por picos")
    st.markdown("Aqui treinamos um modelo de classificação a partir de espectros rotulados. O formato esperado é um CSV com colunas de features (ex: intensidades por wavenumber ou amplitudes de picos) e uma coluna 'label' com a classe (doença/controle).")

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

                # feature importances
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
# Footer: notas de segurança
# ---------------------------
st.markdown("---")
st.caption("⚠️ Atenção: dados de saúde são sensíveis. Configure Row Level Security (RLS) no Supabase, use service accounts apenas em backends seguros e nunca exponha service_role keys no frontend. Garanta conformidade LGPD/GDPR conforme aplicável.")
