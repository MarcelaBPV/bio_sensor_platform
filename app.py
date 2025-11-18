# app.py
# -*- coding: utf-8 -*-
"""
Bio Sensor App - Streamlit
Três abas:
1) Pacientes & Import Forms (inclui import ZIP em lotes)
2) Espectrometria Raman (upload individual e upload de até 10 amostras de uma vez)
3) Otimização (IA)
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
# UI: abas
# ---------------------------
tab_pat, tab_raman, tab_ai = st.tabs(["1️⃣ Pacientes & Import Forms", "2️⃣ Espectrometria Raman", "3️⃣ Otimização (IA)"])

# ---------------------------
# Aba 1: Pacientes & Import Forms (mantém os recursos de before)
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
                            # reusa função importadora simples (adaptar se preferir)
                            imported = []
                            for _, row in df.iterrows():
                                # heurística de mapeamento como antes
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
# Aba 2: Espectrometria Raman (com batch upload até 10 arquivos)
# ---------------------------
with tab_raman:
    st.header("2️⃣ Espectrometria Raman — processamento e anotação")
    if process_raman_pipeline is None:
        st.error("Módulo raman_processing_v2.py não encontrado ou com erro. Coloque no mesmo diretório.")
    patients = get_patients_list(200) if supabase else []
    patient_map = {f\"{p['id']} - {p['full_name']}\": p['id'] for p in patients} if patients else {}

    st.subheader("Escolha paciente / amostra")
    col_pa, col_pb = st.columns([1, 2])
    with col_pa:
        if patient_map:
            sel_patient_label = st.selectbox("Paciente", list(patient_map.keys()))
            sel_patient_id = patient_map[sel_patient_label]
            # listar amostras do paciente
            patient_samples = get_samples_for_patient(sel_patient_id) if supabase else []
            samp_map = {f\"{s['id']} - {s['sample_name']}\": s['id'] for s in patient_samples} if patient_samples else {}
            sel_sample_label = st.selectbox("Amostra (opcional, para salvar)", [""] + list(samp_map.keys()))
            sel_sample_id = samp_map[sel_sample_label] if sel_sample_label else None
        else:
            st.info("Nenhum paciente encontrado — importe formulário ou cadastre um paciente.")
            sel_patient_id = None
            sel_sample_id = None

    with col_pb:
        st.subheader("Upload do espectro / substrato")
        st.markdown("Você pode enviar **até 10 espectros** de uma só vez (cada arquivo será processado individualmente usando o substrato carregado).")
        uploaded_sample_single = st.file_uploader("Espectro da amostra (individual) - txt/csv", type=["txt","csv"], key="up_single")
        uploaded_substrate = st.file_uploader("Espectro do substrato / branco (obrigatório para processar lote)", type=["txt","csv"], key="up_substrate")
        st.markdown("OU")
        batch_files = st.file_uploader("Upload em lote (até 10 arquivos) — selecione múltiplos", type=["txt","csv"], accept_multiple_files=True, key="batch_up")

    st.markdown("---")
    st.subheader("Parâmetros do pipeline")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        resample_points = st.number_input("Pontos reamostragem", value=3000, min_value=200, max_value=10000, step=100)
        sg_window = st.number_input("SG window (ímpar)", value=21, min_value=3, step=2)
    with c2:
        sg_poly = st.number_input("SG polyorder", value=2, min_value=1, max_value=5)
        asls_lambda = st.number_input("ASLS lambda", value=1e5, format="%.0f")
    with c3:
        asls_p = st.number_input("ASLS p", value=0.01, format="%.4f")
        prominence = st.number_input("Prominence (picos)", value=0.02, format="%.4f")

    # helper to parse uploaded into StringIO
    def buf_from_file(f):
        try:
            return io.StringIO(f.getvalue().decode("utf-8"))
        except Exception:
            try:
                return io.StringIO(f.getvalue().decode("latin-1"))
            except Exception:
                return None

    # -----------------
    # Processar arquivo único (interface antiga)
    # -----------------
    if uploaded_sample_single and uploaded_substrate:
        try:
            sample_buf = buf_from_file(uploaded_sample_single)
            substrate_buf = buf_from_file(uploaded_substrate)
            with st.spinner("Processando..."):
                (x, y), peaks_df, fig = process_raman_pipeline(
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
            st.pyplot(fig)
            peaks_df = annotate_molecular_groups(peaks_df)
            st.subheader("Picos detectados e grupos moleculares")
            st.dataframe(peaks_df)

            coldl, coldr = st.columns(2)
            with coldl:
                df_spec = pd.DataFrame({"wavenumber_cm1": x, "intensity_a": y})
                st.download_button("⬇️ Baixar espectro corrigido (CSV)", df_spec.to_csv(index=False).encode("utf-8"), file_name="spectrum_corrected.csv", mime="text/csv")
                st.download_button("⬇️ Baixar picos (CSV)", peaks_df.to_csv(index=False).encode("utf-8"), file_name="raman_peaks.csv", mime="text/csv")
            with coldr:
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

    # -----------------
    # Processar lote (até 10 arquivos)
    # -----------------
    if batch_files:
        if len(batch_files) > 10:
            st.warning("Você enviou mais de 10 arquivos — por favor selecione até 10 por vez.")
        elif uploaded_substrate is None:
            st.warning("Carregue primeiro o espectro do substrato (branco) para processar o lote.")
        else:
            if st.button("🚀 Processar lote (até 10)"):
                substrate_buf = buf_from_file(uploaded_substrate)
                total = len(batch_files)
                progress = st.progress(0)
                results = []
                for i, f in enumerate(batch_files):
                    try:
                        sample_buf = buf_from_file(f)
                        if sample_buf is None:
                            raise ValueError("Não foi possível decodificar o arquivo")
                        with st.spinner(f"Processando {f.name} ({i+1}/{total})"):
                            (x, y), peaks_df, fig = process_raman_pipeline(
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
                        st.subheader(f"Resultado — {f.name}")
                        st.pyplot(fig)
                        peaks_df = annotate_molecular_groups(peaks_df)
                        st.dataframe(peaks_df)

                        # oferecer download
                        df_spec = pd.DataFrame({"wavenumber_cm1": x, "intensity_a": y})
                        st.download_button(f"⬇️ Baixar espectro corrigido ({f.name})", df_spec.to_csv(index=False).encode("utf-8"), file_name=f"{f.name}_corrected.csv", mime="text/csv")
                        st.download_button(f"⬇️ Baixar picos ({f.name})", peaks_df.to_csv(index=False).encode("utf-8"), file_name=f"{f.name}_peaks.csv", mime="text/csv")

                        # salvar no supabase (opcional) — cria sample automaticamente se necessário
                        if st.checkbox(f"Salvar {f.name} no Supabase", key=f"save_{i}"):
                            if not supabase:
                                st.error("Supabase não configurado — não é possível salvar.")
                            else:
                                try:
                                    if sel_sample_id is None:
                                        if sel_patient_id is None:
                                            st.error("Selecione um paciente antes de salvar.")
                                        else:
                                            sample_obj = {
                                                "patient_id": sel_patient_id,
                                                "sample_name": f"{sel_patient_id}_{f.name}",
                                                "description": "Criada automaticamente a partir do upload em lote",
                                                "collection_date": None,
                                                "metadata": {"source_file": f.name},
                                                "substrate": "paper_ag_blood"
                                            }
                                            sample_record = create_sample_record(sample_obj)
                                            sample_id_to_use = sample_record["id"]
                                    else:
                                        sample_id_to_use = sel_sample_id

                                    meas_id = create_measurement_record(sample_id_to_use, "raman", operator=None, notes="Lote 10 upload")
                                    df_to_save = pd.DataFrame({"wavenumber_cm1": x, "intensity_a": y})
                                    insert_raman_spectrum_df(df_to_save, meas_id)
                                    insert_peaks_df(peaks_df, meas_id)
                                    st.success(f"✅ {f.name} salvo. measurement_id={meas_id}")
                                    results.append({"file": f.name, "measurement_id": meas_id})
                                except Exception as e:
                                    st.error(f"Erro ao salvar {f.name}: {e}")
                    except Exception as e:
                        st.error(f"Erro processando {f.name}: {e}")
                    progress.progress(int((i+1)/total * 100)/100.0)
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
