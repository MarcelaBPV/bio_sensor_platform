# raman_processing.py
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


# ----------------------------------------------------------------------
# 1) Mapa de grupos moleculares (exemplo – você pode expandir)
# ----------------------------------------------------------------------

MOLECULAR_MAP: List[Dict] = [
    # faixa_min, faixa_max, nome do grupo molecular
    {"range": (700, 740), "group": "Hemoglobina / porfirinas"},
    {"range": (995, 1005), "group": "Fenilalanina (anéis aromáticos)"},
    {"range": (1440, 1470), "group": "Lipídios / CH2 deformação"},
    {"range": (1650, 1670), "group": "Amidas / proteínas (C=O)"},
    # TODO: adicionar mais faixas conforme sua tabela completa
]


# ----------------------------------------------------------------------
# 2) Regras simples de "doenças" (modo pesquisa, não diagnóstico)
# ----------------------------------------------------------------------

DISEASE_RULES: List[Dict] = [
    {
        "name": "Alteração hemoglobina",
        "description": "Padrão compatível com alterações em heme / porfirinas.",
        "groups_required": ["Hemoglobina / porfirinas"],
    },
    {
        "name": "Alteração proteica",
        "description": "Padrão compatível com alterações em proteínas (amida I).",
        "groups_required": ["Amidas / proteínas (C=O)"],
    },
    {
        "name": "Alteração lipídica",
        "description": "Padrão compatível com alterações em lipídios de membrana.",
        "groups_required": ["Lipídios / CH2 deformação"],
    },
]


# ----------------------------------------------------------------------
# 3) Classes de dados auxiliares
# ----------------------------------------------------------------------

@dataclass
class Peak:
    position_cm1: float
    intensity: float
    group: str | None = None


@dataclass
class DiseaseMatch:
    name: str
    score: int
    description: str


# ----------------------------------------------------------------------
# 4) Função de carregamento de espectro (robusta p/ txt, csv, xlsx)
# ----------------------------------------------------------------------

def load_spectrum(file) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lê um arquivo de espectro Raman e retorna (x, y) como numpy arrays.

    Suporta:
    - .txt no formato do equipamento (#Wave  #Intensity, separado por TAB/espaços, com comentários '#')
    - .csv (vírgula ou ponto e vírgula)
    - .xls / .xlsx

    x -> deslocamento Raman (cm-1)
    y -> intensidade (u.a.)
    """
    filename = getattr(file, "name", "").lower()

    # --- TXT: formato típico de equipamento Raman (#Wave  #Intensity) ---
    if filename.endswith(".txt"):
        df = pd.read_csv(
            file,
            sep=r"\s+",       # qualquer espaço/tab
            comment="#",      # ignora linhas começando com '#'
            engine="python",
        )

    # --- CSV: tenta vírgula, se falhar tenta ponto e vírgula ---
    elif filename.endswith(".csv"):
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, sep=";")

    # --- Excel ---
    elif filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file)

    else:
        # fallback genérico: tenta ler como texto separado por espaço
        df = pd.read_csv(
            file,
            sep=r"\s+",
            comment="#",
            engine="python",
        )

    # Remove colunas completamente vazias
    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 2:
        raise ValueError("Não foi possível identificar as colunas de Raman shift e intensidade.")

    # Prioriza colunas numéricas (caso tenha metadata junto)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        x = numeric_df.iloc[:, 0].to_numpy(dtype=float)
        y = numeric_df.iloc[:, 1].to_numpy(dtype=float)
    else:
        # fallback: pega as duas primeiras colunas e tenta converter
        x = df.iloc[:, 0].astype(float).to_numpy()
        y = df.iloc[:, 1].astype(float).to_numpy()

    return x, y


# ----------------------------------------------------------------------
# 5) Pré-processamento: suavização + normalização
# ----------------------------------------------------------------------

def preprocess_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    smooth: bool = True,
    window_length: int = 9,
    polyorder: int = 3,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica um pré-processamento simples:
    - Suavização Savitzky-Golay (opcional)
    - Normalização 0–1 (opcional)
    """
    y_proc = y.astype(float).copy()

    if smooth:
        # window_length precisa ser ímpar e <= len(y)
        if window_length >= len(y_proc):
            window_length = len(y_proc) - 1
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(window_length, 3)

        y_proc = savgol_filter(y_proc, window_length=window_length, polyorder=polyorder)

    if normalize:
        ymin = float(np.min(y_proc))
        ymax = float(np.max(y_proc))
        if ymax > ymin:
            y_proc = (y_proc - ymin) / (ymax - ymin)

    return x, y_proc


# ----------------------------------------------------------------------
# 6) Detecção automática de picos
# ----------------------------------------------------------------------

def detect_peaks(
    x: np.ndarray,
    y: np.ndarray,
    height: float = 0.1,
    distance: int = 5,
    prominence: float = 0.02,
) -> List[Peak]:
    """
    Detecta picos usando scipy.signal.find_peaks.
    Retorna uma lista de Peak (posição_cm1 + intensidade).
    """
    indices, _ = find_peaks(
        y,
        height=height,
        distance=distance,
        prominence=prominence,
    )

    peaks: List[Peak] = [
        Peak(
            position_cm1=float(x[idx]),
            intensity=float(y[idx]),
            group=None,
        )
        for idx in indices
    ]

    return peaks


# ----------------------------------------------------------------------
# 7) Mapeamento de picos -> grupos moleculares
# ----------------------------------------------------------------------

def map_peaks_to_molecular_groups(peaks: List[Peak]) -> List[Peak]:
    """
    Para cada pico, verifica em qual faixa de MOLECULAR_MAP ele cai e
    atribui o nome do grupo molecular.
    """
    for peak in peaks:
        group_found = None
        for item in MOLECULAR_MAP:
            x_min, x_max = item["range"]
            if x_min <= peak.position_cm1 <= x_max:
                group_found = item["group"]
                break
        peak.group = group_found

    return peaks


# ----------------------------------------------------------------------
# 8) Correlação simples com "doenças" (pesquisa!)
# ----------------------------------------------------------------------

def infer_diseases(peaks: List[Peak]) -> List[DiseaseMatch]:
    """
    Usa regras simples baseadas em presença de grupos moleculares
    para sugerir possíveis 'padrões associados a doenças'.

    IMPORTANTE:
    - Apenas para pesquisa / triagem experimental.
    - NÃO é diagnóstico médico.
    """
    groups_present = {p.group for p in peaks if p.group is not None}

    matches: List[DiseaseMatch] = []

    for rule in DISEASE_RULES:
        required = set(rule["groups_required"])
        score = len(required.intersection(groups_present))
        if score > 0:
            matches.append(
                DiseaseMatch(
                    name=rule["name"],
                    score=score,
                    description=rule["description"],
                )
            )

    # Ordena por score (maior primeiro)
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches
