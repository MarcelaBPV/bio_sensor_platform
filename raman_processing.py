# raman_core.py
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


# ---- 1) Mapa de grupos moleculares (exemplo – você pode expandir) ----

MOLECULAR_MAP: List[Dict] = [
    # faixa_min, faixa_max, nome do grupo molecular
    {"range": (700, 740), "group": "Hemoglobina / porfirinas"},
    {"range": (995, 1005), "group": "Fenilalanina (anéis aromáticos)"},
    {"range": (1440, 1470), "group": "Lipídios / CH2 deformação"},
    {"range": (1650, 1670), "group": "Amidas / proteínas (C=O)"},
    # ... adicione quantas faixas quiser
]


# ---- 2) Regras super simples de "doenças" (apenas pesquisa!) ----

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


# ---- 3) Classes de dados auxiliares ----

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


# ---- 4) Funções de carregamento de espectro ----

def load_spectrum(file) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lê um arquivo de espectro (CSV ou XLSX) e retorna (x, y) como numpy arrays.
    Espera duas colunas: primeira = deslocamento Raman (cm-1), segunda = intensidade.
    """
    filename = file.name.lower()

    if filename.endswith(".csv") or filename.endswith(".txt"):
        df = pd.read_csv(file)
    elif filename.endswith(".xls") or filename.endswith(".xlsx"):
        df = pd.read_excel(file)
    else:
        raise ValueError("Formato de arquivo não suportado. Use CSV ou Excel.")

    if df.shape[1] < 2:
        raise ValueError("O arquivo deve ter pelo menos 2 colunas (x e y).")

    x = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)

    return x, y


# ---- 5) Pré-processamento: suavização + normalização ----

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
    - Savitzky-Golay (opcional)
    - normalização 0–1 (opcional)
    """
    y_proc = y.astype(float).copy()

    # suavização
    if smooth:
        # window_length precisa ser ímpar e <= len(y)
        if window_length >= len(y_proc):
            window_length = len(y_proc) - 1
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(window_length, 3)
        y_proc = savgol_filter(y_proc, window_length=window_length, polyorder=polyorder)

    # normalização 0–1
    if normalize:
        ymin = np.min(y_proc)
        ymax = np.max(y_proc)
        if ymax > ymin:
            y_proc = (y_proc - ymin) / (ymax - ymin)

    return x, y_proc


# ---- 6) Detecção automática de picos ----

def detect_peaks(
    x: np.ndarray,
    y: np.ndarray,
    height: float = 0.1,
    distance: int = 5,
    prominence: float = 0.02,
) -> List[Peak]:
    """
    Detecta picos usando scipy.signal.find_peaks.
    Retorna uma lista de Peak (posição + intensidade).
    """
    indices, props = find_peaks(
        y,
        height=height,
        distance=distance,
        prominence=prominence,
    )

    peaks: List[Peak] = []
    for idx in indices:
        peaks.append(
            Peak(
                position_cm1=float(x[idx]),
                intensity=float(y[idx]),
                group=None,
            )
        )
    return peaks


# ---- 7) Mapeamento de picos -> grupos moleculares ----

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


# ---- 8) Correlação simples com "doenças" (modo pesquisa!) ----

def infer_diseases(peaks: List[Peak]) -> List[DiseaseMatch]:
    """
    Usa regras simples baseadas em presença de grupos moleculares
    para sugerir possíveis 'padrões associados a doenças'.

    IMPORTANTE: Isso é apenas para pesquisa / triagem experimental.
    NÃO é diagnóstico médico.
    """
    # Conjunto de grupos detectados
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
