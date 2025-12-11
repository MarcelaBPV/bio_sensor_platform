# raman_processing.py
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


# ----------------------------------------------------------------------
# 1) Mapa de grupos moleculares (exemplo – você pode expandir)
# ----------------------------------------------------------------------

MOLECULAR_MAP: List[Dict[str, Any]] = [
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

DISEASE_RULES: List[Dict[str, Any]] = [
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
    group: Optional[str] = None


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
# 4.1) Remoção simples de spikes (opcional)
# ----------------------------------------------------------------------

def despike_median(y: np.ndarray, kernel_size: int = 5, z_thresh: float = 6.0) -> np.ndarray:
    """
    Remoção simples de spikes:
    - Aplica um filtro mediano 1D
    - Substitui pontos muito discrepantes (z-score local) pelo valor filtrado
    """
    from scipy.signal import medfilt

    y = y.astype(float).copy()
    y_med = medfilt(y, kernel_size=kernel_size)

    resid = y - y_med
    sigma = np.std(resid) + 1e-12
    z = np.abs(resid) / sigma

    y[z > z_thresh] = y_med[z > z_thresh]
    return y


# ----------------------------------------------------------------------
# 4.2) Estimativa de linha de base por ALS (Eilers) – versão corrigida
# ----------------------------------------------------------------------

def baseline_als(
    y: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    niter: int = 10,
) -> np.ndarray:
    """
    Cálculo de baseline usando o método Asymmetric Least Squares (ALS)
    de Eilers e Boelens.

    Garante que as matrizes W e H tenham shape (L x L),
    evitando erros de broadcasting.
    """
    # Garante array 1D numpy
    y = np.asarray(y, dtype=float).ravel()
    L = y.size

    # Matriz de segunda derivada discreta: (L-2 x L)
    D = np.diff(np.eye(L), 2)
    # H terá shape (L x L)
    H = lam * D.T.dot(D)

    w = np.ones(L, dtype=float)

    for _ in range(niter):
        # W: matriz diagonal de pesos (L x L)
        W = np.diag(w)
        # Z tem shape (L x L)
        Z = W + H
        # Resolve (W + H) z = W y
        z = np.linalg.solve(Z, w * y)
        # Atualiza pesos (assimetria)
        w = p * (y > z) + (1 - p) * (y < z)

    return z


# ----------------------------------------------------------------------
# 5) Pré-processamento: despike + baseline + suavização + normalização
# ----------------------------------------------------------------------

def preprocess_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    use_despike: bool = True,
    smooth: bool = True,
    window_length: int = 9,
    polyorder: int = 3,
    baseline_method: Optional[str] = "als",
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pré-processamento harmonizado:
    - (opcional) Remoção de spikes
    - (opcional) Estimativa e subtração de linha de base
    - (opcional) Suavização (Savitzky-Golay)
    - (opcional) Normalização 0–1

    Retorna (x, y_proc), onde y_proc já está corrigido de baseline, suavizado
    e normalizado conforme os parâmetros.
    """
    y_proc = y.astype(float).copy()

    # 1) Despike (remover spikes pontuais)
    if use_despike:
        y_proc = despike_median(y_proc, kernel_size=5, z_thresh=6.0)

    # 2) Linha de base
    if baseline_method == "als":
        base = baseline_als(y_proc, lam=1e5, p=0.01, niter=10)
        y_proc = y_proc - base
    elif baseline_method is None:
        base = np.zeros_like(y_proc)
    else:
        raise ValueError(f"baseline_method desconhecido: {baseline_method}")

    # 3) Suavização
    if smooth and len(y_proc) >= 7:
        if window_length >= len(y_proc):
            window_length = len(y_proc) - 1
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(window_length, 3)
        if window_length <= len(y_proc):
            y_proc = savgol_filter(y_proc, window_length=window_length, polyorder=polyorder)

    # 4) Normalização
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


# ----------------------------------------------------------------------
# 9) Função de alto nível: do arquivo até grupos moleculares
# ----------------------------------------------------------------------

def process_raman_spectrum_with_groups(
    file,
    preprocess_kwargs: Optional[Dict[str, Any]] = None,
    peak_height: float = 0.1,
    peak_distance: int = 5,
    peak_prominence: float = 0.02,
) -> Dict[str, Any]:
    """
    Fluxo completo para um espectro:

    - Leitura do espectro
    - Pré-processamento (despike, baseline, suavização, normalização)
    - Detecção de picos
    - Mapeamento de picos para grupos moleculares
    - Inferência de padrões (disease rules)

    Retorna um dict com:
    - x_raw, y_raw
    - x_proc, y_proc (já com baseline removida, suavizado e normalizado)
    - peaks (lista de Peak, incluindo group)
    - diseases (lista de DiseaseMatch)
    """
    if preprocess_kwargs is None:
        preprocess_kwargs = {}

    # 1) Leitura
    x_raw, y_raw = load_spectrum(file)

    # 2) Pré-processamento harmonizado
    x_proc, y_proc = preprocess_spectrum(x_raw, y_raw, **preprocess_kwargs)

    # 3) Picos
    peaks = detect_peaks(
        x_proc,
        y_proc,
        height=peak_height,
        distance=peak_distance,
        prominence=peak_prominence,
    )

    # 4) Mapeamento químico
    peaks = map_peaks_to_molecular_groups(peaks)

    # 5) Regras de "doença"
    diseases = infer_diseases(peaks)

    return {
        "x_raw": x_raw,
        "y_raw": y_raw,
        "x_proc": x_proc,
        "y_proc": y_proc,
        "peaks": peaks,
        "diseases": diseases,
    }


# ----------------------------------------------------------------------
# 10) Pequeno teste local (opcional)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import io
    import matplotlib.pyplot as plt

    # Exemplo: gera um espectro sintético simples
    x = np.linspace(700, 1800, 1101)
    y = (
        0.3 * np.exp(-(x - 720) ** 2 / (2 * 5 ** 2)) +   # pico ~ hemoglobina
        0.8 * np.exp(-(x - 1455) ** 2 / (2 * 10 ** 2)) +  # pico ~ lipídios
        0.02 * (x - 1250)                                # leve inclinação de baseline
    )
    y = y + np.random.normal(scale=0.01, size=y.shape)

    buf = io.StringIO()
    df = pd.DataFrame({"x": x, "y": y})
    df.to_csv(buf, index=False, sep="\t")
    buf.seek(0)
    buf.name = "synthetic.txt"

    res = process_raman_spectrum_with_groups(
        buf,
        preprocess_kwargs={
            "use_despike": True,
            "smooth": True,
            "baseline_method": "als",
            "normalize": True,
        },
        peak_height=0.1,
        peak_distance=5,
        peak_prominence=0.02,
    )

    print("Picos encontrados:")
    for p in res["peaks"]:
        print(f"{p.position_cm1:.2f} cm-1 | I={p.intensity:.3f} | group={p.group}")

    print("\nPadrões inferidos:")
    for d in res["diseases"]:
        print(f"{d.name} (score={d.score})")

    # Plot rápido
    plt.figure()
    plt.plot(res["x_raw"], res["y_raw"], label="Bruto")
    plt.plot(res["x_proc"], res["y_proc"], label="Pré-processado")
    plt.xlabel("Deslocamento Raman (cm⁻¹)")
    plt.ylabel("Intensidade (u.a.)")
    plt.legend()
    plt.tight_layout()
    plt.show()
