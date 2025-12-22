from typing import Dict, Any, List
from supabase_client import get_supabase

# -------------------------------------------------
# AMOSTRAS
# -------------------------------------------------
def insert_sample(sample_code: str, sample_type: str, metadata: Dict[str, Any]):
    sb = get_supabase()
    res = sb.table("samples").insert({
        "sample_code": sample_code,
        "sample_type": sample_type,
        "metadata": metadata,
    }).execute()
    return res.data[0]["id"]

# -------------------------------------------------
# ESPECTROS RAMAN
# -------------------------------------------------
def insert_spectrum(
    sample_id: str,
    spectrum_type: str,
    wavenumber: list,
    intensity: list,
    preprocessing_params: Dict[str, Any],
):
    sb = get_supabase()
    res = sb.table("raman_spectra").insert({
        "sample_id": sample_id,
        "spectrum_type": spectrum_type,
        "wavenumber": wavenumber,
        "intensity": intensity,
        "preprocessing_params": preprocessing_params,
    }).execute()
    return res.data[0]["id"]

# -------------------------------------------------
# PICOS RAMAN
# -------------------------------------------------
def insert_peaks(spectrum_id: str, peaks: List[Any]):
    sb = get_supabase()
    rows = [{
        "spectrum_id": spectrum_id,
        "position_cm1": p.position_cm1,
        "intensity": p.intensity,
        "fwhm": p.width,
        "fit_model": p.fit_params["model"] if p.fit_params else None,
        "group_name": p.group,
    } for p in peaks if p.group]

    if rows:
        sb.table("raman_peaks").insert(rows).execute()

# -------------------------------------------------
# FEATURES ML
# -------------------------------------------------
def insert_ml_features(
    sample_id: str,
    spectrum_id: str,
    features: Dict[str, float],
    label: str,
):
    sb = get_supabase()
    sb.table("ml_features").insert({
        "sample_id": sample_id,
        "spectrum_id": spectrum_id,
        "features": features,
        "label": label,
    }).execute()
