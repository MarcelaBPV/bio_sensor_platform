def calibrate_with_fixed_pattern_and_silicon(
    silicon_file,
    sample_file,
    blank_file,
    base_poly_coeffs: np.ndarray,
    silicon_ref_position: float = 520.7,
    preprocess_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Workflow de calibração simplificado com upload conjunto:

    Entradas obrigatórias:
        - silicon_file : espectro do padrão de Silício
        - sample_file  : espectro da amostra
        - blank_file   : espectro do porta-amostra em branco (p.ex. lâmina/substrato)
        - base_poly_coeffs : coeficientes do polinômio global de calibração
        - silicon_ref_position : posição de referência do pico de Si (cm-1)

    Passos:
        1) Lê e pré-processa o espectro de Silício.
        2) Aplica o polinômio base (padrão fixo) ao Si.
        3) Localiza o pico de Si (~520.7 cm-1) e calcula o desvio residual.
        4) Define a função de correção final (poly_base + offset de Si).
        5) Lê e pré-processa a amostra e o blank.
        6) Aplica a correção final aos eixos da amostra e do blank.
        7) Retorna todos os espectros e metadados, deixando pronto
           para filtros/correções adicionais (ex.: subtração de blank).

    Retorna dict com:
        - x_sample_raw, y_sample_raw
        - x_sample_proc, y_sample_proc
        - x_sample_calibrated
        - x_blank_raw, y_blank_raw
        - x_blank_proc, y_blank_proc
        - x_blank_calibrated
        - x_silicon_raw, y_silicon_raw
        - x_silicon_proc, y_silicon_proc
        - x_silicon_calibrated
        - y_sample_blank_corrected (subtração simples de blank, opcional)
        - meta_sample, meta_blank, meta_silicon
        - informações de calibração (coeficientes base, posição do Si, offset)
    """
    if preprocess_kwargs is None:
        preprocess_kwargs = {
            "despike_method": "auto_compare",
            "smooth": True,
            "baseline_method": "als",
            "normalize": False,
        }

    # -----------------------
    # 1) SILÍCIO
    # -----------------------
    x_si_raw, y_si_raw = load_spectrum(silicon_file)
    x_si, y_si, meta_si = preprocess_spectrum(
        x_si_raw, y_si_raw, **preprocess_kwargs
    )

    # Aplica polinômio base (padrão fixo)
    x_si_base = apply_base_wavenumber_correction(x_si, base_poly_coeffs)

    # Procura pico mais intenso na região típica do Si no eixo já corrigido
    mask_si = (x_si_base >= 480) & (x_si_base <= 560)
    if not np.any(mask_si):
        raise RuntimeError("Não há pontos suficientes na janela de Si (480–560 cm-1).")

    idx_max = np.argmax(y_si[mask_si])
    x_si_region = x_si_base[mask_si]
    si_cal_base = float(x_si_region[idx_max])  # posição do pico de Si após correção base

    # Offset residual entre o padrão fixo e o valor de referência do Si
    delta = silicon_ref_position - si_cal_base

    def corrector_final(x_arr: np.ndarray) -> np.ndarray:
        """
        Correção final aplicada aos eixos:
            x_corr = poly_base(x_obs) + delta_Si
        """
        x_base = apply_base_wavenumber_correction(x_arr, base_poly_coeffs)
        return x_base + delta

    x_si_cal = corrector_final(x_si)

    # -----------------------
    # 2) AMOSTRA
    # -----------------------
    x_s_raw, y_s_raw = load_spectrum(sample_file)
    x_s, y_s, meta_s = preprocess_spectrum(
        x_s_raw, y_s_raw, **preprocess_kwargs
    )
    x_s_cal = corrector_final(x_s)

    # -----------------------
    # 3) BLANK (porta-amostra em branco)
    # -----------------------
    x_b_raw, y_b_raw = load_spectrum(blank_file)
    x_b, y_b, meta_b = preprocess_spectrum(
        x_b_raw, y_b_raw, **preprocess_kwargs
    )
    x_b_cal = corrector_final(x_b)

    # -----------------------
    # 4) Subtração simples de blank (opcional)
    #    (interpola o blank no eixo calibrado da amostra)
    # -----------------------
    # Garante que usamos o mesmo eixo para a correção
    y_b_interp = np.interp(x_s_cal, x_b_cal, y_b)
    y_sample_blank_corrected = y_s - y_b_interp

    return {
        # Amostra
        "x_sample_raw": x_s_raw,
        "y_sample_raw": y_s_raw,
        "x_sample_proc": x_s,
        "y_sample_proc": y_s,
        "x_sample_calibrated": x_s_cal,

        # Blank
        "x_blank_raw": x_b_raw,
        "y_blank_raw": y_b_raw,
        "x_blank_proc": x_b,
        "y_blank_proc": y_b,
        "x_blank_calibrated": x_b_cal,

        # Silício
        "x_silicon_raw": x_si_raw,
        "y_silicon_raw": y_si_raw,
        "x_silicon_proc": x_si,
        "y_silicon_proc": y_si,
        "x_silicon_calibrated": x_si_cal,

        # Correção de blank (já pronto para filtros posteriores)
        "y_sample_blank_corrected": y_sample_blank_corrected,

        # Metadados
        "meta_sample": meta_s,
        "meta_blank": meta_b,
        "meta_silicon": meta_si,

        # Info da calibração
        "calibration": {
            "base_poly_coeffs": np.asarray(base_poly_coeffs, dtype=float).tolist(),
            "si_cal_base": si_cal_base,
            "silicon_ref_position": silicon_ref_position,
            "laser_zero_delta": delta,
        },
    }
