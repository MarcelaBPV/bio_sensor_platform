st.markdown("### Pipeline estilo ramanchada2 (Figura 1)")

run_rc2 = st.button("Rodar pipeline completo (rc2-style)")

if run_rc2:
    if not sample_file:
        st.error("Carregue o espectro da amostra.")
        st.stop()

    # Carrega amostra
    sample_spec = load_spectrum_file(sample_file)

    # Carrega silício (opcional)
    silicon_spec = None
    if si_file:
        silicon_spec = load_spectrum_file(si_file)

    # Coeficientes base (opcionais – se quiser calibração X)
    base_poly_coeffs = None
    if coeffs_str.strip():
        base_poly_coeffs = np.fromstring(coeffs_str, sep=",")

    # Aqui você pode, no futuro, carregar também ref_measured/ref_certified
    res_rc2 = run_full_pipeline(
        sample_spec=sample_spec,
        silicon_spec=silicon_spec,
        base_poly_coeffs=base_poly_coeffs,
        ref_measured=None,
        ref_certified=None,
        use_lmfit=use_lmfit,
    )

    st.success("Pipeline rc2-style concluído.")

    # Plots tipo (raw vs processado vs calibrado)
    fig2, axs2 = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axs2[0].plot(res_rc2["x_raw"], res_rc2["y_raw"], lw=0.7)
    axs2[0].set_title("Raw")

    spec_proc = res_rc2["spec_processed"]
    axs2[1].plot(spec_proc.x, spec_proc.y, lw=0.9)
    axs2[1].set_title("Processado (despike + baseline + smooth + norm)")

    axs2[2].plot(res_rc2["x_cal"], res_rc2["y_cal_int"], lw=0.9)
    axs2[2].set_title("No eixo calibrado (se disponível)")

    for ax in axs2:
        ax.set_xlabel("Wavenumber / cm⁻¹")
        ax.set_ylabel("Intensidade (a.u.)")

    st.pyplot(fig2)

    # Tabela de picos
    peaks = res_rc2["peaks"]
    if peaks:
        df_peaks_rc2 = pd.DataFrame(
            [
                {
                    "position_cm-1": p.position_cm1,
                    "intensity": p.intensity,
                    "width": p.width or "",
                    "fit_params": p.fit_params or {},
                }
                for p in peaks
            ]
        )
        st.subheader("Picos detectados (pipeline rc2-style)")
        st.dataframe(df_peaks_rc2)
    else:
        st.info("Nenhum pico detectado com os parâmetros atuais.")
