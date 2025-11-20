def ruwe(t_obs, scan_angle, parallax_factor, g_mag, ra_diffs, dec_diffs):
    # A matrix (N,5), astrometric signal
    A = pt.stack([
        pt.sin(scan_angle),
        pt.cos(scan_angle),
        parallax_factor,
        t_obs * pt.sin(scan_angle),
        t_obs * pt.cos(scan_angle)
    ], axis=1)
    # synthetic along-scan positions for observations
    al_positions = ra_diffs*pt.sin(scan_angle) + dec_diffs*pt.cos(scan_angle)
    # AL errors
    al_errors = sigma_ast(g_mag)
    # ACAT matrix, model projection matrix
    ATA = A.T @ A #((5,N)x(N,5)=(5,5))
    C = pt.linalg.solve(ATA, pt.eye(5))  # (ATA)-1, (5,5)
    ACAT = A @ C @ A.T #((N,5)x(5,5)x(5,N)=(N,N))
    # residuals
    r = al_positions - ACAT @ al_positions #actual data - model data
    # RUWE
    N = t_obs.shape[0]
    ruwe_model = pt.sqrt(pt.sum((r / al_errors)**2)/(N - 5))

    return ruwe_model
