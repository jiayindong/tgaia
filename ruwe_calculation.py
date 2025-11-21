# ruwe_calculation.py
import pytensor.tensor as pt


def ruwe(t_obs, scan_angle, parallax_factor, g_mag, ra_diffs, dec_diffs):
    """
    Compute model RUWE from synthetic along-scan residuals.
    All inputs are PyTensor tensors of length N.
    """
    # Design matrix A (N, 5)
    A = pt.stack(
        [
            pt.sin(scan_angle),
            pt.cos(scan_angle),
            parallax_factor,
            t_obs * pt.sin(scan_angle),
            t_obs * pt.cos(scan_angle),
        ],
        axis=1,
    )

    # Along-scan positions from RA/DEC offsets (same projection as forward model)
    al_positions = ra_diffs * pt.sin(scan_angle) + dec_diffs * pt.cos(scan_angle)

    # AL uncertainties; sigma_ast should map G mag → AL error (mas)
    al_errors = sigma_ast(g_mag)

    # ACAT projection matrix
    ATA = A.T @ A
    C = pt.linalg.solve(ATA, pt.eye(5))
    ACAT = A @ C @ A.T

    # residuals
    r = al_positions - ACAT @ al_positions

    # RUWE definition
    N = t_obs.shape[0]
    ruwe_model = pt.sqrt(pt.sum((r / al_errors) ** 2) / (N - 5))

    return ruwe_model
