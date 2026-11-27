"""Small-system analytical, solver and resolution diagnostics."""
from decimal import Decimal, localcontext

import numpy as np

from markovq import quantum as q


def stability():
    rows = []
    target = np.diag([1, 0])
    for p in (.8, .1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-18):
        a, b = np.sqrt(1-p), np.sqrt(p)
        u = np.array([[a,b],[-b,a]])
        r = q.eventual_detection(u,[0,1],target)
        with localcontext() as ctx:
            ctx.prec = 80
            reference = float(1 / Decimal(str(p)))
        def finite(x):
            return float(x) if np.isfinite(x) else None
        rows.append({
            "per_attempt": p, "decimal_reference":reference, "mean":finite(r.conditional_mean),
            "relative_error":finite(abs(r.conditional_mean/reference-1)), "status":r.diagnostics.mean_status,
            "radius":finite(r.diagnostics.spectral_radius), "condition_estimate":finite(r.diagnostics.condition_bound),
            "relative_residual":finite(r.diagnostics.relative_residual),
            "minimum_duration_eigenvalue":finite(r.diagnostics.minimum_eigenvalue),
            "duration_hermiticity_error":finite(r.diagnostics.hermiticity_error),
        })
    phases=[]
    h=q.hadamard()
    for delta in [0.,1e-3,1e-7,1e-11,1e-15]:
        u=h@np.diag([1,np.exp(1j*delta)])@h
        m=q.quantum_mixing(u,[1,0],37)
        r=q.eventual_detection(u,[0,1],target)
        phases.append({"separation":delta,"finite_spectral_error":float(max(abs(m.average-m.finite_spectral))),
                       "limit":m.limiting_average.tolist(),"limit_sensitivity_tv":m.limit_sensitivity_tv,
                       "within_resolution_splitting_tv":m.unresolved_splitting_tv,"mixing_status":m.spectral_status,
                       "detection_status":r.diagnostics.status,"probability_range":r.diagnostics.probability_range,
                       "dark_invariance_residual":r.diagnostics.dark_invariance_residual,"target_residual":r.diagnostics.dark_target_residual})
    return {"geometric":rows,"phases":phases}
