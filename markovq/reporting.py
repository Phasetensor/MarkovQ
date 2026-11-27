"""Portable validated experiment records with canonical input fingerprints."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
import platform
from importlib.metadata import version
from typing import Mapping

import numpy as np

from .passage import Passage
from .quantum.measurement import EventualDetection


def scientific_value(value, status: str, reason: str) -> dict:
    """Attach the scientific origin of a nonfinite or missing observation."""
    if status not in {"infinite", "undefined", "unresolved", "overflow", "missing", "unexpected"} or not reason:
        raise ValueError("invalid scientific status")
    if value is not None and np.isfinite(value):
        raise ValueError("expected a nonfinite value")
    sign = None if value is None or np.isnan(value) else ("positive" if value > 0 else "negative")
    return {"value": None, "status": status, "reason": reason, "sign": sign}


def _quantity(value, status, reason):
    return value if np.isfinite(value) else scientific_value(value, status, reason)


def _encode(value):
    if isinstance(value, EventualDetection):
        d = value.diagnostics
        fields = value._asdict()
        fields["probability"] = _quantity(value.probability, "unresolved", d.status)
        fields["conditional_mean"] = _quantity(value.conditional_mean, d.mean_status, "conditional detection time")
        fields["diagnostics"] = dict(d._asdict(), unconditional_mean=_quantity(d.unconditional_mean, d.unconditional_status, "unconditional detection time"))
        return _encode(fields)
    if isinstance(value, Passage):
        return _encode(dict(value._asdict(), mean=[_quantity(x, "infinite" if np.isinf(x) else "unexpected", "unconditional hitting time") for x in value.mean],
                            conditional_mean=[_quantity(x, "undefined" if p == 0 else "unexpected", "conditional hitting time") for x,p in zip(value.conditional_mean,value.probability)]))
    if hasattr(value, "_asdict"):
        return _encode(value._asdict())
    if isinstance(value, Mapping):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _encode(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_encode(v) for v in value]
    if isinstance(value, np.generic):
        return _encode(value.item())
    if isinstance(value, complex):
        return {"real": _encode(value.real), "imag": _encode(value.imag)}
    if isinstance(value, float) and not math.isfinite(value):
        return scientific_value(value, "unexpected", "nonfinite value without a diagnostic")
    return value


def experiment_record(
    *,
    model: str,
    inputs: Mapping,
    protocol: str,
    seed: int,
    solver: str,
    tolerance: float,
    baseline: str,
    runtime_s: float,
    peak_rss_bytes: int,
    limitations: str,
    results: Mapping,
) -> dict:

    if (
        not all(
            isinstance(x, str) and x.strip()
            for x in (model, protocol, solver, baseline, limitations)
        )
        or not all(math.isfinite(x) and x >= 0 for x in (runtime_s, tolerance))
        or peak_rss_bytes < 0
    ):
        raise ValueError("incomplete scientific record")
    if not {"initial_state", "graph", "boundaries", "dataset"} <= inputs.keys():
        raise ValueError("missing model inputs")
    payload = json.dumps(inputs, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return {
        "schema": 2,
        "model": model,
        "inputs": json.loads(payload),
        "input_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "protocol": protocol,
        "seed": seed,
        "solver": solver,
        "tolerance": tolerance,
        "baseline": baseline,
        "runtime_s": runtime_s,
        "peak_rss_bytes": peak_rss_bytes,
        "limitations": limitations,
        "results": _encode(results),
        "versions": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
        },
    }


def render(record: Mapping, fmt: str = "json") -> str:
    record = _encode(record)
    payload = json.dumps(record, sort_keys=True, allow_nan=False, indent=2)
    if fmt == "json":
        return payload
    if fmt == "html":
        return (
            "<!doctype html><meta charset=utf-8><title>MarkovQ experiment</title><pre>"
            + html.escape(payload)
            + "</pre>"
        )
    if fmt == "csv":
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(["field", "json_value"])
        writer.writerows(
            (key, json.dumps(record[key], sort_keys=True, allow_nan=False))
            for key in sorted(record)
        )
        return stream.getvalue()
    raise ValueError("unknown report format")
