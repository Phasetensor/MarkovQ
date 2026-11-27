"""Applications and acceptance cases for MarkovQ, using no provider credentials."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from markovq import absorption, line_walk
from markovq import quantum as q
from markovq.batch import simulate_endpoints
from markovq.networks import heldout_metrics, rank_nodes, spectral_communities
from markovq.noise import kraus_model, noisy_walk, readout
from markovq.reporting import experiment_record, render

DATA = Path(__file__).parent / "data" / "crambin.csv"


def protein():
    with DATA.open() as stream:
        rows = list(csv.DictReader(stream))
    coords = np.array([[float(r[k]) for k in "xyz"] for r in rows])
    residue = np.array([int(r["residue"]) for r in rows])
    helix = np.array([int(r["helix"]) for r in rows], bool)
    dist = cdist(coords, coords)
    graph = ((dist <= 8) & (dist > 0)).astype(float)
    # All training seeds in helix 7–19. Evaluate residues 22–46 only.
    pool = np.flatnonzero(helix & (residue <= 19))
    test = residue >= 22
    pos = np.flatnonzero(test & helix)
    neg = np.flatnonzero(test & ~helix)
    trials = []
    score_history = {}
    for seed in (501, 502, 503, 504, 505):
        selected = np.sort(np.random.default_rng(seed).choice(pool, 4, replace=False))
        started = perf_counter()
        scores = rank_nodes(graph, selected, restart=0.2, time=0.5)
        elapsed = perf_counter() - started
        trials.append(
            {
                "rng_seed": seed,
                "seed_residues": residue[selected].tolist(),
                "runtime_s": elapsed,
                "metrics": {
                    name: heldout_metrics(score, selected, pos, neg, 5)
                    for name, score in scores.items()
                },
            }
        )
        for name, score in scores.items():
            score_history.setdefault(name, []).append(score[test])
    stability = {}
    for name, values in score_history.items():
        correlations = [
            float(spearmanr(a, b)[0]) for i, a in enumerate(values) for b in values[:i]
        ]
        stability[name] = float(np.mean(correlations))
    communities = spectral_communities(graph)
    return {
        "dataset": "PDB 1CRN, crambin, chain A",
        "input_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "nodes": len(rows),
        "edges": int(graph.sum() / 2),
        "positive_test_residues": residue[pos].tolist(),
        "negative_test_residues": residue[neg].tolist(),
        "trials": trials,
        "mean_pairwise_spearman": stability,
        "community_method": communities["status"],
        "community_labels": communities["labels"].tolist(),
    }


def noise_benchmark():
    step = q.cycle_walk(4).operator
    rho = np.diag([1] + [0] * 7)
    ideal = noisy_walk(step, rho, [np.eye(8)], 8)
    rows = []
    for name in ("depolarizing", "dephasing", "amplitude-damping"):
        for p in (0.0, 0.02, 0.15, 1.0):
            ops = [np.kron(k, np.eye(4)) for k in kraus_model(name, p)]
            out = noisy_walk(step, rho, ops, 8)
            position = out.populations[-1].reshape(2, 4).sum(axis=0)
            ideal_pos = ideal.populations[-1].reshape(2, 4).sum(axis=0)
            rows.append(
                {
                    "channel": name,
                    "strength": p,
                    "position": position.tolist(),
                    "purity": float(out.purity[-1]),
                    "coherence": float(out.coherence[-1]),
                    "tv": float(abs(position - ideal_pos).sum() / 2),
                    "trace_error": float(out.trace_error.max()),
                    "readout_tv": float(
                        abs(
                            readout(
                                position, 0.92 * np.eye(4) + 0.08 * np.ones((4, 4)) / 4
                            )
                            - position
                        ).sum()
                        / 2
                    ),
                }
            )
    return rows


def batch_benchmark():
    rows = []
    for n in (16, 64, 256):
        np.random.default_rng(901)
        a = sparse.diags(
            [0.25 * np.ones(n - 1), 0.5 * np.ones(n), 0.25 * np.ones(n - 1)],
            [-1, 0, 1],
            format="lil",
        )
        a[0, 0] = a[-1, -1] = 0.75
        p = a.tocsr()
        law = np.zeros(n)
        law[n // 2] = 1
        reference = law.copy()
        for _ in range(16):
            reference = reference @ p
        for workers in (1, 2):
            elapsed = []
            for _ in range(3):
                start = perf_counter()
                result = simulate_endpoints(
                    p, law, 16, 10000, seed=901, batch_size=1000, workers=workers
                )
                elapsed.append(perf_counter() - start)
            median = float(np.median(elapsed))
            rows.append(
                {
                    "states": n,
                    "workers": workers,
                    "shots": 10000,
                    "steps": 16,
                    "seed": 901,
                    "counts": result.counts.tolist(),
                    "max_standard_error": float(max(result.standard_errors)),
                    "max_probability_error": float(
                        max(abs(result.counts / 10000 - reference))
                    ),
                    "sparse_bytes": result.storage_bytes,
                    "median_seconds": median,
                    "min_seconds": min(elapsed),
                    "max_seconds": max(elapsed),
                    "paths_per_second": 10000 / median,
                }
            )
    for i, row in enumerate(rows):
        row["parallel_efficiency"] = rows[i - i % 2]["median_seconds"] / (
            row["median_seconds"] * row["workers"]
        )
    return rows


def build_studies(include_suite=True):
    studies = []

    def add(
        name, inputs, protocol, solver, baseline, limits, calculate, seed=0, tol=1e-10
    ):
        start = perf_counter()
        results = calculate()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
        studies.append(
            experiment_record(
                model=name,
                inputs=inputs,
                protocol=protocol,
                seed=seed,
                solver=solver,
                tolerance=tol,
                baseline=baseline,
                runtime_s=perf_counter() - start,
                peak_rss_bytes=peak,
                limitations=limits,
                results=results,
            )
        )

    def setup(graph, state, boundaries="finite", dataset="synthetic"):
        return {
            "graph": graph,
            "initial_state": state,
            "boundaries": boundaries,
            "dataset": dataset,
        }

    def spread():
        output = []
        for n in (8, 16, 32, 64, 128):
            x, p = q.line_distribution(n)
            output.append(
                {"steps": n, "quantum_msd": float(p @ (x * x)), "classical_msd": n}
            )
        return {"rows": output}

    add(
        "Line spreading",
        setup("integer line", "origin, symmetric complex coin", "reachable light cone"),
        "terminal position",
        "coined shift",
        "classical exact MSD = n",
        "Specific initial coin and unbiased steps.",
        spread,
    )

    def ruin():
        r = absorption(line_walk(9, boundary="absorb").P)
        return {
            "mean": r.mean.tolist(),
            "exact_mean": [i * (8 - i) for i in range(1, 8)],
        }

    add(
        "Gambler's ruin",
        setup("path 0–8", "all interior states", "absorbing endpoints"),
        "first boundary entry",
        "linear solve",
        "i(8-i)",
        "Finite fair game.",
        ruin,
    )

    def sz():
        w = q.szegedy([[0.75, 0.25], [0.25, 0.75]])
        return {
            "phases": np.sort(np.angle(np.linalg.eigvals(w.operator))).tolist(),
            "classical_eigenvalues": [0.5, 1.0],
        }

    add(
        "Szegedy correspondence",
        setup([[0.75, 0.25], [0.25, 0.75]], "entire operator"),
        "spectrum",
        "two reflections",
        "2 arccos(0.5)",
        "Small reversible kernel.",
        sz,
    )

    def detection():
        from scipy.linalg import block_diag

        u = block_diag(q.hadamard(), 1)
        v = np.array([1, 0, 1]) / np.sqrt(2)
        target = np.diag([1, 0, 0])
        d = q.first_detection(u, v, target, 12)
        rec = q.recurrence(u, v, 12)
        return {
            "first": d.first.tolist(),
            "survival": d.survival.tolist(),
            "return_occupancy": rec.occupancy.tolist(),
            "eventual": q.eventual_detection(u, v, target).probability,
            "renewal_error": float(
                max(abs(abs(rec.first_amplitudes) ** 2 - rec.monitored.first))
            ),
        }

    add(
        "Monitored recurrence and detection",
        setup("Hadamard plus invariant dark site", "equal bright/dark amplitudes"),
        "evolve then project",
        "conditional evolution",
        "geometric detection mass",
        "Finite protocol; monitored return uses initial-state projector.",
        detection,
    )
    add(
        "Biological network ranking",
        setup(
            "8 angstrom C-alpha contacts",
            "four helix-1 seed residues",
            dataset="PDB 1CRN, chain A",
        ),
        "held-out helix 2, negatives from residues >=22",
        "restart, diffusion, CTQW",
        "degree and one-step walk",
        "Single plant protein, structurally correlated annotation, fixed graph, no independent biological discovery.",
        protein,
        501,
    )
    if include_suite:
        from benchmarks.phase_suite import run

        add(
            "PhaseTensor noise-aware walk",
            setup("four-cycle", "coin 0 at vertex 0"),
            "pilot then fresh final samples",
            "Qiskit and synthetic readout",
            "ideal Pauli and position samples",
            "Independent directional readout model; no hardware execution.",
            run,
            2415,
        )
    return studies


def export(directory: Path, include_suite=True):
    directory.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(build_studies(include_suite), 1):
        for fmt in ("json", "csv", "html"):
            (directory / f"study-{i}.{fmt}").write_text(render(record, fmt))
    evidence = {
        "noise": noise_benchmark(),
        "batch": batch_benchmark(),
        "platform": platform.platform(),
    }
    (directory / "applied.json").write_text(
        json.dumps(evidence, allow_nan=False, indent=2)
    )


if __name__ == "__main__":
    export(Path(sys.argv[1]), "--core-only" not in sys.argv[2:])
