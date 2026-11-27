"""Independent endpoint trajectories with worker-invariant random streams."""

from __future__ import annotations

import hashlib
import struct
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Iterable, NamedTuple

import numpy as np
from scipy import sparse

from ._types import MatrixInput
from .chains import count, distribution, validate_transition_matrix


class BatchResult(NamedTuple):
    counts: np.ndarray
    standard_errors: np.ndarray
    batches: int
    storage_bytes: int


class Partition(NamedTuple):
    fingerprint: str
    partition_id: int
    counts: np.ndarray


@dataclass(frozen=True)
class EndpointPlan:
    transition: sparse.csr_matrix
    initial: np.ndarray
    steps: int
    trajectories: int
    seed: int
    batch_size: int
    fingerprint: str

    @property
    def batches(self) -> int:
        return (self.trajectories + self.batch_size - 1) // self.batch_size


def endpoint_plan(
    transition: MatrixInput,
    initial,
    steps: int,
    trajectories: int,
    *,
    seed: int,
    batch_size: int = 2048,
) -> EndpointPlan:
    p = sparse.csr_matrix(validate_transition_matrix(transition), copy=True)
    p.sum_duplicates()
    p.eliminate_zeros()
    p.sort_indices()
    law = distribution(initial, p.shape[0]).copy()
    n, b = count(trajectories), count(batch_size)
    steps, seed = count(steps), count(seed)
    if not min(n, b):
        raise ValueError("positive trajectory and batch counts required")
    fingerprint = _fingerprint(p, law, steps, n, seed, b)
    for array in (p.data, p.indices, p.indptr, law):
        array.flags.writeable = False
    return EndpointPlan(p, law, steps, n, seed, b, fingerprint)


def _fingerprint(p, law, steps, n, seed, b):
    digest = hashlib.sha256()
    digest.update(str((p.shape, steps, n, seed, b)).encode("ascii"))
    for a, dtype in [
        (p.data, "<f8"),
        (p.indices, "<i8"),
        (p.indptr, "<i8"),
        (law, "<f8"),
    ]:
        digest.update(struct.pack("<Q", len(a)))
        digest.update(a.astype(dtype, copy=False).tobytes())
    return digest.hexdigest()


def _verify(plan):
    p = plan.transition
    if (
        p.format != "csr"
        or p.data.dtype != np.float64
        or plan.initial.dtype != np.float64
        or any(a.dtype.kind != "i" for a in (p.indptr, p.indices))
        or any(a.ndim != 1 for a in (p.data, p.indices, p.indptr, plan.initial))
        or _fingerprint(
            p, plan.initial, plan.steps, plan.trajectories, plan.seed, plan.batch_size
        )
        != plan.fingerprint
    ):
        raise ValueError("inconsistent endpoint plan")


def run_partition(plan: EndpointPlan, index: int) -> Partition:
    # Snapshot both mutable containers before verifying the inputs used below.
    plan = replace(plan, transition=plan.transition.copy(), initial=plan.initial.copy())
    _verify(plan)
    i = count(index)
    if i >= plan.batches:
        raise ValueError("partition outside plan")
    rng = np.random.default_rng(np.random.SeedSequence(plan.seed, spawn_key=(i,)))
    n = min(plan.batch_size, plan.trajectories - i * plan.batch_size)
    counts = rng.multinomial(n, plan.initial)
    p = plan.transition
    for _ in range(plan.steps):
        following = np.zeros_like(counts)
        for vertex in np.flatnonzero(counts):
            start, end = p.indptr[vertex : vertex + 2]
            following[p.indices[start:end]] += rng.multinomial(
                counts[vertex], p.data[start:end]
            )
        counts = following
    return Partition(plan.fingerprint, i, counts)


def merge_partitions(plan: EndpointPlan, parts: Iterable[Partition]) -> BatchResult:
    _verify(plan)
    total = np.zeros(plan.transition.shape[0], np.int64)
    seen = set()
    for part in parts:
        i = count(part.partition_id)
        x = np.asarray(part.counts)
        if part.fingerprint != plan.fingerprint or i >= plan.batches or i in seen:
            raise ValueError("foreign, duplicate or invalid partition")
        n = min(plan.batch_size, plan.trajectories - i * plan.batch_size)
        if (
            x.shape != total.shape
            or x.dtype.kind not in "iu"
            or (x < 0).any()
            or sum(map(int, x)) != n
        ):
            raise ValueError("invalid partition counts")
        seen.add(i)
        total += x.astype(np.int64)
    _verify(plan)
    if len(seen) != plan.batches:
        raise ValueError("incomplete partitions")
    p = plan.transition
    prob = total / plan.trajectories
    return BatchResult(
        total,
        np.sqrt(prob * (1 - prob) / plan.trajectories),
        plan.batches,
        p.data.nbytes + p.indices.nbytes + p.indptr.nbytes,
    )


def _initialize(plan):
    global _worker_plan
    _worker_plan = plan


_worker_plan: EndpointPlan


def _execute(index):
    return run_partition(_worker_plan, index)


def simulate_endpoints(
    transition: MatrixInput,
    initial,
    steps: int,
    trajectories: int,
    *,
    seed: int,
    batch_size: int = 2048,
    workers: int = 1,
) -> BatchResult:
    plan = endpoint_plan(
        transition, initial, steps, trajectories, seed=seed, batch_size=batch_size
    )
    w = count(workers)
    if not w:
        raise ValueError("positive worker count required")
    if w == 1:
        return merge_partitions(
            plan, (run_partition(plan, i) for i in range(plan.batches))
        )
    with ProcessPoolExecutor(
        max_workers=min(w, plan.batches), initializer=_initialize, initargs=(plan,)
    ) as pool:
        return merge_partitions(plan, pool.map(_execute, range(plan.batches)))
