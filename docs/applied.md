# Applied modules and scientific acceptance

Install the core package with pip. To reproduce all six case studies, use Python 3.12 and install the references and interop extras. Run the following from the checkout.

```sh
python -m pip install '.[references,interop]' pytest
python -m pytest -q
python -Werror -m benchmarks.applied results
```

The exporter writes one JSON, CSV and HTML record per study plus applied.json with noise and throughput measurements. Use --core-only to run the first five studies and the two performance experiments without optional packages. The scientific workflow archives the full exports under applied-scientific-studies.

| Module | API and tests | Scientific reference and limits |
|---|---|---|
| 21. Noise | markovq.noise implements kraus_model, validate_kraus, apply_channel, readout and noisy_walk. Tests in test_applied.py and test_applied_references.py. | Full density matrices evolve by a unitary step followed by a trace-preserving channel. Final position readout uses row assignment probabilities. Tests compare the action with Qiskit and analytic qubit cases. Memory grows quadratically in Hilbert-space dimension.|
| 22. Biological networks | markovq.networks provides rank_nodes, spectral_communities and heldout_metrics. The executable case is benchmarks.applied.protein. | Uses degree, one-step propagation, personalized restart, heat diffusion and continuous-time quantum probabilities. Tied scores share top-k positions fractionally. AUPRC uses average precision. The five fixed seed selections measure sensitivity within one protein. Structural labels correlate with the coordinates. External generalization and disease mechanism inference are untested.|
| 23. Distributed simulation | markovq.batch provides endpoint_plan, run_partition, merge_partitions and simulate_endpoints. Tests compare worker counts, reordered partitions and exact distributions. | Independent multinomial movements from each occupied site exactly sample endpoint counts for independent walkers. The plan fixes sparse kernel, initial law, seed, step count and partition size. External workers may execute individual partitions. Reassembly checks completeness and plan identity. Path histories require separate simulation. Measurements here use local processes.|
| 24. Circuit execution and interoperability | markovq.circuits and markovq.interop. Tests in test_applied_references.py and test_phase_suite.py. | Power-of-two cycle circuits are checked as full operators up to phase. General dense graph synthesis has an eight-qubit ceiling. Target compilation gives resource counts. The experiment uses ideal statevector evolution and a separately sampled synthetic readout model, CalibMap histories, ShotWeave allocations and BornLens comparisons. No provider shots were purchased.|
| 25. Scientific reporting | markovq.reporting.experiment_record and render. Reproduction through benchmarks.applied and benchmarks.phase_suite. | Machine-readable records include a canonical input digest, versions, protocol, assumptions, resources, references and results. Peak RSS is cumulative for the parent process and excludes child RSS. Wall times depend on platform and startup. Input fingerprints identify exact values.|

## Evidence by benchmark family

A (spreading) and B (ruin) have individual study exports. C (classification), D (mixing) and E (CTMC) use the [classical benchmarks](benchmark-results.json) and the analytical checks indexed in [foundations](foundations.md). F (graph evolution), G (Szegedy), H (detection), I (return and mixing), J (open dynamics) and K (search) are indexed in [quantum acceptance](quantum-acceptance.md) and [numerical consolidation](consolidation-review.md). L (biological ranking) is [study 5](case-studies/study-5.json). M (noise) and N (scalability) have [recorded local measurements](applied-benchmarks.json). O (PhaseTensor interoperability) is the sixth study in the scientific workflow artifact.

## Case studies

The six exported studies cover line spreading, gambler's ruin, Szegedy phase correspondence, monitored recurrence and detection, residue-network ranking, and a noise-aware four-cycle circuit. Studies 1–5 have archived local results in this directory. Study 6 uses separately installed releases of all three PhaseTensor libraries. BornLens tests position distributions separately from simultaneous Pauli comparisons. Resource estimates describe this specific measurement protocol. See the [dataset manifest](../benchmarks/data/README.md) for input provenance and held-out labels.

The optional execute_backend interface submits measured circuits to a supplied Qiskit backend implementing run and waits for its result. The backend controls access, sampling and charges. CI exercises only BasicSimulator. QuTiP master-equation exponentials independently check amplitude damping and dephasing.
