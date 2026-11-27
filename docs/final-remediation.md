# Final remediation and release preparation

Baseline 05c29d621ef10ecb7bb8c4631a96c46b2440c5c3. The seven findings concern existing public functionality. Final CI receipts are linked from the acceptance message.

| ID and status | Reproduction and cause | Correction, reference and regression evidence | Limitation |
|---|---|---|---|
|R09 · Resolved|Swapping amplitudes 1e-180 and 2e-180 gave two zeros and best index 0.|quantum/search.py ranks validated log probabilities, retaining first maximum. test_final_remediation compares a permutation and a continuous two-site swap at pi/2 with their exact amplitude ratios.|Amplitudes already rounded to zero have no recoverable magnitude.|
|R10 · Mitigated|For an identity step, target site 0 and dark amplitude 1e-100, the baseline returned an unconditional time of one.|quantum/measurement.py uses exact coordinate reachability for the represented matrices and a projection uncertainty diagnostic in the remaining cases. The independent two-site distribution has non-detection mass equal to the squared dark amplitude. Tests include amplitudes down to 1e-250 and cancellation on a rotated bright state.|General floating-point subspaces give numerical evidence. Ambiguous near-zero weights can produce an unresolved expectation.|
|R11 · Documented limitation|NamedTuple additions in development changed positional arity.|Named attributes remain the documented interface. See the field inventory below and the API smoke tests.|Code unpacking earlier development snapshots positionally must migrate.|
|R12 · Resolved|The archived synthetic experiment had a corrected estimate above one. Measurement-error inversion is unconstrained.|interop.py reports the raw estimator, domain flag, empirical and ideal means and sampling standard error. test_final_remediation exercises both boundaries and both exterior regions. The archive preserves the observed overshoot.|The reported sampling error conditions on supplied calibration values. Calibration intervals are separate.|
|R13 · Resolved|Strict JSON rejected infinity and bare NaN.|reporting.py schema 2 uses typed records for nonfinite values. Tests export a disconnected hitting problem, defective and unresolved detection, search overflow and a nested unexpected NaN through JSON, CSV and HTML.|An unlabelled nonfinite value has unknown provenance. Generic data producers must supply scientific meaning explicitly.|
|R14 · Resolved|Replacing the transition indices in a constructed plan changed results without changing its digest.|batch.py owns independent canonical buffers and verifies inputs at partition and merge boundaries. Tests exercise pickle, caller mutations, replacements, reorderings, invalid counts and multiple worker counts.|Digests identify inputs and cannot authenticate untrusted external workers or prove the trajectory counts are genuine.|
|R15 · Resolved|Subtracting input width from active wires gives a negative answer for a compiled idle circuit.|circuits.py derives address sets from operations and input placement from transpile layout. Tests use identity, explicit ancillas, a linear routing target, two optimization levels and BasicSimulator.|Address counts refer to compiled instructions. Device occupancy, pulse scheduling and hardware runtime remain unmeasured.|

## Public result field inventory

Use named attributes when consuming results. In particular EventualDetection exposes probability, conditional_mean, dark_projector and diagnostics. DetectionDiagnostics adds unconditional_status, dark_amplitude, dark_amplitude_error and dark_weight_status. Mixing includes finite_spectral, limit_sensitivity_tv, spectral_status and unresolved_splitting_tv. SearchResult retains best_index and success and includes log_success. SearchResult.expected_cost and resource_status use an explicit candidate index. The following inventory describes the current development snapshot.

- `BatchResult` — counts, standard_errors, batches, storage_bytes

- `Partition` — fingerprint, partition_id, counts

- `StateClass` — states, recurrent, period

- `NoisyWalk` — states, populations, purity, coherence, trace_error

- `Passage` — probability, mean, conditional_mean

- `Absorption` — transient, classes, visits, probabilities, mean, condition_number

- `ImplicitWalk` — A, B, operator

- `Detection` — first, survival, surviving_ket

- `DetectionDiagnostics` — status, mean_status, unconditional_mean, phase_tol, rank_tol, dark_invariance_residual, dark_target_residual, probability_range, spectral_radius, condition_bound, relative_residual, hermiticity_error, minimum_eigenvalue, unconditional_status, dark_amplitude, dark_amplitude_error, dark_weight_status

- `EventualDetection` — probability, conditional_mean, dark_projector, diagnostics

- `Recurrence` — amplitudes, occupancy, first_amplitudes, monitored

- `Mixing` — instantaneous, average, limiting_average, tv_to_limit, finite_spectral, limit_sensitivity_tv, spectral_status, unresolved_splitting_tv

- `OpenEvolution` — states, populations, coherence_l1, trace_error, hermiticity_error, minimum_eigenvalue

- `SearchResult` — candidates, success, best_index, discrete_queries, oracle_time, preparations_per_trial, measurements_per_trial, log_success

- `CoinedWalk` — operator, position, coin

- `SzegedyWalk` — A, B, reflection_a, reflection_b, operator, discriminant

- `Spectrum` — values, vectors, residual, orthogonality_error

## Release preparation

The PyPI JSON endpoint for the normalized name markovq returned HTTP 404 during the readiness check. This establishes the absence of a public listing at that time. It does not reserve the name or verify that a pending publisher is configured. The target is v0.1.0. The package and citation metadata remain at 0.1.0.dev0 pending approval. The tag-only workflow checks the package version against the tag, builds once, verifies the SHA-256 manifest, produces GitHub provenance and submits the same distributions through PyPI OIDC. The source archive includes the license, citation, benchmarks, residue data, reports and typed package sources.

A release operator must configure or confirm the PyPI pending publisher for Phasetensor/MarkovQ, workflow release.yml, environment pypi. Configure release approval and tag permissions in the repository settings. Version promotion, tagging, publication and public installation verification are subsequent release operations. The scientific workflow performs an isolated wheel installation and tests the development archives.

The independent rerun of [44 scaling scenarios](final-scaling.json) uses the original consolidation benchmark, graph sizes and seeds. The applied benchmarks retain their original seed choices, including the five held-out residue ranking evaluations. The final projection status distinguishes an exactly identified positive coordinate component, a numerically positive weight, a zero in the resolved model, a weight below resolution, and an ambiguous subspace.
