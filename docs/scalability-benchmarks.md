# Controlled scalability measurements

The accompanying [44-row record](scalability-data.json) contains median, minimum and maximum of three timings per case, process peak RSS, numerical residuals, conservation errors and NumPy build metadata. Reproduce on Linux or macOS with `OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -Werror -m benchmarks.consolidation`. Each method, size and graph receives a fresh subprocess; three repetitions share that process. RSS is the OS lifetime high-water mark including Python, libraries and reference checks. The timed interval excludes graph input generation and the independent reference. For Szegedy it includes initialization, coherent preparation and one forward action. No distributed workers or hardware jobs enter these figures.

| Method | Size | Median time (ms) | Peak RSS (MiB) |
|---|---:|---:|---:|
| continuous | 1024 | 0.702 | 101.98 |
| szegedy-dense | 32 | 51.589 | 128.31 |
| szegedy-implicit | 32 | 0.454 | 101.80 |
| szegedy-implicit | 1024 | 1.303 | 141.44 |
| lindblad/expm | 16 | 35.769 | 114.17 |
| lindblad/sparse | 16 | 1.682 | 101.75 |
| lindblad/matrix-free | 16 | 1.723 | 101.91 |
| coined | 1024 | 13.521 | 102.56 |

At 1,024 vertices the implicit Szegedy walk acts on a state of length 1,048,576. One real dense unitary of that dimension would require 8 TiB before work arrays; no such allocation was attempted. Continuous walks cover paths, cycles, stars, complete graphs, grids, a seeded sparse random graph and weighted paths. Timings through 1,024 vertices apply to the weighted path, not every family. Dense reference comparisons run through 64 vertices; larger runs use reverse propagation. The coined line benchmark uses steps as its size axis. Detection and mixing use a Hadamard block with additional invariant states and a 20-step horizon. That model checks interference without making a general claim about dense random spectra.

Open-system runs use a coherent cycle Hamiltonian H with collapse operator 0.2H. Commutation gives an independent closed form in the energy basis. Dense explicit-generator runs stop at dimension 16. The sparse path reaches dimension 32 and direct matrix-free action reaches 64. A separate regression covers noncommuting collapse operators. The largest reference error over the heterogeneous metrics was 2.80e-14 and the largest normalization discrepancy was 1.81e-13 (rounded upwards). Solver-specific metrics and units are defined in the benchmark source.

The macOS run used the version-specific full-propagator workaround documented in the numerical review; the dense Liouvillian was still constructed. Timings should not be transferred to other BLAS implementations. RSS at small sizes is dominated by the interpreter and imported numerical libraries.
