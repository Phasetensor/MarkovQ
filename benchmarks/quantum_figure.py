"""Render the measured quantum benchmark record as a standalone SVG."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render() -> None:
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "docs/quantum-benchmarks.json").read_text())
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "MarkovQ-quantum"})
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    spread = data["spreading"]
    steps = [row["steps"] for row in spread]
    for key, label in [("classical_msd", "Classical"), ("quantum_msd", "Hadamard")]:
        axes[0].loglog(steps, [row[key] for row in spread], "o-", label=label)
    axes[0].set(
        xlabel="Steps",
        ylabel="Mean squared displacement",
        title="Spreading on the line",
    )
    axes[0].legend()
    monitored = data["monitored"]
    for key, label in [
        ("reference_mass", "Exact"),
        ("empirical_mass", "10,000 trajectories"),
    ]:
        axes[1].plot(np.cumsum(monitored[key]), marker=".", label=label)
    axes[1].axhline(monitored["eventual_probability"], color="grey", linestyle=":")
    axes[1].set(
        xlabel="Measurement attempts",
        ylabel="Cumulative detection probability",
        title="Detection with a dark state",
    )
    axes[1].legend()
    search = data["search"]
    axes[2].plot(search["discrete_queries"], search["discrete_success"], "o-")
    axes[2].set(
        xlabel="Marking queries",
        ylabel="Success probability",
        title="Four-vertex amplitude amplification",
        xticks=[0, 1, 2],
        ylim=(0, 1.05),
    )
    for axis in axes:
        axis.spines[["right", "top"]].set_visible(False)
    figure.savefig(root / "docs/quantum-benchmarks.svg", metadata={"Date": None})
    plt.close(figure)


if __name__ == "__main__":
    render()
