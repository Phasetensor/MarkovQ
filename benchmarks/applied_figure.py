"""Render the five held-out AUROC observations for each ranking method."""

import json
import sys
from pathlib import Path


def figure(record: dict) -> str:
    trials = record["results"]["trials"]
    methods = sorted(trials[0]["metrics"])
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 300">',
        "<title>Crambin held-out ranking by seed selection</title>",
        '<rect width="720" height="300" fill="white"/>',
        '<g font-family="sans-serif" fill="#172c40">',
        '<text x="18" y="30" font-size="20">Crambin · held-out AUROC</text>',
    ]
    for tick in range(6):
        x = 190 + 96 * tick
        parts.append(f'<path d="M{x} 52v188" stroke="#ddd"/>')
        parts.append(f'<text x="{x - 10}" y="266">{tick / 5:.1f}</text>')
    for i, method in enumerate(methods):
        y = 70 + 36 * i
        parts.append(f'<text x="18" y="{y + 6}">{method}</text>')
        for j, trial in enumerate(trials):
            x = 190 + 480 * trial["metrics"][method]["auroc"]
            parts.append(f'<circle cx="{x:.2f}" cy="{y + 3 * j}" r="3" fill="#075"/>')
    parts.append("</g></svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    print(figure(json.loads(Path(sys.argv[1]).read_text())))
