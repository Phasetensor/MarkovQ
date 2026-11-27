"""Inspect the exact wheel and sdist produced by the scientific build."""

import tarfile
import zipfile
from email import message_from_bytes
from pathlib import Path

wheel, = Path("dist").glob("*.whl")
source, = Path("dist").glob("*.tar.gz")
with zipfile.ZipFile(wheel) as archive:
    files = archive.namelist()
    assert {"markovq/py.typed", "markovq/reporting.py", "markovq/quantum/search.py"} <= set(files)
    metadata = message_from_bytes(archive.read(next(f for f in files if f.endswith(".dist-info/METADATA"))))
    assert metadata["Name"] == "markovq" and metadata["Requires-Python"] == ">=3.9"
    assert {"interop", "references"} <= set(metadata.get_all("Provides-Extra"))
    assert any(f.endswith("/LICENSE") for f in files)
with tarfile.open(source) as archive:
    files = [f.split("/", 1)[-1] for f in archive.getnames()]
    assert {"LICENSE","CITATION.cff","benchmarks/data/crambin.csv","pyproject.toml","markovq/py.typed"} <= set(files)
