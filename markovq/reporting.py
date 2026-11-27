"""Portable validated experiment records with canonical input fingerprints."""
from __future__ import annotations
import csv
import hashlib
import html
import io
import json
import platform
from importlib.metadata import version
from typing import Mapping

def experiment_record(*,model: str,inputs: Mapping,protocol: str,seed: int,solver: str,tolerance: float,baseline: str,runtime_s: float,peak_rss_bytes: int,limitations: str,results: Mapping) -> dict:
    import math
    if not all(isinstance(x,str) and x.strip() for x in (model,protocol,solver,baseline,limitations)) or not all(math.isfinite(x) and x>=0 for x in (runtime_s,tolerance)) or peak_rss_bytes<0:
        raise ValueError("incomplete scientific record")
    if not {"initial_state","graph","boundaries","dataset"}<=inputs.keys():
        raise ValueError("missing model inputs")
    payload=json.dumps(inputs,sort_keys=True,allow_nan=False,separators=(",",":"))
    return {"schema":1,"model":model,"inputs":json.loads(payload),"input_sha256":hashlib.sha256(payload.encode()).hexdigest(),
        "protocol":protocol,"seed":seed,"solver":solver,"tolerance":tolerance,"baseline":baseline,"runtime_s":runtime_s,
        "peak_rss_bytes":peak_rss_bytes,"limitations":limitations,"results":json.loads(json.dumps(results,allow_nan=False)),
        "versions":{"python":platform.python_version(),"numpy":version("numpy"),"scipy":version("scipy")}}

def render(record: Mapping,fmt: str="json") -> str:
    payload=json.dumps(record,sort_keys=True,allow_nan=False,indent=2)
    if fmt=="json":
        return payload
    if fmt=="html":
        return "<!doctype html><meta charset=utf-8><title>MarkovQ experiment</title><pre>"+html.escape(payload)+"</pre>"
    if fmt=="csv":
        stream=io.StringIO()
        writer=csv.writer(stream)
        writer.writerow(["field","json_value"])
        writer.writerows((key,json.dumps(record[key],sort_keys=True,allow_nan=False)) for key in sorted(record))
        return stream.getvalue()
    raise ValueError("unknown report format")
