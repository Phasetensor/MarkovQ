import csv
import html
import io
import json
import pickle
from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from markovq import hitting
from markovq.batch import endpoint_plan, merge_partitions, run_partition
from markovq.interop import estimator_report
from markovq.quantum import continuous_search, discrete_search, eventual_detection
from markovq.reporting import render, scientific_value


def test_search_underflow_and_ties():
    step = np.eye(3)[[1, 0, 2]]
    for a,b,index in [(1e-180, 2e-180,1),(0,1e-180,1),(1e-160,2e-160,1),(0,0,0),(1e-180,1e-180,0)]:
        r=discrete_search(step,[a,b,1],[0],[0,1])
        assert r.best_index == index
        if b==2e-180:
            assert r.success.argmax()==0 and r.cost_status(index)=="probability-underflow"
    # H restricted to first two sites is -X (the diagonal cancels the oracle).
    adj=np.array([[-1,1,0],[1,0,0],[0,0,0]])
    c=continuous_search(adj,1,[1e-180,2e-180,1],[0],[0,np.pi/2])
    assert c.best_index==1 and c.success.tolist()==[0,0]
    for log in [np.inf,np.nan,1,-np.inf,-4]:
        obj=discrete_search(np.eye(2),[1,0],[0],[0])._replace(log_success=np.array([log]))
        with pytest.raises(ValueError):
            obj.expected_cost(0)


@pytest.mark.parametrize("amp", [1e-12,1e-100,1e-250,0.0])
def test_defective_time(amp):
    d=eventual_detection(np.eye(2),[1,amp],np.diag([1,0]))
    assert d.conditional_mean==1
    assert d.diagnostics.unconditional_mean==(np.inf if amp else 1)
    assert d.diagnostics.unconditional_status==("infinite" if amp else "finite")


def test_unresolved_projection():
    v=np.array([1,2])/np.sqrt(5)
    d=eventual_detection(np.eye(2),v,np.outer(v,v))
    assert np.isnan(d.diagnostics.unconditional_mean)
    assert d.diagnostics.unconditional_status=="unresolved"


@pytest.mark.parametrize("value", [-.1,0,.5,1,1.01])
def test_estimator(value):
    r=estimator_report(value,.03)
    assert r["unconstrained_estimate"]==value
    assert r["domain_status"]==("inside" if 0<=value<=1 else "outside")


def test_integrity():
    m=sparse.eye(2,format="csr")
    x=np.array([1.,0])
    plan=endpoint_plan(m,x,1,10,seed=3)
    m.data[:]=0
    x[:]=0
    copy=pickle.loads(pickle.dumps(plan))
    part=run_partition(copy,0)
    assert part.counts.tolist()==[10,0]
    with pytest.raises(ValueError):
        run_partition(replace(plan,fingerprint="wrong"),0)
    plan.transition.indices=np.array([1,0])
    for action in (lambda:run_partition(plan,0),lambda:merge_partitions(plan,[part])):
        with pytest.raises(ValueError):
            action()


def test_export_statuses():
    h=hitting(np.eye(2),[0])
    d=eventual_detection(np.eye(2),[1,1e-220],np.diag([1,0]))
    v=np.array([1,2])/np.sqrt(5)
    u=eventual_detection(np.eye(2),v,np.outer(v,v))
    search=discrete_search(np.eye(2),[1e-180,1],[0],[0])
    cost=search.expected_cost(0)["independent_trials"]
    result=dict(h=h,d=d,u=u,missing=scientific_value(None,"missing","no observation"),o=scientific_value(cost,"overflow",search.resource_status(0)["independent_trials"]),nested=[np.array([np.nan,np.float32(2)])])
    data=json.loads(render(result),parse_constant=lambda x:pytest.fail(x))
    assert data["h"]["mean"][1]["status"]=="infinite"
    assert data["h"]["conditional_mean"][1]["status"]=="undefined"
    assert data["d"]["diagnostics"]["unconditional_mean"]["status"]=="infinite"
    assert data["u"]["diagnostics"]["unconditional_mean"]["status"]=="unresolved"
    assert data["nested"][0][0]["status"]=="unexpected"
    assert data["o"]["status"]=="overflow"
    assert json.loads(html.unescape(render(result,"html").split("<pre>")[1].split("</pre>")[0]))==data
    assert {k:json.loads(v) for k,v in list(csv.reader(io.StringIO(render(result,"csv"))))[1:]}==data
