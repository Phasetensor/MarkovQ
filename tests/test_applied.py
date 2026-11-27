import csv
import io
import json
import numpy as np
import pytest
from numpy.testing import assert_allclose as close
from scipy import sparse
from markovq.batch import simulate_endpoints
from markovq.networks import heldout_metrics,rank_nodes
from markovq.noise import apply_channel,kraus_model,noisy_walk,readout
from markovq.quantum import hadamard
from markovq.reporting import experiment_record,render

@pytest.mark.parametrize("p",[0.,.3,1.])
def test_noise(p):
    plus=np.full((2,2),.5)
    excited=np.diag([0.,1.])
    for name,initial,ref in [
        ("bit-flip",excited,np.diag([p,1-p])),
        ("phase-flip",plus,[[.5,.5-p],[.5-p,.5]]),
        ("dephasing",plus,[[.5,.5*(1-p)],[.5*(1-p),.5]]),
        ("depolarizing",excited,(1-p)*excited+p*np.eye(2)/2),
        ("amplitude-damping",excited,np.diag([p,1-p]))]:
        close(apply_channel(initial,kraus_model(name,p)),ref,atol=1e-14)
    result=noisy_walk(hadamard(),excited,kraus_model("dephasing",p),4)
    close(result.trace_error,0,atol=1e-14)
    assert result.purity.max()<1+1e-12
    close(readout([.25,.75],[[.9,.1],[.2,.8]]),[.375,.625])

def test_ranking():
    a=[[0,1,0],[1,0,0],[0,0,0]]
    r=rank_nodes(a,[0],time=.7)
    close(r["one_step"],[0,1,0])
    close(r["restart_pagerank"],[1/1.8,.8/1.8,0])
    close(r["quantum"],[np.cos(.7)**2,np.sin(.7)**2,0],atol=1e-12)
    close(r["heat"],[(1+np.exp(-1.4))/2,(1-np.exp(-1.4))/2,0])
    metrics=heldout_metrics([1,1,1],[],[0],[1,2],2)
    close(list(metrics.values()),[.5,1/3,1/3,2/3])

def test_batch():
    p=sparse.csr_matrix([[.9,.1],[.2,.8]])
    args=(p,[1,0],8,10000)
    a=simulate_endpoints(*args,seed=33)
    b=simulate_endpoints(*args,seed=33,workers=2)
    close(a.counts,b.counts)
    theory=np.linalg.matrix_power(p.toarray(),8)[0]
    assert np.all(abs(a.counts/10000-theory)<5*np.sqrt(theory*(1-theory)/10000))
    assert a.batches==5 and a.storage_bytes>0

def test_report():
    inputs={"initial_state":[1],"graph":[[1]],"boundaries":"closed","dataset":"synthetic"}
    record=experiment_record(model="chain",inputs=inputs,protocol="endpoint",seed=1,solver="exact",tolerance=0,baseline="identity",runtime_s=.01,peak_rss_bytes=1,limitations="<finite>",results={"p":1})
    assert json.loads(render(record))==record
    assert "&lt;finite&gt;" in render(record,"html")
    rows=list(csv.reader(io.StringIO(render(record,"csv"))))
    assert {k:json.loads(v) for k,v in rows[1:]}==record
    options={k:v for k,v in record.items() if k not in {"schema","input_sha256","versions"}}
    with pytest.raises(ValueError):experiment_record(**dict(options,inputs={}))
    with pytest.raises(ValueError):experiment_record(**dict(options,model=""))

@pytest.mark.parametrize("action",[
 lambda:kraus_model("bad",.2),lambda:kraus_model("dephasing",2),
 lambda:apply_channel(np.eye(2)/2,[]),lambda:apply_channel(np.eye(2)/2,[np.eye(2)*2]),
 lambda:apply_channel(np.eye(2)/2,[np.eye(3)]),
 lambda:noisy_walk(np.eye(2),np.eye(2)/2,[np.eye(3)],2),
 lambda:rank_nodes([[0,1],[0,0]],[0]),lambda:rank_nodes(np.eye(2),[]),
 lambda:heldout_metrics([0,1],[0],[0],[1],1),lambda:heldout_metrics([0,np.nan],[],[0],[1],1),
 lambda:heldout_metrics([0,1],[],[0],[1],3),
 lambda:simulate_endpoints(np.eye(2),[0,1],1,0,seed=0),lambda:render({},"bad")])
def test_rejections(action):
    with pytest.raises(ValueError): action()
