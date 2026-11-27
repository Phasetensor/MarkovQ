"""Run using the public PhaseTensor distributions installed by the interop job."""
import json
from datetime import datetime,timedelta,timezone

from markovq.interop import phase_experiment

def calibration():
    from calibmap import Record
    at=datetime(2026,1,5,tzinfo=timezone.utc)
    rows=[]
    for ago in (2,1,0):
        t=at-timedelta(days=ago)
        for bit in (0,1):
            for name,base in (("false_one",.02),("false_zero",.06)):
                rows.append(Record("model",name,(bit,),t,t,t,base+.01*bit+.001*ago,"probability","fixture","synthetic",sample_size=1000))
        rows.append(Record("model","gate_error",(0,1),t,t,t,.01+.001*ago,"probability","fixture","synthetic",gate="cx"))
    return rows,at

def run():
    rows,at=calibration()
    return phase_experiment(rows,at,seed=2415)

if __name__=="__main__":
    print(json.dumps(run(),allow_nan=False,sort_keys=True,indent=2))
