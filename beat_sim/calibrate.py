"""Recalibrate the body-sim caps: per-transition peak signals across a folder of favourite maps.

    python -m beat_sim.calibrate <folder of map folders> [n_maps] [out.json]

Prints p50/p90/p99 of tip speed, tip acceleration and joint-speed ratio; paste the p99s into metrics.py.
"""
import sys, json, random, numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from beat_weaver.parsers.beatmap_parser import parse_map_folder
from beat_sim import simulate
from beat_sim.body import QD_MAX, fk

def one(d):
    out=[]
    try: bms=parse_map_folder(Path(d))
    except Exception: return out
    for bm in bms:
        if bm.difficulty_info.difficulty not in ("Expert","ExpertPlus") or bm.difficulty_info.characteristic!="Standard": continue
        try: r,trajs=simulate(bm.notes, return_trajectories=True)
        except Exception as e: continue
        for tr in trajs:
            rs=tr.results; mirror=tr.hand==0
            qd=np.abs(tr.qd)/QD_MAX
            for i in range(1,len(rs)):
                a,b=rs[i-1],rs[i]; win=(tr.t>=a.swing.t)&(tr.t<=b.swing.t)
                if not win.any(): continue
                hilt=np.linalg.norm(np.gradient(np.array([fk(q,mirror)["hilt"] for q in tr.q[win]]),0.01,axis=0),axis=1).max() if win.sum()>2 else 0
                out.append((bm.difficulty_info.difficulty, b.swing.t-a.swing.t, float(tr.tip_speed[win].max()), float(tr.tip_accel[win].max()), float(qd[win].max()), float(hilt), float(b.err)))
    return out

if __name__=="__main__":
    ds=sorted(Path(sys.argv[1]).iterdir()); random.seed(0)
    ds=random.sample(ds, min(len(ds), int(sys.argv[2]) if len(sys.argv)>2 else 24))
    rows=[]
    with ProcessPoolExecutor() as ex:
        for r in ex.map(one, [str(d) for d in ds]): rows+=r
    if len(sys.argv)>3: json.dump(rows, open(sys.argv[3],"w"))
    a=np.array([r[1:] for r in rows]); print(len(rows),"transitions")
    for i,nme in enumerate(["gap","tip_speed","tip_accel","qd_ratio","hilt_speed","err"]):
        v=a[:,i]; print(f"{nme:10s} p50={np.percentile(v,50):8.2f} p90={np.percentile(v,90):8.2f} p99={np.percentile(v,99):8.2f}")
