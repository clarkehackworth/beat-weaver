"""Bands for the music-following metrics: python -m beat_sim.calibrate_music <folder of map folders with audio> [n] [out.json]"""
import sys, json, random, numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from beat_weaver.parsers.beatmap_parser import parse_map_folder
from beat_sim.music import envelope, stats, find_audio

def one(d):
    out=[]
    try:
        bms=parse_map_folder(Path(d)); a=find_audio(Path(d))
        if a is None: return out
        env=envelope(a)
    except Exception as e: return out
    for bm in bms:
        if bm.difficulty_info.difficulty not in ("Expert","ExpertPlus") or bm.difficulty_info.characteristic!="Standard": continue
        n=[x for x in bm.notes if 0<=x.x<=3 and 0<=x.y<=2]
        if len(n)<100: continue
        st=stats(n, env, bm.metadata.bpm)
        st["lead_in"]=float(min(x.time_seconds for x in n)); st["diff"]=bm.difficulty_info.difficulty
        out.append(st)
    return out

if __name__=="__main__":
    ds=sorted(Path(sys.argv[1]).iterdir()); random.seed(0); ds=random.sample(ds, min(len(ds), int(sys.argv[2])))
    rows=[]
    with ProcessPoolExecutor(8) as ex:
        for r in ex.map(one,[str(d) for d in ds]): rows+=r
    json.dump(rows, open(sys.argv[3],"w"))
    for diff in ("Expert","ExpertPlus"):
        rs=[r for r in rows if r["diff"]==diff]; print(diff, len(rs))
        for k in ("energy_corr","onset_hit","lead_in"):
            v=np.array([r[k] for r in rs]); print(f"  {k:12s} p10={np.percentile(v,10):.3f} p25={np.percentile(v,25):.3f} p50={np.percentile(v,50):.3f} p90={np.percentile(v,90):.3f}")
