import json,glob,os
for f in sorted(glob.glob("/data1/j50057870/world-model/runs/[ip][mp][go]_*/log.jsonl")):
    L=[json.loads(x) for x in open(f)]; n=f.split("/")[-2]; b=max(L,key=lambda r:r["gates"]-3*r["crash"])
    for t,r in (("best",b),("last",L[-1])): print(f"{n:16s} {t} it {r['it']:5d} h {r['hours']:.2f} crash {r['crash']:.3f} gates {r['gates']:.2f} lap {r['lap_time']:.2f} rate {r['lap_rate']:.2f}")
