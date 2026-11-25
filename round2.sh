#!/usr/bin/env bash
cd ~/projects/world-model; R=/data1/j50057870/world-model/runs; PY=${PY:-python}
docker wait wm-colon0 wm-colon1 >/dev/null 2>&1; [ -f /data1/j50057870/world-model/data/on0/ok ] && [ -f /data1/j50057870/world-model/data/on1/ok ] || { echo collection failed; exit 1; }
docker wait wm-img_m >/dev/null 2>&1; sleep 20
./run.sh wm_s2 4 $PY train_wm.py --size s --init $R/wm_s/last.pt --data '[to][rn][0-9]' --lr 1.5e-4 --steps 10000 --out $R/wm_s2 >/dev/null
docker wait wm-wm_s2 >/dev/null 2>&1; grep EVAL $R/wm_s2.log | tail -1 | cut -c1-700
./run.sh img_s2 4 $PY imagine.py --wm $R/wm_s2/last.pt --bc $R/img_s/best.pt --data '[to][rn][0-9]' --out $R/img_s2 --iters 100000 --hours 2 --kl .005 >/dev/null
docker wait wm-img_s2; tail -1 $R/img_s2/log.jsonl
