#!/usr/bin/env bash
# queue: each line waits for its predecessor container, then starts the next job on that GPU (only if the GPU is empty)
cd ~/projects/world-model; R=/data1/j50057870/world-model/runs; PY=${PY:-python}
w(){ docker wait "wm-$1" >/dev/null 2>&1; sleep 15; }
go(){ n=$1; g=$2; shift 2
  for k in $(seq 120); do [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 500 ] && break; sleep 30; done
  [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 500 ] || { echo "$n: GPU $g busy, skipped"; return; }
  ./run.sh $n $g $PY imagine.py --bc $R/bc0/bc.pt --out $R/$n --iters 100000 "$@" && docker wait wm-$n; }
( w wm_m; go img_m 4 --wm $R/wm_m/last.pt --hours 2 ) &
( w wm_l; go img_l 5 --wm $R/wm_l/last.pt --hours 2 ) &
( w wm_xl; go img_xl 6 --wm $R/wm_xl/last.pt --hours 2 ) &
( w wm_xl; go img_s_kl005 7 --wm $R/wm_s/last.pt --hours 2 --kl .005 ) &
( w ppo_real; go ppo_real_kl005 2 --real 1 --hours 2 --kl .005 ) &
( w img_s; go img_s_seed1 3 --wm $R/wm_s/last.pt --hours 2 --seed 1 ) &
wait; echo chain finished
