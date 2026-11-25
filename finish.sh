#!/usr/bin/env bash
# runs in tmux wm-finish: matched round-2 policy runs, then held-out test and videos
cd ~/projects/world-model; R=/data1/j50057870/world-model/runs; PY=${PY:-python}
go(){ n=$1; g=$2; shift 2; [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 500 ] || { echo "$n: GPU $g busy, skipped"; return; }
  [ -d $R/$n ] && { echo "$n exists, skipped"; return; }; ./run.sh $n $g $PY imagine.py --out $R/$n --iters 100000 --hours 2 "$@"; }
go img_s2 4 --wm $R/wm_s2/last.pt --bc $R/img_s/best.pt --data '[to][rn][0-9]' --kl .005
go img_s_cont 5 --wm $R/wm_s/last.pt --bc $R/img_s/best.pt --kl .005
go img_s2_kl02 3 --wm $R/wm_s2/last.pt --bc $R/img_s/best.pt --data '[to][rn][0-9]' --kl .02
go ppo_real_kl005 2 --real 1 --bc $R/bc0/bc.pt --kl .005
go ppo_real_kl005_s1 6 --real 1 --bc $R/bc0/bc.pt --kl .005 --seed 1
for n in img_s2 img_s_cont img_s2_kl02 ppo_real_kl005 ppo_real_kl005_s1; do docker wait wm-$n 2>/dev/null; done
./run.sh final 7 $PY final_test.py; docker wait wm-final
for n in img_s2 img_l ppo_real; do ./run.sh film_$n 2 $PY film.py $R/$n/best.pt $R/film_$n.mp4; docker wait wm-film_$n; done
echo "finish.sh complete"
