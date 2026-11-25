#!/usr/bin/env bash
# usage: run.sh NAME GPUS cmd...   (detached container on host 37, project at /workspace; log in /data1/.../runs/NAME.log)
set -euo pipefail
NAME=$1; GPUS=$2; shift 2
D=/data1/j50057870/world-model
docker run -d --rm --name "wm-$NAME" --gpus "\"device=$GPUS\"" --ipc=host --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e USER=wm -e HF_HOME=$D/hf -e TORCH_HOME=$D/hf/torch -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=4 ${DENV:-} -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$HOME/projects/world-model:/workspace" -v $D:$D -w /workspace "${WM_IMAGE:?set WM_IMAGE}" \
  bash -c 'L=$1; shift; "$@" > "$L" 2>&1; echo "exit $?" >> "$L"' _ "$D/runs/$NAME.log" "$@"
