#!/usr/bin/env bash
# Give MPI rank r of the node only the r-th visible GPU (the SLATE adapter needs one device per rank):
#   mpirun -np 8 reproducers/references/one-gpu-per-rank.sh build/bin/qr_reference_slate ...
set -euo pipefail
rank=${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-0}}
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=, read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES="${devices[$rank]}"
else
    export CUDA_VISIBLE_DEVICES="$rank"
fi
exec "$@"
