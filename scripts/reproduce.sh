#!/bin/sh
# Train the five fixed presets and record one episode from each.
set -eu
cd "$(dirname "$0")/.."
output=${1:-results}
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
export JAX_ENABLE_X64=false
uv sync --locked
run() {
    uv run --locked python demo.py --env "$1" --seed 0 \
        --n-envs "${N_ENVS:-64}" --updates "${UPDATES:-5000}" \
        --lr-policy "$2" --lr-value "$3" --beta "$4" \
        --max-steps "${MAX_STEPS:-5000}" --output "$output"
}
run breakout 0.003 0.001 0.04
run asterix 0.002 0.002 0.07
run freeway 0.002 0.0001 0.01
run space_invaders 0.002 0.0015 0.07
run seaquest 0.002 0.0004 0.03
