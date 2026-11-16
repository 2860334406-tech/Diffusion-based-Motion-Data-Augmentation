#!/usr/bin/env bash
set -euo pipefail

policy_bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
policy_repo_root="$(cd -- "${policy_bundle_dir}/../.." && pwd)"

(
  cd -- "${policy_bundle_dir}"
  sha256sum --check SHA256SUMS
)

cd -- "${policy_repo_root}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

exec uv run --locked python scripts/evaluate_force_unloading.py \
  --checkpoint "${policy_bundle_dir}/policy.pt" \
  --prior-checkpoint "${policy_bundle_dir}/prior.pt" \
  --output "${policy_repo_root}/logs/reproduction/policy_handoff/chest_speed_gradient_v1/evaluation.json" \
  --task-id Smp-Force-Unloading-G1 \
  --stage c \
  --stationary-reward-weight 0.1 \
  --gsi-max-root-planar-speed 0.15 \
  --stationary-speed-scale 10.0 \
  --sphere-radius 0.10 \
  --target-probabilities 1.0 0.0 0.0 \
  --push-direction-mode front_to_back \
  --num-episodes "${POLICY_NUM_EPISODES:-1024}" \
  --num-envs "${POLICY_NUM_ENVS:-256}" \
  --seed 42 \
  --device "${POLICY_DEVICE:-cuda:0}" \
  --reference-speed 0.20 \
  --max-force-weight-ratio 0.15 \
  --ramp-time 0.25 \
  --wait-range-s 1.0 3.0 \
  --max-steps 20000 \
  --video-length 600 \
  --video-folder "${policy_repo_root}/logs/reproduction/policy_handoff/chest_speed_gradient_v1/videos" \
  --video-height 720 \
  --video-width 1280
