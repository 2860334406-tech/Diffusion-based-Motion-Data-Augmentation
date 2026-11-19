"""Parameter grid for trajectory-level waist augmentation anchors.

Emits ``ClipSpec`` objects whose only job is to cover the requested joint ranges
(see the project's "expected data" table). The physics envelope measured on the
existing G1 controller bounds what is actually achievable:

  shoulder abduction   up to 55 deg
  knee flexion         18 deg / 33 deg
  toe-out              up to 8 deg
  hip yaw              up to 7 deg
  entry speed          <= ~17 deg/s (waist RMSE gate 0.04 rad)

Anything faster fails ``waist_tracking`` long before the effort gate; 65 deg/s
destabilises the simulation outright. Speed tiers are therefore capped at 17 deg/s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from smp.data.shoulder_ik import ClipSpec

# Segment length the diffusion model uses (30 fps). Measured anchor trajectories
# span 99-240 frames (median 174), so 160 keeps most of them fully covered while
# still holding a 4.3 s entry plus a short hold.
HORIZON_FRAMES = 160
FPS = 30
MAX_ENTRY_S = 4.3
HOLD_S = (1.2, 1.8)

STANCE_WIDTHS = (0.22, 0.24, 0.26)
KNEE_FLEXIONS = (0.32, 0.58)           # 18 deg / 33 deg
ARM_VARIANTS = (0.26, 0.55, 0.96)      # 15 / 31.5 / 55 deg abduction
WRIST_ROLLS = (-0.10, 0.0, 0.10)       # +-5.7 deg
TOE_OUTS = (-0.12, 0.0, 0.12)          # +-6.9 deg
HIP_YAWS = (-0.10, 0.0, 0.10)          # +-5.7 deg
SPEED_TIERS = (9.0, 13.0, 17.0)        # deg/s; 17 is the measured ceiling
YAW_RANGE_DEG = (15.0, 70.0)

TARGET_PER_FAMILY = 200                # 6 families x 200 = 1200 specs
BASE_PER_FAMILY = 200


@dataclass(frozen=True)
class MeshPlan:
  """Counts what the grid will emit; printed before a long generation run."""

  specs: list[ClipSpec]

  @property
  def n(self) -> int:
    return len(self.specs)

  def summary(self) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in self.specs:
      key = f"c{s.condition_id}-roll{round(math.degrees(s.shoulder_roll_rad))}"
      out[key] = out.get(key, 0) + 1
    return out


def _latin_hypercube(count: int, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
  """Stratified 1D sample in [low, high]."""
  edges = np.linspace(0.0, 1.0, count + 1)
  u = edges[:-1] + rng.random(count) * (edges[1:] - edges[:-1])
  return low + u * (high - low)


def build_specs(
  seed: int = 42,
  per_family: int = BASE_PER_FAMILY,
  stage: str = "formal",
) -> MeshPlan:
  """Build the deterministic anchor grid.

  Families = (stance width x knee flexion); inside each family we stratify over
  waist amplitude and cycle speed tier / abduction / wrist roll / toe-out /
  hip-yaw so every requested dimension gets coverage without a full cross product.
  """
  rng = np.random.default_rng(seed)
  specs: list[ClipSpec] = []
  yaw_low, yaw_high = YAW_RANGE_DEG

  for stance_index, width in enumerate(STANCE_WIDTHS):
    for knee_index, knee in enumerate(KNEE_FLEXIONS):
      cell = stance_index * len(KNEE_FLEXIONS) + knee_index
      amps = _latin_hypercube(per_family, yaw_low, yaw_high, rng)
      # Shuffle direction so the two sides are balanced inside every family.
      signs = np.array([1, -1] * ((per_family + 1) // 2))[:per_family]
      rng.shuffle(signs)

      for slot in range(per_family):
        condition_id = 1 if signs[slot] > 0 else 2
        speed = SPEED_TIERS[slot % len(SPEED_TIERS)]
        entry = min(amps[slot] / speed, MAX_ENTRY_S)
        # The realised speed is limited by the horizon, not the tier.
        realised_speed = amps[slot] / entry
        abduct = ARM_VARIANTS[(slot // len(SPEED_TIERS)) % len(ARM_VARIANTS)]
        wrist = WRIST_ROLLS[slot % len(WRIST_ROLLS)]
        toe = TOE_OUTS[(slot + stance_index) % len(TOE_OUTS)]
        hip_yaw = HIP_YAWS[(slot + knee_index) % len(HIP_YAWS)]
        arm_jitter = tuple(float(x) for x in rng.uniform(-1.0, 1.0, 4))
        family = f"{stage}-s{stance_index}k{knee_index}"

        specs.append(
          ClipSpec(
            clip_id=f"{family}-a{slot:03d}",
            family_id=family,
            stage=stage,
            split="train",
            condition_id=condition_id,
            grid_index=cell,
            slot_index=slot,
            attempt=0,
            seed=seed,
            yaw_amplitude=math.radians(amps[slot]),
            entry_duration_s=float(entry),
            hold_duration_s=float(rng.uniform(*HOLD_S)),
            stance_width_m=width,
            knee_flexion_rad=knee,
            arm_variant=arm_jitter,
            com_offset_y_m=float(rng.uniform(-0.008, 0.008)),
            target_mean_yaw_speed_deg_s=float(realised_speed),
            yaw_min_deg=yaw_low,
            yaw_max_deg=yaw_high,
            shoulder_roll_rad=abduct,
            shoulder_yaw_rad=float(rng.uniform(-0.10, 0.10)),
            shoulder_pitch_rad=0.20 + float(rng.uniform(-0.09, 0.09)),
            elbow_rad=0.60 + float(rng.uniform(-0.17, 0.17)),
            wrist_roll_rad=wrist,
            wrist_pitch_rad=float(rng.uniform(-0.07, 0.07)),
            wrist_yaw_rad=float(rng.uniform(-0.07, 0.07)),
            hip_yaw_rad=hip_yaw,
            waist_pitch_rad=float(rng.uniform(-0.035, 0.035)),
            waist_roll_rad=float(rng.uniform(-0.035, 0.035)),
            toe_out_rad=toe,
          )
        )

  return MeshPlan(specs=specs)


def build_stand_specs(seed: int = 42, count: int = 36, stage: str = "formal") -> list[ClipSpec]:
  """Standing baselines matching the same arm/knee grid, no waist motion."""
  rng = np.random.default_rng(seed + 1)
  specs: list[ClipSpec] = []
  for i in range(count):
    stance_index = i % len(STANCE_WIDTHS)
    knee_index = (i // len(STANCE_WIDTHS)) % len(KNEE_FLEXIONS)
    abduct = ARM_VARIANTS[i % len(ARM_VARIANTS)]
    specs.append(
      ClipSpec(
        clip_id=f"{stage}-stand-s{stance_index}k{knee_index}-{i:03d}",
        family_id=f"{stage}-stand-s{stance_index}k{knee_index}",
        stage=stage,
        split="train",
        condition_id=0,
        grid_index=stance_index * len(KNEE_FLEXIONS) + knee_index,
        slot_index=i,
        attempt=0,
        seed=seed,
        yaw_amplitude=0.0,
        entry_duration_s=0.0,
        hold_duration_s=3.0,
        stance_width_m=STANCE_WIDTHS[stance_index],
        knee_flexion_rad=KNEE_FLEXIONS[knee_index],
        arm_variant=tuple(float(x) for x in rng.uniform(-1.0, 1.0, 4)),
        com_offset_y_m=float(rng.uniform(-0.008, 0.008)),
        shoulder_roll_rad=abduct,
        shoulder_yaw_rad=float(rng.uniform(-0.10, 0.10)),
        wrist_roll_rad=float(rng.uniform(-0.10, 0.10)),
        toe_out_rad=float(rng.uniform(-0.12, 0.12)),
      )
    )
  return specs