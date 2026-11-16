"""Run the low-speed-prior force-unloading curriculum from a scratch policy.

The curriculum is intentionally split into independent training runs.  Stage A
starts with a fresh actor, critic, normalizers, and optimizer; later stages
resume the complete checkpoint produced by the immediately preceding stage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

import torch
import tyro
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "Smp-Force-Unloading-G1"
EXPECTED_PRIOR_SHA256 = (
  "5574dc5720a356937f6a9bc01b77f3d5e8456970d59e59c59b05e2ca6a92d0b7"
)
_WANDB_URL_RE = re.compile(r"https://wandb\.ai/[^\s]+/runs/[A-Za-z0-9]+")
_ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_CHECKPOINT_RE = re.compile(r"model_(\d+)\.pt$")


@dataclass(frozen=True)
class StageSpec:
  key: str
  label: str
  iterations: int
  reference_speed: float
  max_force_weight_ratio: float
  ramp_time: float
  wait_range_s: tuple[float, float]
  stationary_reward_weight: float
  evaluation_stage: str


STAGE_A = StageSpec(
  key="a",
  label="stand",
  iterations=3000,
  reference_speed=0.05,
  max_force_weight_ratio=0.15,
  ramp_time=0.25,
  wait_range_s=(1.0e9, 1.0e9),
  stationary_reward_weight=0.9,
  evaluation_stage="a",
)
STAGE_A_EXTENSION = StageSpec(
  key="a_extension",
  label="stand_extension",
  iterations=2000,
  reference_speed=0.05,
  max_force_weight_ratio=0.15,
  ramp_time=0.25,
  wait_range_s=(1.0e9, 1.0e9),
  stationary_reward_weight=0.9,
  evaluation_stage="a",
)
CONTACT_STAGES = (
  StageSpec(
    key="b1",
    label="gentle_contact",
    iterations=4000,
    reference_speed=0.01,
    max_force_weight_ratio=0.05,
    ramp_time=1.0,
    wait_range_s=(1.0, 3.0),
    stationary_reward_weight=0.1,
    evaluation_stage="b1",
  ),
  StageSpec(
    key="b2",
    label="medium_contact",
    iterations=4000,
    reference_speed=0.03,
    max_force_weight_ratio=0.10,
    ramp_time=0.5,
    wait_range_s=(1.0, 3.0),
    stationary_reward_weight=0.1,
    evaluation_stage="b2",
  ),
  StageSpec(
    key="c",
    label="nominal_chest_push",
    iterations=19000,
    reference_speed=0.05,
    max_force_weight_ratio=0.15,
    ramp_time=0.25,
    wait_range_s=(1.0, 3.0),
    stationary_reward_weight=0.1,
    evaluation_stage="c",
  ),
)


@dataclass(frozen=True)
class CurriculumCfg:
  task_id: str = TASK_ID
  all_contact_evaluation: bool = False
  prior_checkpoint: str = "datasets/pretrain_ckpt/pretrained_lafan_low_speed_balance.pt"
  prior_sha256: str = EXPECTED_PRIOR_SHA256
  conditional_score_calibration: str = ""
  seed: int = 42
  gpu_id: int = 0
  num_envs: int = 4096
  smoke_num_envs: int = 64
  evaluation_num_envs: int = 256
  evaluation_num_episodes: int = 1024
  action_std: float = 0.10
  learning_rate: float = 1.0e-4
  save_interval: int = 1000
  gsi_max_root_planar_speed: float = 0.5
  stationary_speed_scale: float = 1.0
  sphere_radius: float = 0.175
  contact_reference_speeds: tuple[float, float, float] = (0.01, 0.03, 0.05)
  target_probabilities: tuple[float, ...] = (1.0, 0.0, 0.0)
  push_direction_mode: Literal["target_default", "front_to_back"] = "target_default"
  parent_stage_a_manifest: str = ""
  experiment_name: str = "smp_force_unloading_g1"
  wandb_project: str = "smp"
  run_prefix: str = "lafan_scratch_curriculum"
  log_root: str = "logs/reproduction/rsl_rl"
  state_dir: str = "logs/reproduction/curriculum/lafan_low_speed_scratch_seed42"
  evaluation_dir: str = "logs/reproduction/evaluation"
  run_smoke: bool = True
  shoulder_split_evaluation: bool = False
  smoke_only: bool = False
  resume_manifest: bool = True
  record_final_video: bool = True
  final_video_length: int = 600
  dry_run: bool = False


@dataclass(frozen=True)
class ResumeSource:
  run_dir: Path
  checkpoint: Path


def curriculum_stages() -> tuple[StageSpec, ...]:
  return (STAGE_A, STAGE_A_EXTENSION, *CONTACT_STAGES)


def contact_stages_for_cfg(cfg: CurriculumCfg) -> tuple[StageSpec, ...]:
  if cfg.sphere_radius <= 0.0:
    raise ValueError("sphere_radius must be positive")
  if any(speed <= 0.0 for speed in cfg.contact_reference_speeds):
    raise ValueError("contact_reference_speeds must all be positive")
  return tuple(
    replace(stage, reference_speed=speed)
    for stage, speed in zip(CONTACT_STAGES, cfg.contact_reference_speeds, strict=True)
  )


def should_extend_stage_a(report: dict[str, Any]) -> bool:
  return not bool(report["gate"]["passed"])


def _utc_now() -> str:
  return datetime.now(timezone.utc).isoformat()


def _safe_print(*args: Any, **kwargs: Any) -> bool:
  """Print while allowing a detached parent terminal to disappear safely."""
  try:
    print(*args, **kwargs)
  except BrokenPipeError:
    return False
  return True


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  os.replace(temporary, path)


def _iter_tensors(value: Any) -> Iterator[torch.Tensor]:
  if isinstance(value, torch.Tensor):
    yield value
  elif isinstance(value, dict):
    for item in value.values():
      yield from _iter_tensors(item)
  elif isinstance(value, (tuple, list)):
    for item in value:
      yield from _iter_tensors(item)


def validate_checkpoint(path: Path) -> int:
  if not path.is_file() or path.stat().st_size == 0:
    raise RuntimeError(f"Missing or empty checkpoint: {path}")
  checkpoint = torch.load(path, map_location="cpu", weights_only=True)
  required = {
    "actor_state_dict",
    "critic_state_dict",
    "optimizer_state_dict",
    "iter",
  }
  missing = required.difference(checkpoint)
  if missing:
    raise RuntimeError(f"Checkpoint {path} is missing keys: {sorted(missing)}")
  for tensor in _iter_tensors(
    {
      "actor": checkpoint["actor_state_dict"],
      "critic": checkpoint["critic_state_dict"],
      "optimizer": checkpoint["optimizer_state_dict"],
    }
  ):
    if not torch.isfinite(tensor).all():
      raise RuntimeError(f"Checkpoint contains NaN/Inf: {path}")
  infos = checkpoint.get("infos", {})
  if "smp_prior" in infos:
    if "smp_score" in infos:
      score = infos["smp_score"]
      required_score = {"mode", "calibration_sha256", "prior_sha256", "fixed_timesteps"}
      if required_score.difference(score):
        raise RuntimeError(f"Incomplete frozen conditional SMP score identity: {path}")
      if "smp_normalizer" in infos:
        raise RuntimeError(
          f"V3 checkpoint unexpectedly contains an online normalizer: {path}"
        )
      if score["prior_sha256"] != infos["smp_prior"]["sha256"]:
        raise RuntimeError(f"V3 checkpoint score/Prior identity mismatch: {path}")
    else:
      normalizer = infos.get("smp_normalizer", {})
      if set(("mean", "count", "max_count")).difference(normalizer):
        raise RuntimeError(f"Missing conditional SMP normalization state: {path}")
      if (
        normalizer["mean"].shape != (6, 50)
        or normalizer["count"].shape != (6, 50)
        or not torch.isfinite(normalizer["mean"]).all()
        or (normalizer["count"] < 0).any()
      ):
        raise RuntimeError(f"Corrupt conditional SMP normalization state: {path}")
  return int(checkpoint["iter"])


def find_latest_checkpoint(run_dir: Path) -> Path:
  candidates: list[tuple[int, Path]] = []
  for path in run_dir.glob("model_*.pt"):
    match = _CHECKPOINT_RE.fullmatch(path.name)
    if match:
      candidates.append((int(match.group(1)), path))
  if not candidates:
    raise RuntimeError(f"No model checkpoint found in {run_dir}")
  return max(candidates, key=lambda item: item[0])[1]


def expected_terminal_iteration(stage: StageSpec, resume_iteration: int | None) -> int:
  start_iteration = 0 if resume_iteration is None else resume_iteration
  return start_iteration + stage.iterations - 1


def remaining_iterations(target_iteration: int, checkpoint_iteration: int) -> int:
  if checkpoint_iteration > target_iteration:
    raise ValueError(
      f"Checkpoint iteration {checkpoint_iteration} exceeds target {target_iteration}"
    )
  return target_iteration - checkpoint_iteration + 1


def is_recoverable_interruption(record: dict[str, Any]) -> bool:
  if record.get("status") != "failed":
    return False
  if record.get("verified_recoverable_interruption") is True:
    return True
  error = str(record.get("error", "")).lower()
  recoverable_markers = (
    "broken pipe",
    "exit code -2",
    "sigint",
    "keyboardinterrupt",
    "exit code -15",
    "sigterm",
  )
  return any(marker in error for marker in recoverable_markers)


def gpu_identity(gpu_id: int) -> dict[str, Any]:
  """Resolve a physical GPU index without initializing CUDA in this process."""
  if gpu_id < 0:
    raise ValueError(f"gpu_id must be non-negative, got {gpu_id}")
  result = subprocess.run(
    [
      "nvidia-smi",
      "--query-gpu=index,uuid,name",
      "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
  )
  for line in result.stdout.splitlines():
    index_text, uuid, name = (part.strip() for part in line.split(",", maxsplit=2))
    if int(index_text) == gpu_id:
      return {"physical_index": gpu_id, "uuid": uuid, "name": name}
  raise RuntimeError(f"Physical GPU index {gpu_id} is not available")


def _find_latest_stage_checkpoint(
  cfg: CurriculumCfg,
  stage: StageSpec,
  experiment_root: Path,
) -> tuple[ResumeSource, int] | None:
  base_name = f"{cfg.run_prefix}_{stage.key}"
  normal_suffix = f"_{base_name}_seed{cfg.seed}"
  recovery_marker = f"_{base_name}_resume_from_"
  seed_suffix = f"_seed{cfg.seed}"
  candidates: list[tuple[int, Path, Path]] = []
  for run_dir in experiment_root.iterdir():
    if not run_dir.is_dir():
      continue
    run_name = run_dir.name
    if not (
      run_name.endswith(normal_suffix)
      or (recovery_marker in run_name and run_name.endswith(seed_suffix))
    ):
      continue
    try:
      checkpoint = find_latest_checkpoint(run_dir)
    except RuntimeError:
      continue
    match = _CHECKPOINT_RE.fullmatch(checkpoint.name)
    assert match is not None
    candidates.append((int(match.group(1)), run_dir, checkpoint))
  if not candidates:
    return None
  filename_iteration, run_dir, checkpoint = max(candidates, key=lambda item: item[0])
  checkpoint_iteration = validate_checkpoint(checkpoint)
  if checkpoint_iteration != filename_iteration:
    raise RuntimeError(
      f"Checkpoint iteration mismatch for {checkpoint}: "
      f"filename={filename_iteration}, payload={checkpoint_iteration}"
    )
  return ResumeSource(run_dir, checkpoint), checkpoint_iteration


def _assert_finite_training_log(run_dir: Path) -> None:
  event_files = sorted(run_dir.glob("events.out.tfevents.*"))
  if not event_files:
    raise RuntimeError(f"No TensorBoard event file found in {run_dir}")
  accumulator = EventAccumulator(str(event_files[-1]))
  accumulator.Reload()
  for tag in accumulator.Tags().get("scalars", []):
    values = accumulator.Scalars(tag)
    if any(not math.isfinite(event.value) for event in values):
      raise RuntimeError(f"Non-finite scalar in {run_dir}: {tag}")
  nan_tag = "Episode_Termination/nan_or_inf"
  if nan_tag in accumulator.Tags().get("scalars", []):
    if max(event.value for event in accumulator.Scalars(nan_tag)) > 0.0:
      raise RuntimeError(f"NaN/Inf termination was observed in {run_dir}")


def build_train_command(
  cfg: CurriculumCfg,
  stage: StageSpec,
  run_name: str,
  resume: ResumeSource | None,
  *,
  smoke: bool = False,
) -> list[str]:
  num_envs = cfg.smoke_num_envs if smoke else cfg.num_envs
  iterations = 2 if smoke else stage.iterations
  save_interval = 1 if smoke else cfg.save_interval
  logger = "tensorboard" if smoke else "wandb"
  rail_prefix = "--env.events.rail-pusher-controller.params.rail-cfg."
  init_prefix = "--env.events.init-smp-state.params."
  command = [
    sys.executable,
    "scripts/train.py",
    cfg.task_id,
    "--env.scene.num-envs",
    str(num_envs),
    "--env.seed",
    str(cfg.seed),
    init_prefix + "ckpt-path",
    cfg.prior_checkpoint,
    init_prefix + "gsi-max-root-planar-speed",
    str(cfg.gsi_max_root_planar_speed),
    rail_prefix + "sphere-radius",
    str(cfg.sphere_radius),
    rail_prefix + "reference-speed",
    str(stage.reference_speed),
    rail_prefix + "max-force-weight-ratio",
    str(stage.max_force_weight_ratio),
    rail_prefix + "ramp-time",
    str(stage.ramp_time),
    rail_prefix + "wait-range-s",
    f"({stage.wait_range_s[0]:g},{stage.wait_range_s[1]:g})",
    rail_prefix + "stationary-reward-weight",
    str(stage.stationary_reward_weight),
    rail_prefix + "stationary-speed-scale",
    str(cfg.stationary_speed_scale),
    rail_prefix + "target-probabilities",
    "(" + ",".join(f"{p:g}" for p in cfg.target_probabilities) + ")",
    rail_prefix + "push-direction-mode",
    cfg.push_direction_mode,
    "--agent.max-iterations",
    str(iterations),
    "--agent.save-interval",
    str(save_interval),
    "--agent.logger",
    logger,
    "--agent.experiment-name",
    cfg.experiment_name,
    "--agent.run-name",
    run_name,
    "--agent.actor.distribution-cfg.init-std",
    str(cfg.action_std),
    "--agent.algorithm.learning-rate",
    str(cfg.learning_rate),
    "--agent.upload-model",
    "False",
    "--log-root",
    cfg.log_root,
    "--enable-nan-guard",
    "True",
  ]
  if cfg.conditional_score_calibration:
    command.extend(
      [
        init_prefix + "conditional-score-calibration-path",
        cfg.conditional_score_calibration,
      ]
    )
  if smoke:
    command.extend(
      [
        init_prefix + "gsi-buffer-size",
        "64",
        init_prefix + "gsi-batch-size",
        "64",
        init_prefix + "compile-model",
        "False",
      ]
    )
  else:
    command.extend(["--agent.wandb-project", cfg.wandb_project])
  if resume is None:
    command.extend(["--agent.resume", "False"])
  else:
    command.extend(
      [
        "--agent.resume",
        "True",
        "--agent.load-run",
        resume.run_dir.name,
        "--agent.load-checkpoint",
        resume.checkpoint.name,
      ]
    )
  return command


def build_evaluation_command(
  cfg: CurriculumCfg,
  stage: StageSpec,
  checkpoint: Path,
  output: Path,
  *,
  video_folder: Path | None = None,
  target_probabilities: tuple[float, ...] | None = None,
) -> list[str]:
  eval_targets = target_probabilities or cfg.target_probabilities
  command = [
    sys.executable,
    "scripts/evaluate_force_unloading.py",
    "--task-id",
    cfg.task_id,
    "--stationary-reward-weight",
    str(stage.stationary_reward_weight),
    "--checkpoint",
    str(checkpoint),
    "--output",
    str(output),
    "--stage",
    stage.evaluation_stage,
    "--prior-checkpoint",
    cfg.prior_checkpoint,
    "--gsi-max-root-planar-speed",
    str(cfg.gsi_max_root_planar_speed),
    "--stationary-speed-scale",
    str(cfg.stationary_speed_scale),
    "--sphere-radius",
    str(cfg.sphere_radius),
    "--target-probabilities",
    *(f"{p:g}" for p in eval_targets),
    "--push-direction-mode",
    cfg.push_direction_mode,
    "--num-episodes",
    str(cfg.evaluation_num_episodes),
    "--num-envs",
    str(cfg.evaluation_num_envs),
    "--seed",
    str(cfg.seed),
    "--reference-speed",
    str(stage.reference_speed),
    "--max-force-weight-ratio",
    str(stage.max_force_weight_ratio),
    "--ramp-time",
    str(stage.ramp_time),
    "--wait-range-s",
    str(stage.wait_range_s[0]),
    str(stage.wait_range_s[1]),
  ]
  if cfg.conditional_score_calibration:
    command.extend(
      [
        "--conditional-score-calibration",
        cfg.conditional_score_calibration,
      ]
    )
  if video_folder is not None:
    command.extend(
      [
        "--video",
        "--video-length",
        str(cfg.final_video_length),
        "--video-folder",
        str(video_folder),
      ]
    )
  return command


def _run_process(
  command: list[str],
  log_path: Path,
  *,
  dry_run: bool,
  on_start: Any | None = None,
) -> str | None:
  rendered = shlex.join(command)
  mirror_to_console = _safe_print(f"[CURRICULUM] {rendered}", flush=True)
  log_path.parent.mkdir(parents=True, exist_ok=True)
  with log_path.open("a", encoding="utf-8") as log:
    log.write(f"\n[{_utc_now()}] {rendered}\n")
    log.flush()
    if dry_run:
      return None
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
      command,
      cwd=REPO_ROOT,
      env=env,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      bufsize=1,
    )
    if on_start is not None:
      on_start(process.pid)
    wandb_url: str | None = None
    assert process.stdout is not None
    try:
      for line in process.stdout:
        if mirror_to_console:
          mirror_to_console = _safe_print(line, end="", flush=True)
        log.write(line)
        cleaned = _ANSI_RE.sub("", line)
        match = _WANDB_URL_RE.search(cleaned)
        if match:
          wandb_url = match.group(0)
      return_code = process.wait()
    except KeyboardInterrupt as exc:
      process.send_signal(signal.SIGINT)
      try:
        remaining, _ = process.communicate(timeout=60)
      except subprocess.TimeoutExpired:
        process.terminate()
        try:
          remaining, _ = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
          process.kill()
          remaining, _ = process.communicate()
      if remaining:
        log.write(remaining)
      log.flush()
      raise RuntimeError(
        f"Command interrupted by SIGINT (exit code -2): {rendered}"
      ) from exc
    log.flush()
  if return_code != 0:
    raise RuntimeError(f"Command failed with exit code {return_code}: {rendered}")
  return wandb_url


def _find_new_run_dir(
  experiment_root: Path,
  run_name: str,
  before: set[Path],
) -> Path:
  candidates = set(experiment_root.glob(f"*_{run_name}"))
  created = candidates.difference(before)
  pool = created or candidates
  if not pool:
    raise RuntimeError(f"Could not locate output directory for run {run_name}")
  return max(pool, key=lambda path: path.stat().st_mtime_ns)


def _new_manifest(cfg: CurriculumCfg, prior: Path, prior_sha: str) -> dict[str, Any]:
  return {
    "schema_version": 1,
    "status": "running",
    "created_at_utc": _utc_now(),
    "updated_at_utc": _utc_now(),
    "task_id": cfg.task_id,
    "prior": {"path": str(prior), "sha256": prior_sha},
    "config": asdict(cfg),
    "scratch_policy": {
      "actor": True,
      "critic": True,
      "observation_normalizers": True,
      "optimizer": True,
      "forward_checkpoint_loaded": False,
    },
    "stages": {},
  }


def _load_or_create_manifest(
  cfg: CurriculumCfg,
  manifest_path: Path,
  prior: Path,
  prior_sha: str,
) -> dict[str, Any]:
  if manifest_path.exists() and cfg.resume_manifest:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["prior"]["sha256"] != prior_sha:
      raise RuntimeError("Manifest prior SHA-256 does not match current prior")
    manifest["config"] = asdict(cfg)
    for record in manifest.get("stages", {}).values():
      if record.get("status") == "complete":
        record.pop("error", None)
    return manifest
  return _new_manifest(cfg, prior, prior_sha)


def _record_stage(
  manifest: dict[str, Any],
  manifest_path: Path,
  key: str,
  updates: dict[str, Any],
) -> None:
  record = manifest["stages"].setdefault(key, {})
  if updates.get("status") == "running":
    record.clear()
  record.update(updates)
  manifest["updated_at_utc"] = _utc_now()
  _atomic_json(manifest_path, manifest)


def _completed_source(
  manifest: dict[str, Any], key: str
) -> tuple[ResumeSource, dict[str, Any]] | None:
  record = manifest["stages"].get(key)
  if not record or record.get("status") != "complete":
    return None
  run_dir = Path(record["run_dir"])
  checkpoint = Path(record["checkpoint"])
  validate_checkpoint(checkpoint)
  report = json.loads(Path(record["evaluation"]).read_text(encoding="utf-8"))
  return ResumeSource(run_dir, checkpoint), report


def _run_smoke(
  cfg: CurriculumCfg,
  manifest: dict[str, Any],
  manifest_path: Path,
  state_dir: Path,
  experiment_root: Path,
) -> None:
  if (
    not cfg.run_smoke or manifest["stages"].get("smoke", {}).get("status") == "complete"
  ):
    return
  key = "smoke"
  run_name = f"{cfg.run_prefix}_scratch_smoke_seed{cfg.seed}"
  command = build_train_command(cfg, STAGE_A, run_name, None, smoke=True)
  before = set(experiment_root.glob(f"*_{run_name}"))
  _record_stage(
    manifest,
    manifest_path,
    key,
    {
      "status": "running",
      "started_at_utc": _utc_now(),
      "command": command,
      "scratch": True,
    },
  )

  def on_start(pid: int) -> None:
    _record_stage(manifest, manifest_path, key, {"pid": pid})

  try:
    _run_process(
      command,
      state_dir / "smoke.log",
      dry_run=cfg.dry_run,
      on_start=on_start,
    )
    if cfg.dry_run:
      _record_stage(manifest, manifest_path, key, {"status": "dry_run"})
      return
    run_dir = _find_new_run_dir(experiment_root, run_name, before)
    checkpoint = find_latest_checkpoint(run_dir)
    iteration = validate_checkpoint(checkpoint)
    _assert_finite_training_log(run_dir)
    agent_text = (run_dir / "params" / "agent.yaml").read_text(encoding="utf-8")
    env_text = (run_dir / "params" / "env.yaml").read_text(encoding="utf-8")
    if "resume: false" not in agent_text.lower():
      raise RuntimeError("Scratch smoke unexpectedly enabled resume")
    if "forward23000" in agent_text.lower() or "pretrained_loco.pt" in env_text:
      raise RuntimeError("Scratch smoke contains a locomotion/Forward dependency")
    _record_stage(
      manifest,
      manifest_path,
      key,
      {
        "status": "complete",
        "finished_at_utc": _utc_now(),
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_iteration": iteration,
      },
    )
  except Exception as exc:
    _record_stage(manifest, manifest_path, key, {"status": "failed", "error": str(exc)})
    raise


def _assert_finite_evaluation(report: dict[str, Any], label: str) -> None:
  if not report.get("finite", False):
    raise RuntimeError(f"Non-finite evaluation for {label}")
  nan_rate = float(report.get("metrics", {}).get("failure_nan", {}).get("mean", 0.0))
  if nan_rate > 0.0:
    raise RuntimeError(f"NaN/Inf evaluation failures in {label}")


def _evaluate_stage(
  cfg: CurriculumCfg,
  stage: StageSpec,
  checkpoint: Path,
  state_dir: Path,
  evaluation_dir: Path,
) -> tuple[Path, dict[str, Any], str | dict[str, str] | None]:
  output = evaluation_dir / f"{cfg.run_prefix}_{stage.key}_seed{cfg.seed}.json"
  if cfg.all_contact_evaluation and stage.key in {"b1", "b2", "c"}:
    return _evaluate_all_contacts(cfg, stage, checkpoint, state_dir, evaluation_dir)
  split_shoulders = cfg.shoulder_split_evaluation and stage.key in {"b1", "b2", "c"}
  if not split_shoulders:
    video_folder = None
    if stage.key == "c" and cfg.record_final_video:
      video_folder = state_dir / "final_replay"
    command = build_evaluation_command(
      cfg, stage, checkpoint, output, video_folder=video_folder
    )
    _run_process(
      command,
      state_dir / f"{stage.key}_evaluation.log",
      dry_run=False,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    _assert_finite_evaluation(report, stage.key)
    video_record = str(video_folder.resolve()) if video_folder else None
    return output, report, video_record

  sides = {
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, 0.0, 1.0),
  }
  reports: dict[str, Any] = {}
  videos: dict[str, str] = {}
  for side, probabilities in sides.items():
    side_output = output.with_name(f"{output.stem}_{side}.json")
    video_folder = None
    if stage.key == "c" and cfg.record_final_video:
      video_folder = state_dir / "final_replay" / side
      videos[side] = str(video_folder.resolve())
    command = build_evaluation_command(
      cfg,
      stage,
      checkpoint,
      side_output,
      video_folder=video_folder,
      target_probabilities=probabilities,
    )
    _run_process(
      command,
      state_dir / f"{stage.key}_{side}_evaluation.log",
      dry_run=False,
    )
    side_report = json.loads(side_output.read_text(encoding="utf-8"))
    _assert_finite_evaluation(side_report, f"{stage.key}/{side}")
    reports[side] = side_report

  checks = {
    f"{side}/{name}": passed
    for side, report in reports.items()
    for name, passed in report["gate"]["checks"].items()
  }
  combined = {
    "finite": True,
    "split_shoulders": True,
    "checkpoint": str(checkpoint.resolve()),
    "sides": reports,
    "gate": {"passed": all(checks.values()), "checks": checks},
  }
  _atomic_json(output, combined)
  return output, combined, videos or None


def _evaluate_all_contacts(cfg, stage, checkpoint, state_dir, evaluation_dir):
  outputs, videos = {}, {}
  targets = {
    "stand": (1.0, 0.0, 0.0, 0.0),
    "left_shoulder": (0.0, 1.0, 0.0, 0.0),
    "right_shoulder": (0.0, 0.0, 1.0, 0.0),
    "chest": (1.0, 0.0, 0.0, 0.0),
    "back": (0.0, 0.0, 0.0, 1.0),
  }
  for name, probabilities in targets.items():
    selected = (
      replace(
        stage,
        evaluation_stage="a",
        wait_range_s=(1e9, 1e9),
        stationary_reward_weight=0.9,
      )
      if name == "stand"
      else stage
    )
    path = evaluation_dir / f"{cfg.run_prefix}_{stage.key}_{name}_seed{cfg.seed}.json"
    video_folder = None
    if stage.key == "c" and cfg.record_final_video:
      video_folder = state_dir / "final_replay" / name
      videos[name] = str(video_folder.resolve())
    command = build_evaluation_command(
      cfg,
      selected,
      checkpoint,
      path,
      video_folder=video_folder,
      target_probabilities=probabilities,
    )
    _run_process(
      command, state_dir / f"{stage.key}_{name}_evaluation.log", dry_run=False
    )
    report = json.loads(path.read_text())
    _assert_finite_evaluation(report, f"{stage.key}/{name}")
    outputs[name] = report
  checks = {
    f"{name}/{check}": passed
    for name, report in outputs.items()
    for check, passed in report["gate"]["checks"].items()
  }
  combined = {
    "finite": True,
    "conditions": outputs,
    "checkpoint": str(checkpoint.resolve()),
    "gate": {"passed": all(checks.values()), "checks": checks},
  }
  output = evaluation_dir / f"{cfg.run_prefix}_{stage.key}_seed{cfg.seed}.json"
  _atomic_json(output, combined)
  return output, combined, videos or None


def _sync_conditional_evaluation(cfg, wandb_url, report, warning):
  """Scalar-only evaluation summary; retain explicit retry state on network errors."""
  if not cfg.all_contact_evaluation:
    return None
  if not wandb_url:
    return {"status": "pending", "reason": "training run URL unavailable"}
  try:
    import wandb

    run_path = wandb_url.split("wandb.ai/", 1)[1].replace("/runs/", "/")
    run = wandb.Api(timeout=30).run(run_path)
    summary = {
      "evaluation/gate_passed": report["gate"]["passed"],
      "evaluation/warning": warning or "",
      "evaluation/gate_checks": report["gate"]["checks"],
    }
    reports = report.get("conditions", {"stand": report})
    for condition, result in reports.items():
      for metric, values in result.get("metrics", {}).items():
        for statistic, value in values.items():
          summary[f"evaluation/{condition}/{metric}/{statistic}"] = value
    run.summary.update(summary)
    return {"status": "synced", "run_url": wandb_url}
  except Exception as error:
    _safe_print(
      "[CURRICULUM] W&B evaluation summary pending; local JSON retained", flush=True
    )
    return {
      "status": "pending",
      "run_url": wandb_url,
      "error_type": type(error).__name__,
    }


def _run_training_stage(
  cfg: CurriculumCfg,
  manifest: dict[str, Any],
  manifest_path: Path,
  state_dir: Path,
  evaluation_dir: Path,
  experiment_root: Path,
  stage: StageSpec,
  resume: ResumeSource | None,
) -> tuple[ResumeSource, dict[str, Any]]:
  existing = _completed_source(manifest, stage.key)
  if existing is not None:
    return existing

  previous_record = dict(manifest["stages"].get(stage.key, {}))
  if cfg.all_contact_evaluation and previous_record:
    status = previous_record.get("status")
    if status in {"running", "evaluating"}:
      raise RuntimeError(
        "Conditional stage is already running or was interrupted without a clean exit; "
        "inspect its PID/log and checkpoint before marking a verified interruption. "
        "Refusing to silently repeat the training budget."
      )
    if status == "failed" and not is_recoverable_interruption(previous_record):
      raise RuntimeError(
        "Previous conditional stage hard-failed. Diagnose before resuming its "
        "validated checkpoint; automatic retraining is prohibited."
      )
  original_resume = resume
  resume_iteration = validate_checkpoint(resume.checkpoint) if resume else None
  target_iteration = expected_terminal_iteration(stage, resume_iteration)
  training_stage = stage
  recovery: dict[str, Any] | None = None
  if is_recoverable_interruption(previous_record):
    partial = _find_latest_stage_checkpoint(cfg, stage, experiment_root)
    if partial is None:
      raise RuntimeError(
        f"No valid partial checkpoint found for interrupted stage {stage.key}"
      )
    partial_source, partial_iteration = partial
    if partial_iteration >= target_iteration:
      raise RuntimeError(
        f"Interrupted stage {stage.key} already reached target iteration "
        f"{target_iteration}; evaluation-only recovery is required"
      )
    if resume_iteration is not None and partial_iteration <= resume_iteration:
      raise RuntimeError(
        f"Partial checkpoint {partial_source.checkpoint} made no progress beyond "
        f"parent iteration {resume_iteration}"
      )
    training_stage = replace(
      stage,
      iterations=remaining_iterations(target_iteration, partial_iteration),
    )
    resume = partial_source
    recovery = {
      "reason": previous_record.get("error"),
      "interrupted_at_utc": previous_record.get("finished_at_utc"),
      "partial_run_dir": str(partial_source.run_dir.resolve()),
      "partial_checkpoint": str(partial_source.checkpoint.resolve()),
      "partial_checkpoint_iteration": partial_iteration,
      "remaining_iterations": training_stage.iterations,
      "target_checkpoint_iteration": target_iteration,
    }

  if recovery is None:
    run_name = f"{cfg.run_prefix}_{stage.key}_seed{cfg.seed}"
  else:
    run_name = (
      f"{cfg.run_prefix}_{stage.key}_resume_from_"
      f"{recovery['partial_checkpoint_iteration']}_seed{cfg.seed}"
    )
  command = build_train_command(cfg, training_stage, run_name, resume)
  before = set(experiment_root.glob(f"*_{run_name}"))
  _record_stage(
    manifest,
    manifest_path,
    stage.key,
    {
      "status": "running",
      "label": stage.label,
      "started_at_utc": _utc_now(),
      "iterations_requested": stage.iterations,
      "iterations_this_run": training_stage.iterations,
      "target_checkpoint_iteration": target_iteration,
      "command": command,
      "resumed_from": str(resume.checkpoint.resolve()) if resume else None,
      "original_parent_checkpoint": (
        str(original_resume.checkpoint.resolve()) if original_resume else None
      ),
      "scratch": original_resume is None,
      "recovery": recovery,
    },
  )

  def on_start(pid: int) -> None:
    _record_stage(manifest, manifest_path, stage.key, {"pid": pid})

  try:
    wandb_url = _run_process(
      command,
      state_dir / f"{stage.key}_train.log",
      dry_run=cfg.dry_run,
      on_start=on_start,
    )
    if cfg.dry_run:
      _record_stage(manifest, manifest_path, stage.key, {"status": "dry_run"})
      raise RuntimeError("Dry-run mode does not produce curriculum checkpoints")

    run_dir = _find_new_run_dir(experiment_root, run_name, before)
    checkpoint = find_latest_checkpoint(run_dir)
    iteration = validate_checkpoint(checkpoint)
    if iteration != target_iteration:
      raise RuntimeError(
        f"Stage {stage.key} ended at checkpoint iteration {iteration}, "
        f"expected {target_iteration}"
      )
    _assert_finite_training_log(run_dir)

    evaluation_output, report, video_record = _evaluate_stage(
      cfg,
      stage,
      checkpoint,
      state_dir,
      evaluation_dir,
    )

    gate_passed = bool(report["gate"]["passed"])
    warning = (
      None
      if gate_passed
      else (f"Quality gate failed for {stage.key}; continuing by configured policy")
    )
    evaluation_sync = _sync_conditional_evaluation(cfg, wandb_url, report, warning)
    _record_stage(
      manifest,
      manifest_path,
      stage.key,
      {
        "status": "complete",
        "wandb_evaluation_sync": evaluation_sync,
        "finished_at_utc": _utc_now(),
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_iteration": iteration,
        "evaluation": str(evaluation_output.resolve()),
        "gate_passed": gate_passed,
        "gate_checks": report["gate"]["checks"],
        "warning": warning,
        "wandb_url": wandb_url,
        "video_folder": video_record,
      },
    )
    if warning:
      _safe_print(f"[CURRICULUM] WARNING: {warning}", flush=True)
    return ResumeSource(run_dir, checkpoint), report
  except Exception as exc:
    _record_stage(
      manifest,
      manifest_path,
      stage.key,
      {"status": "failed", "error": str(exc), "finished_at_utc": _utc_now()},
    )
    raise


def _load_parent_stage_a(
  cfg: CurriculumCfg,
) -> tuple[ResumeSource, dict[str, Any], dict[str, Any]]:
  parent_path = (REPO_ROOT / cfg.parent_stage_a_manifest).resolve()
  if not parent_path.is_file():
    raise FileNotFoundError(parent_path)
  parent = json.loads(parent_path.read_text(encoding="utf-8"))
  if parent["prior"]["sha256"] != cfg.prior_sha256:
    raise RuntimeError("Parent stage-A prior SHA-256 does not match this curriculum")
  parent_cfg = parent["config"]
  if float(parent_cfg["gsi_max_root_planar_speed"]) != cfg.gsi_max_root_planar_speed:
    raise RuntimeError("Parent stage-A GSI speed limit does not match")
  if float(parent_cfg["stationary_speed_scale"]) != cfg.stationary_speed_scale:
    raise RuntimeError("Parent stage-A stationary speed scale does not match")

  stages = parent["stages"]
  key = (
    "a_extension" if stages.get("a_extension", {}).get("status") == "complete" else "a"
  )
  record = stages.get(key)
  if not record or record.get("status") != "complete":
    raise RuntimeError("Parent manifest has no completed terminal stage A")
  run_dir = Path(record["run_dir"])
  checkpoint = Path(record["checkpoint"])
  validate_checkpoint(checkpoint)
  report = json.loads(Path(record["evaluation"]).read_text(encoding="utf-8"))
  lineage = {
    "manifest": str(parent_path),
    "stage": key,
    "run_dir": str(run_dir.resolve()),
    "checkpoint": str(checkpoint.resolve()),
    "evaluation": str(Path(record["evaluation"]).resolve()),
  }
  return ResumeSource(run_dir, checkpoint), report, lineage


def run(cfg: CurriculumCfg) -> None:
  gpu = gpu_identity(cfg.gpu_id)
  os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
  os.environ["MUJOCO_EGL_DEVICE_ID"] = str(cfg.gpu_id)
  _safe_print(
    f"[CURRICULUM] physical GPU {cfg.gpu_id} ({gpu['uuid']}); local CUDA device 0",
    flush=True,
  )
  prior = (REPO_ROOT / cfg.prior_checkpoint).resolve()
  if not prior.is_file():
    raise FileNotFoundError(prior)
  prior_sha = _sha256(prior)
  if prior_sha != cfg.prior_sha256:
    raise RuntimeError(
      f"Prior SHA-256 mismatch: expected {cfg.prior_sha256}, got {prior_sha}"
    )

  state_dir = (REPO_ROOT / cfg.state_dir).resolve()
  evaluation_dir = (REPO_ROOT / cfg.evaluation_dir).resolve()
  experiment_root = (REPO_ROOT / cfg.log_root / cfg.experiment_name).resolve()
  state_dir.mkdir(parents=True, exist_ok=True)
  evaluation_dir.mkdir(parents=True, exist_ok=True)
  experiment_root.mkdir(parents=True, exist_ok=True)
  manifest_path = state_dir / "curriculum_state.json"
  manifest = _load_or_create_manifest(cfg, manifest_path, prior, prior_sha)
  manifest["gpu"] = gpu
  _atomic_json(manifest_path, manifest)

  _run_smoke(cfg, manifest, manifest_path, state_dir, experiment_root)
  if cfg.smoke_only:
    manifest["status"] = "smoke_complete"
    manifest["updated_at_utc"] = _utc_now()
    _atomic_json(manifest_path, manifest)
    return
  if cfg.dry_run:
    manifest["status"] = "dry_run"
    _atomic_json(manifest_path, manifest)
    return

  if cfg.parent_stage_a_manifest:
    source, stage_a_report, lineage = _load_parent_stage_a(cfg)
    manifest["stage_a_parent"] = lineage
    manifest["updated_at_utc"] = _utc_now()
    _atomic_json(manifest_path, manifest)
  else:
    source, stage_a_report = _run_training_stage(
      cfg,
      manifest,
      manifest_path,
      state_dir,
      evaluation_dir,
      experiment_root,
      STAGE_A,
      None,
    )
    if should_extend_stage_a(stage_a_report):
      source, _ = _run_training_stage(
        cfg,
        manifest,
        manifest_path,
        state_dir,
        evaluation_dir,
        experiment_root,
        STAGE_A_EXTENSION,
        source,
      )

  for stage in contact_stages_for_cfg(cfg):
    source, _ = _run_training_stage(
      cfg,
      manifest,
      manifest_path,
      state_dir,
      evaluation_dir,
      experiment_root,
      stage,
      source,
    )

  manifest["status"] = "complete"
  manifest["finished_at_utc"] = _utc_now()
  manifest["final_checkpoint"] = str(source.checkpoint.resolve())
  manifest["updated_at_utc"] = _utc_now()
  _atomic_json(manifest_path, manifest)
  _safe_print(f"[CURRICULUM] Complete: {manifest_path}", flush=True)


if __name__ == "__main__":
  run(tyro.cli(CurriculumCfg))
