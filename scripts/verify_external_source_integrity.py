"""Verify publisher blob identities; recover only transport-corrupted new files."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os

import numpy as np
from scripts.review_external_shoulder_sources import (
  OUTPUT,
  REPOS,
  digest,
  get_bytes,
  save_json,
  url_for,
  utc_now,
)


def matches(payload, metadata):
  if len(payload) != metadata["size"]:
    return False
  if metadata.get("lfs"):
    return hashlib.sha256(payload).hexdigest() == metadata["lfs"]["sha256"]
  header = f"blob {len(payload)}\0".encode()
  return hashlib.sha1(header + payload).hexdigest() == metadata["blobId"]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repair-transport", action="store_true")
  args = parser.parse_args()
  plan = json.loads((OUTPUT / "plan.json").read_text())
  downloads_path = OUTPUT / "downloads.json"
  downloads = json.loads(downloads_path.read_text())
  if downloads["plan_sha256"] != digest(OUTPUT / "plan.json"):
    raise ValueError("Acquisition plan mismatch")
  metadata = {}
  for dataset, (repo, revision) in REPOS.items():
    identity_path = OUTPUT / "metadata" / f"{dataset}_publisher_blobs.json"
    if identity_path.exists():
      response = json.loads(identity_path.read_text())
    else:
      response = json.loads(
        get_bytes(
          f"https://huggingface.co/api/datasets/{repo}/revision/{revision}?blobs=true"
        )
      )
      if response["sha"] != revision:
        raise ValueError("Publisher revision mismatch")
      save_json(identity_path, response)
    metadata[dataset] = {r["rfilename"]: r for r in response["siblings"]}
  repairs, verified = [], []
  backup = OUTPUT / "transport_recovery"
  for item in plan["documents"] + plan["sources"]:
    key = item["dataset"] + "/" + item["remote_path"]
    path = OUTPUT / "raw" / key
    identity = metadata[item["dataset"]][item["remote_path"]]
    payload = path.read_bytes()
    if not matches(payload, identity):
      if not args.repair_transport:
        raise ValueError(f"Publisher identity mismatch: {key}")
      replacement = get_bytes(url_for(item["dataset"], item["remote_path"]))
      if not matches(replacement, identity):
        raise ValueError(f"Refetched bytes still fail publisher identity: {key}")
      if item["remote_path"].endswith(".npy"):
        array = np.load(io.BytesIO(replacement), allow_pickle=False)
        if array.ndim != 2 or array.shape[1] != 36 or not np.isfinite(array).all():
          raise ValueError(
            "Publisher file is malformed; do not repair numeric contents"
          )
      old_hash = hashlib.sha256(payload).hexdigest()
      backup.mkdir(exist_ok=True)
      archived = backup / (old_hash + "_truncated" + path.suffix)
      if archived.exists():
        raise RuntimeError("Recovery archive already exists; inspect before repeating")
      os.rename(path, archived)
      temporary = path.with_suffix(path.suffix + ".verified")
      with temporary.open("xb") as stream:
        stream.write(replacement)
        stream.flush()
        os.fsync(stream.fileno())
      os.replace(temporary, path)
      payload = replacement
      repairs.append(
        {
          "source_id": key,
          "original_download_sha256": old_hash,
          "archived_path": str(archived),
          "publisher": identity,
          "verified_sha256": digest(path),
        }
      )
      downloads["files"][key].update(sha256=digest(path), bytes=len(payload))
    if item in plan["sources"]:
      array = (
        np.load(io.BytesIO(payload), allow_pickle=False)
        if item["remote_path"].endswith(".npy")
        else np.loadtxt(io.BytesIO(payload), delimiter=",")
      )
      if array.ndim != 2 or array.shape[1] != 36 or not np.isfinite(array).all():
        raise ValueError(f"Invalid source array despite correct bytes: {key}")
    verified.append({"source_id": key, "publisher": identity, "sha256": digest(path)})
  if repairs:
    archived_downloads = backup / "downloads_before_repair.json"
    if archived_downloads.exists():
      raise RuntimeError("Prior recovery state exists; do not overwrite")
    os.rename(downloads_path, archived_downloads)
    downloads["transport_repairs"] = repairs
    downloads["updated_at_utc"] = utc_now()
    save_json(downloads_path, downloads)
    incomplete = OUTPUT / "review"
    if incomplete.exists():
      if (incomplete / "manifest.json").exists():
        raise RuntimeError("Completed review exists: do not automatically replace it")
      archived_review = OUTPUT / "review_attempt01_transport_truncation"
      if archived_review.exists():
        raise RuntimeError("Attempt archive already exists")
      os.rename(incomplete, archived_review)
      save_json(
        archived_review / "failure.json",
        {
          "status": "aborted_transport_truncation",
          "recorded_at_utc": utc_now(),
          "repairs": repairs,
          "thresholds_changed": False,
          "reason": "HTTP read returned fewer bytes than publisher; no invalid data accepted",
        },
      )
  report = {
    "completed_at_utc": utc_now(),
    "passed": True,
    "plan_sha256": digest(OUTPUT / "plan.json"),
    "verified": verified,
    "repairs": repairs,
    "script_sha256": digest(__file__),
  }
  path = OUTPUT / "publisher_integrity.json"
  if path.exists():
    raise RuntimeError("Integrity report already exists; do not overwrite")
  save_json(path, report)
  print(
    json.dumps(
      {
        "verified_files": len(verified),
        "transport_repairs": len(repairs),
        "report": str(path),
      }
    ),
    flush=True,
  )


if __name__ == "__main__":
  main()
