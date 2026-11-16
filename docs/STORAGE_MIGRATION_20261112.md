# SMP storage migration — 2026-11-12

- Completed at: `2026-11-12T04:12:19Z`
- Persistent archive: `/personal_fast_storage/smp_experiments/smp_archive_20261112/root_smp`
- Access model: original workspace paths are absolute symbolic links to the archive.
- Verification: source and destination directory structure matched; every regular file passed SHA-256 comparison before local backups were removed.
- Migrated: 35 directory roots, 977 files, 5,163,934,795 bytes (about 4.81 GiB logical size).

## Migrated content

- `logs/reproduction/rsl_rl/`: all closed PPO run directories and checkpoints; no training process was active during migration.
- `logs/reproduction/conditional_contact_v1/`: complete v1 data, candidate prior, validation reports, previews, and local W&B record.
- `logs/reproduction/replay/` and `logs/reproduction/video_checks/`.
- `logs/reproduction/pretrain/`: reproduction/smoke/intermediate pretraining checkpoints. Published priors under `datasets/pretrain_ckpt/` were not changed.
- Three completed curriculum `final_replay/` directories.
- 27 online W&B run directories that had both a local summary and a normal shutdown marker.

## Deliberately retained on the system disk

- `wandb/offline-run-20261105_101410-90fqg9v8`: offline, therefore not confirmed synced.
- Nine online run directories whose local completion/sync evidence was incomplete: `1n01b16f`, `yhwnl2nu`, `2jo6692l`, `4ikmyq96`, `rm9nurmi`, `gy6acoon`, `0y6tpff0`, `0c297apw`, and `tm3pga3v`.
- Dataset-review videos under `logs/reproduction/data/`; these are data curation previews, not policy replays.
- The active/final published prior locations under `datasets/pretrain_ckpt/`.
- `.venv`, source code, CUDA/system libraries, and caches.

## Result

- System disk before removal: about 5.0 GiB free.
- System disk after removal: about 9.7 GiB free (`25 GiB` total, `61%` used).
- Archive filesystem at completion: about 447 TiB shared free space. This is filesystem-wide capacity, not a guaranteed per-user quota or backup.

