"""Carry a training run to another machine through git: pack it into the repo, unpack it there.

GitHub refuses files over 100 MB, and a checkpoint with its optimizer state
is larger, so ``pack`` stores the run's latest checkpoint as two files:
``checkpoint.pt`` (weights, entropy temperatures, RNG, iteration: everything
but the optimizer) and ``optimizer.pt`` (the actor's Adam moments). It also
copies the files a seamless resume needs: config, metadata, metrics, log,
and dashboard. ``unpack`` rebuilds ``runs/<game>/<run>/`` with the
checkpoint whole again and its latest.json. The usual train command then
resumes it exactly:

    .venv/bin/python tools/run_snapshot.py pack runs/piquet/partie1 snapshots/piquet/partie1
    .venv/bin/python tools/run_snapshot.py unpack snapshots/piquet/partie1
    tools/train_supervised.sh piquet partie1 configs/piquet/partie.toml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUN_FILES = ("config.toml", "metadata.json", "metrics.csv", "train.log", "dashboard.png")
SPLIT = ("trainer", "actor_optimizer")  # the optimizer state, stored on its own


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack(run: Path, out: Path) -> None:
    latest = json.loads((run / "checkpoints" / "latest.json").read_text())
    checkpoint = run / "checkpoints" / latest["path"]
    if _sha256(checkpoint) != latest["sha256"]:
        sys.exit(f"{checkpoint} does not match latest.json")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    optimizer = payload[SPLIT[0]].pop(SPLIT[1])
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in RUN_FILES:
        if (run / name).exists():
            shutil.copy2(run / name, out / name)
    torch.save(payload, out / "checkpoint.pt")
    torch.save(optimizer, out / "optimizer.pt")
    manifest = {
        "game": run.parent.name,
        "run": run.name,
        "checkpoint": latest["path"],
        "iteration": latest["iteration"],
        "sha256": {name: _sha256(out / name) for name in ("checkpoint.pt", "optimizer.pt")},
    }
    (out / "snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"packed {run} at iteration {latest['iteration']} into {out}")


def unpack(snapshot: Path, runs: Path) -> None:
    manifest = json.loads((snapshot / "snapshot.json").read_text())
    for name, digest in manifest["sha256"].items():
        if _sha256(snapshot / name) != digest:
            sys.exit(f"{snapshot / name} is corrupt (SHA-256 mismatch)")
    run = runs / manifest["game"] / manifest["run"]
    checkpoint = run / "checkpoints" / manifest["checkpoint"]
    if checkpoint.exists():
        sys.exit(f"{checkpoint} already exists; not overwriting")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        if (snapshot / name).exists():
            shutil.copy2(snapshot / name, run / name)
    payload = torch.load(snapshot / "checkpoint.pt", map_location="cpu", weights_only=False)
    payload[SPLIT[0]][SPLIT[1]] = torch.load(snapshot / "optimizer.pt", map_location="cpu", weights_only=False)
    torch.save(payload, checkpoint)
    latest = {
        "iteration": manifest["iteration"],
        "path": checkpoint.name,
        "sha256": _sha256(checkpoint),
        "size": checkpoint.stat().st_size,
        "written_at": "unpacked",
    }
    (checkpoint.parent / "latest.json").write_text(json.dumps(latest, indent=2) + "\n")
    print(f"unpacked {run} at iteration {manifest['iteration']}; resume with the run's train command")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    packing = commands.add_parser("pack", help="copy a run's latest state into a git-friendly snapshot")
    packing.add_argument("run", type=Path, help="runs/<game>/<run>")
    packing.add_argument("out", type=Path, help="snapshot directory, e.g. snapshots/<game>/<run>")
    unpacking = commands.add_parser("unpack", help="rebuild runs/<game>/<run> from a snapshot")
    unpacking.add_argument("snapshot", type=Path)
    unpacking.add_argument("--runs", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    if args.command == "pack":
        pack(args.run, args.out)
    else:
        unpack(args.snapshot, args.runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
