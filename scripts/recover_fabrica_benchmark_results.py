#!/usr/bin/env python3
"""Recover a complete all-Fabrica benchmark index from per-part artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/fabrica_all_v1.yaml",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=REPO_ROOT / "artifacts/grasp_generation_benchmark_pdz",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def recover_results(config_path: Path, benchmark_root: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    part_records: list[dict[str, object]] = []
    orientation_records: list[dict[str, object]] = []
    incomplete: list[dict[str, object]] = []
    for assembly_name in sorted(str(name) for name in config["assemblies"]):
        part_ids = sorted(
            (str(value) for value in config["assemblies"][assembly_name]),
            key=lambda value: (int(value), value),
        )
        for local_part_id in part_ids:
            part_dir = benchmark_root / "parts" / assembly_name / local_part_id
            stable_path = part_dir / "stable_orientations.json"
            if not stable_path.is_file():
                incomplete.append(
                    {
                        "assembly_name": assembly_name,
                        "local_part_id": local_part_id,
                        "reason": "missing_stable_orientations",
                    }
                )
                continue
            stable = json.loads(stable_path.read_text(encoding="utf-8"))
            expected = int(stable.get("stable_orientation_count", 0))
            part_orientations: list[dict[str, object]] = []
            for orientation in stable.get("orientations", []):
                orientation_id = str(orientation["orientation_id"])
                details_path = (
                    part_dir / "orientations" / orientation_id / "details.json"
                )
                if not details_path.is_file():
                    incomplete.append(
                        {
                            "assembly_name": assembly_name,
                            "local_part_id": local_part_id,
                            "orientation_id": orientation_id,
                            "reason": "missing_orientation_details",
                        }
                    )
                    continue
                details = json.loads(details_path.read_text(encoding="utf-8"))
                record = {
                    "assembly": assembly_name,
                    "part_id": local_part_id,
                    "target_mesh_path": str(details["target"]["target_mesh_path"]),
                    "orientation_id": orientation_id,
                    "status": str(details["status"]),
                    "stage1_assembly_feasible_count": int(
                        details.get("stage1", {}).get("assembly_feasible_count", 0)
                    ),
                    "stage2_ground_feasible_count": int(
                        details.get("stage2", {}).get("ground_feasible_count", 0)
                    ),
                    "details_path": str(details_path.resolve()),
                    "error": details.get("error"),
                }
                part_orientations.append(record)
                orientation_records.append(record)
            statuses = Counter(str(item["status"]) for item in part_orientations)
            part_records.append(
                {
                    "assembly": assembly_name,
                    "part_id": local_part_id,
                    "target_mesh_path": f"obj/fabrica/{assembly_name}/{local_part_id}.obj",
                    "stable_orientation_count": expected,
                    "completed_orientation_count": len(part_orientations),
                    "status": (
                        "direct_success"
                        if statuses["direct_success"] > 0
                        else "no_direct_success"
                    ),
                    "orientation_status_counts": dict(statuses),
                }
            )
    if incomplete:
        raise RuntimeError(
            f"Cannot recover the benchmark; {len(incomplete)} artifacts are incomplete: "
            f"{incomplete[:8]}"
        )
    expected_parts = sum(len(values) for values in config["assemblies"].values())
    if len(part_records) != expected_parts:
        raise RuntimeError(
            f"Recovered {len(part_records)}/{expected_parts} configured parts."
        )
    return {
        "schema_version": 1,
        "provenance": {
            "recovered_from_per_part_artifacts": True,
            "recovered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config_path": str(config_path.resolve()),
            "benchmark_root": str(benchmark_root.resolve()),
            "target_count": len(part_records),
        },
        "parts": part_records,
        "orientations": orientation_records,
        "summary": {
            "part_count": len(part_records),
            "orientation_count": len(orientation_records),
            "orientation_status_counts": dict(
                Counter(str(item["status"]) for item in orientation_records)
            ),
            "part_status_counts": dict(
                Counter(str(item["status"]) for item in part_records)
            ),
        },
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    benchmark_root = args.benchmark_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else benchmark_root / "results.json"
    )
    payload = recover_results(config_path, benchmark_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[DONE] Recovered {payload['summary']['part_count']} parts and "
        f"{payload['summary']['orientation_count']} orientations into {output}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
