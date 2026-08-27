#!/usr/bin/env python3
"""Write coverage, yield, and provenance reports for an all-Fabrica grasp manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--benchmark-results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_reports(
    *,
    config_path: Path,
    inventory_path: Path,
    benchmark_results_path: Path,
    manifest_path: Path,
    output_root: Path,
) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inventory = _load(inventory_path)
    benchmark = _load(benchmark_results_path)
    manifest = _load(manifest_path)
    inventory_keys = {str(part["part_key"]) for part in inventory["parts"]}
    represented_keys = {str(part["part_key"]) for part in manifest["parts"]}
    exclusions = {str(item["part_key"]): dict(item) for item in manifest["exclusions"]}
    covered_keys = represented_keys | set(exclusions)
    missing_coverage = sorted(inventory_keys - covered_keys)
    unknown_coverage = sorted(covered_keys - inventory_keys)
    if missing_coverage or unknown_coverage:
        raise ValueError(
            "Manifest coverage does not exactly match the source inventory: "
            f"missing={missing_coverage[:8]} unknown={unknown_coverage[:8]}"
        )

    selected_by_orientation = Counter(
        str(target["orientation_id"]) for target in manifest["targets"]
    )
    alternates_by_orientation = Counter(
        str(target["orientation_id"]) for target in manifest["alternates"]
    )
    orientations_by_part: dict[str, list[dict[str, object]]] = defaultdict(list)
    for orientation in manifest["orientations"]:
        orientations_by_part[str(orientation["part_key"])].append(orientation)

    part_rows: list[dict[str, object]] = []
    orientation_rows: list[dict[str, object]] = []
    inventory_by_key = {str(part["part_key"]): part for part in inventory["parts"]}
    for part_key in sorted(inventory_keys):
        source = inventory_by_key[part_key]
        part_orientations = orientations_by_part.get(part_key, [])
        exclusion = exclusions.get(part_key)
        part_rows.append(
            {
                "assembly_name": source["assembly_name"],
                "local_part_id": source["local_part_id"],
                "part_key": part_key,
                "source_sha256": source["source_sha256"],
                "vertex_count": source["vertex_count"],
                "face_count": source["face_count"],
                "largest_extent_m": source["largest_extent_m"],
                "represented": part_key in represented_keys,
                "retained_orientation_count": len(part_orientations),
                "selected_target_count": sum(
                    selected_by_orientation[str(item["orientation_id"])]
                    for item in part_orientations
                ),
                "alternate_target_count": sum(
                    alternates_by_orientation[str(item["orientation_id"])]
                    for item in part_orientations
                ),
                "exclusion_reason": "" if exclusion is None else exclusion["reason"],
                "exclusion_detail": "" if exclusion is None else exclusion.get("detail", ""),
            }
        )
        for orientation in part_orientations:
            orientation_id = str(orientation["orientation_id"])
            orientation_rows.append(
                {
                    "assembly_name": source["assembly_name"],
                    "local_part_id": source["local_part_id"],
                    "part_key": part_key,
                    "orientation_id": orientation_id,
                    "local_orientation_id": orientation["local_orientation_id"],
                    "ground_feasible_count": orientation.get("ground_feasible_count", 0),
                    "training_width_feasible_count": orientation.get(
                        "training_width_feasible_count", 0
                    ),
                    "selected_target_count": selected_by_orientation[orientation_id],
                    "alternate_target_count": alternates_by_orientation[orientation_id],
                    "train_target_count": orientation.get("split_target_counts", {}).get(
                        "train", 0
                    ),
                    "validation_target_count": orientation.get(
                        "split_target_counts", {}
                    ).get("validation", 0),
                    "test_target_count": orientation.get("split_target_counts", {}).get(
                        "test", 0
                    ),
                }
            )

    reports_root = output_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    with (reports_root / "yield_by_assembly_part.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(part_rows[0]))
        writer.writeheader()
        writer.writerows(part_rows)
    with (reports_root / "yield_by_assembly_part_orientation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames = list(orientation_rows[0]) if orientation_rows else ["part_key"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(orientation_rows)
    excluded_payload = {
        "schema_version": 1,
        "excluded_part_count": len(exclusions),
        "parts": [exclusions[key] for key in sorted(exclusions)],
    }
    (reports_root / "excluded_parts.json").write_text(
        json.dumps(excluded_payload, indent=2) + "\n", encoding="utf-8"
    )

    benchmark_summary = dict(benchmark.get("summary", {}))
    data_root = output_root / "merged"
    planned_path = data_root / "planned_manifest.json"
    paths_path = data_root / "paths.npz"
    rotation_path = data_root / "rotation_resets.npz"
    goal_path = data_root / "goal_catalog.npz"
    moveit_complete = False
    if planned_path.is_file():
        planned = _load(planned_path)
        moveit_complete = bool(planned.get("moveit_validation", {}).get("complete"))

    def _target_ids(path: Path) -> list[str] | None:
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as arrays:
            return arrays["target_ids"].astype(str).tolist()

    path_ids = _target_ids(paths_path)
    rotation_ids = _target_ids(rotation_path)
    goal_ids = _target_ids(goal_path)
    exact_alignment = (
        path_ids is not None
        and rotation_ids is not None
        and goal_ids is not None
        and path_ids == rotation_ids == goal_ids
    )
    yield_payload = {
        "schema_version": 1,
        "inventory": {
            "configured_parts": int(inventory["configured_part_count"]),
            "valid_source_parts": int(inventory["valid_part_count"]),
            "source_exclusions": int(inventory["excluded_part_count"]),
        },
        "grasp_generation": {
            "processed_parts": int(benchmark_summary.get("part_count", 0)),
            "processed_orientations": int(benchmark_summary.get("orientation_count", 0)),
            "orientation_status_counts": benchmark_summary.get(
                "orientation_status_counts", {}
            ),
        },
        "manifest": {
            "represented_parts": len(represented_keys),
            "excluded_parts": len(exclusions),
            "retained_orientations": len(manifest["orientations"]),
            "selected_targets": len(manifest["targets"]),
            "alternate_targets": len(manifest["alternates"]),
        },
        "downstream_validation": {
            "moveit_plan_validation_complete": moveit_complete,
            "path_asset_complete": path_ids is not None,
            "path_target_count": 0 if path_ids is None else len(path_ids),
            "rotation_reset_validation_complete": rotation_ids is not None,
            "rotation_target_count": 0 if rotation_ids is None else len(rotation_ids),
            "mujoco_goal_capture_complete": goal_ids is not None,
            "goal_target_count": 0 if goal_ids is None else len(goal_ids),
            "exact_target_alignment": exact_alignment,
            "training_ready": bool(moveit_complete and exact_alignment),
        },
    }
    (reports_root / "yield_by_stage.json").write_text(
        json.dumps(yield_payload, indent=2) + "\n", encoding="utf-8"
    )
    dataset_config = output_root / "dataset_config.yaml"
    shutil.copyfile(config_path, dataset_config)
    index = {
        "schema_version": 1,
        "dataset_name": str(config["dataset"]["name"]),
        "build_status": (
            "training_catalog_generated"
            if yield_payload["downstream_validation"]["training_ready"]
            else "grasp_manifest_generated"
        ),
        "training_ready": bool(
            yield_payload["downstream_validation"]["training_ready"]
        ),
        "source_config_sha256": _sha256(config_path),
        "inventory_sha256": _sha256(inventory_path),
        "benchmark_results_sha256": _sha256(benchmark_results_path),
        "merged_manifest_sha256": _sha256(manifest_path),
        "assembly_count": int(manifest["assembly_count"]),
        "configured_part_count": int(inventory["configured_part_count"]),
        "represented_part_count": len(represented_keys),
        "excluded_part_count": len(exclusions),
        "selected_target_count": len(manifest["targets"]),
        "alternate_target_count": len(manifest["alternates"]),
        "split_scheme_metadata": manifest["split_scheme_metadata"],
        "artifacts": {
            "inventory": str(inventory_path),
            "merged_manifest": str(manifest_path),
            "benchmark_results": str(benchmark_results_path),
            "yield_by_stage": str(reports_root / "yield_by_stage.json"),
            "excluded_parts": str(reports_root / "excluded_parts.json"),
        },
    }
    (output_root / "dataset_index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    return index


def main() -> None:
    args = parse_args()
    index = build_reports(
        config_path=args.config.expanduser().resolve(),
        inventory_path=args.inventory.expanduser().resolve(),
        benchmark_results_path=args.benchmark_results.expanduser().resolve(),
        manifest_path=args.manifest.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
    )
    print(
        f"[DONE] Reports cover {index['represented_part_count']} represented and "
        f"{index['excluded_part_count']} explicitly excluded parts; "
        f"selected targets={index['selected_target_count']}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
