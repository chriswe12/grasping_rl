#!/usr/bin/env python3
"""Build portable, part-disjoint training shards for the Fabrica-all catalog."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping.fabrica_grasp_debug import load_grasp_bundle
from grasp_planning.mujoco import build_bundle_local_mesh
from grasp_planning.rl.fabrica_dataset import (
    FABRICA_DATASET_NAME,
    FABRICA_DATASET_SCHEMA_VERSION,
    FABRICA_SHARD_COUNT,
    FABRICA_SHARD_SCHEMA_VERSION,
    FABRICA_SUPPORTED_SHARD_COUNTS,
    assign_parts_to_shards,
    canonical_json_sha256,
    repo_relative,
    sha256_file,
    subset_target_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "isaac_rl/data/fabrica_all_v1",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        choices=FABRICA_SUPPORTED_SHARD_COUNTS,
        default=FABRICA_SHARD_COUNT,
        help="Build or refresh this rank-count layout while preserving the other supported layouts.",
    )
    return parser.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def _portable_artifact(path: Path) -> dict[str, object]:
    return {
        "path": repo_relative(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _part_radius(bundle_path: Path) -> float:
    mesh = build_bundle_local_mesh(load_grasp_bundle(bundle_path))
    vertices = np.asarray(mesh.vertices_obj, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"Invalid bundle-local mesh vertices from {bundle_path}")
    # The object may occupy any catalog support orientation before an in-plane
    # yaw perturbation. The full 3-D radius is therefore a conservative XY
    # sweep radius about the bundle-local part origin.
    return float(np.ceil(np.linalg.norm(vertices, axis=1).max() * 1000.0) / 1000.0)


def main() -> None:  # noqa: C901 - artifact finalization is intentionally linear
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    merged_root = dataset_root / "merged"
    source_paths = {
        "goal_catalog": merged_root / "goal_catalog.npz",
        "paths": merged_root / "paths.npz",
        "rotation_resets": merged_root / "rotation_resets.npz",
    }
    source = {name: _load_npz(path) for name, path in source_paths.items()}
    target_ids = source["goal_catalog"]["target_ids"].astype(str)
    for name in ("paths", "rotation_resets"):
        if not np.array_equal(target_ids, source[name]["target_ids"].astype(str)):
            raise ValueError(f"{name}.target_ids do not exactly match goal_catalog.target_ids")
    if len(set(target_ids.tolist())) != len(target_ids):
        raise ValueError("Merged catalog target_ids are not unique")

    planned_manifest_path = merged_root / "planned_manifest.json"
    planned_manifest = json.loads(planned_manifest_path.read_text(encoding="utf-8"))
    parts_by_name = {str(item["part_id"]): item for item in planned_manifest["parts"]}
    catalog_part_names = tuple(source["goal_catalog"]["part_names"].astype(str).tolist())
    if set(catalog_part_names) != set(parts_by_name):
        raise ValueError("Planned manifest parts do not exactly match catalog part_names")

    part_metadata: dict[str, dict[str, object]] = {}
    for part_name in catalog_part_names:
        part = parts_by_name[part_name]
        expected_usd = (
            dataset_root
            / "assemblies"
            / str(part["assembly_name"])
            / "usd"
            / f"part_{part['local_part_id']}_bundle_local.usd"
        )
        if not expected_usd.is_file():
            raise FileNotFoundError(
                f"Missing bundle-local part USD for {part_name}: {expected_usd}. Run build_assembly_part_usds.py first."
            )
        radius = _part_radius(Path(part["source_current_stage2_bundle"]).expanduser().resolve())
        part_metadata[part_name] = {
            "part_name": part_name,
            "assembly_name": str(part["assembly_name"]),
            "local_part_id": str(part["local_part_id"]),
            "usd_path": repo_relative(expected_usd),
            "usd_sha256": sha256_file(expected_usd),
            "usd_size_bytes": expected_usd.stat().st_size,
            "xy_rotation_radius_m": radius,
        }

    part_ids = source["goal_catalog"]["part_ids"].astype(str)
    split_ids = source["goal_catalog"]["split_ids"].astype(str)
    assignments = assign_parts_to_shards(part_ids, split_ids, shard_count=args.shard_count)
    shard_root = (
        dataset_root / "shards"
        if args.shard_count == FABRICA_SHARD_COUNT
        else dataset_root / "shard_layouts" / f"layout_{args.shard_count:02d}"
    )
    shard_records: list[dict[str, object]] = []
    covered_targets: list[str] = []
    for shard_index, shard_parts in enumerate(assignments):
        output_root = shard_root / f"shard_{shard_index:02d}"
        output_root.mkdir(parents=True, exist_ok=True)
        indices = np.flatnonzero(np.isin(part_ids, np.asarray(shard_parts))).astype(np.int64)
        usd_paths = tuple(str(part_metadata[name]["usd_path"]) for name in shard_parts)
        for name, arrays in source.items():
            subset = subset_target_arrays(
                arrays,
                indices,
                part_names=shard_parts if "part_ids" in arrays else None,
                part_usd_paths=usd_paths if "part_ids" in arrays else None,
            )
            if "source_planned_manifest" in subset:
                subset["source_planned_manifest"] = np.asarray(repo_relative(planned_manifest_path))
            if "source_paths_asset" in subset:
                subset["source_paths_asset"] = np.asarray(repo_relative(output_root / "paths.npz"))
            np.savez_compressed(output_root / f"{name}.npz", **subset)

        split_counts = Counter(split_ids[indices].tolist())
        manifest = {
            "schema_version": FABRICA_SHARD_SCHEMA_VERSION,
            "dataset_name": FABRICA_DATASET_NAME,
            "shard_index": shard_index,
            "shard_count": args.shard_count,
            "assignment_method": "part_disjoint_greedy_train_then_total_v1",
            "target_count": int(indices.size),
            "split_counts": {name: int(split_counts.get(name, 0)) for name in ("train", "validation", "test")},
            "assembly_names": sorted({str(part_metadata[name]["assembly_name"]) for name in shard_parts}),
            "part_names": list(shard_parts),
            "part_usd_paths": list(usd_paths),
            "part_xy_rotation_radii_m": [float(part_metadata[name]["xy_rotation_radius_m"]) for name in shard_parts],
            "parts": [part_metadata[name] for name in shard_parts],
            "artifacts": {
                name: _portable_artifact(output_root / f"{name}.npz")
                for name in ("goal_catalog", "paths", "rotation_resets")
            },
        }
        manifest_path = output_root / "part_inventory.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        shard_records.append(
            {
                "shard_index": shard_index,
                "manifest": repo_relative(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "target_count": int(indices.size),
                "split_counts": manifest["split_counts"],
                "part_count": len(shard_parts),
            }
        )
        covered_targets.extend(target_ids[indices].tolist())
        print(
            f"[SHARD] {shard_index}: targets={indices.size} splits={dict(split_counts)} parts={len(shard_parts)}",
            flush=True,
        )

    if sorted(covered_targets) != sorted(target_ids.tolist()) or len(covered_targets) != len(set(covered_targets)):
        raise RuntimeError("Shard target sets are not a disjoint, complete partition of the merged catalog")

    index_path = dataset_root / "dataset_index.json"
    previous = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    split_counts = Counter(split_ids.tolist())
    previous_layout_group = previous.get("shard_layouts", {})
    previous_layout_items = (
        dict(previous_layout_group.get("items", {})) if isinstance(previous_layout_group, dict) else {}
    )
    previous_default_layout = previous.get("shards", {})
    if isinstance(previous_default_layout, dict) and int(previous_default_layout.get("count", 0)) > 0:
        previous_layout_items[str(int(previous_default_layout["count"]))] = previous_default_layout
    current_layout = {
        "count": args.shard_count,
        "assignment_method": "part_disjoint_greedy_train_then_total_v1",
        "items": shard_records,
    }
    previous_layout_items[str(args.shard_count)] = current_layout
    ordered_layout_items = {
        key: previous_layout_items[key] for key in sorted(previous_layout_items, key=lambda value: int(value))
    }
    if str(FABRICA_SHARD_COUNT) not in ordered_layout_items:
        raise ValueError(
            f"The default {FABRICA_SHARD_COUNT}-rank layout is missing. Build it before layout {args.shard_count}."
        )

    index = {
        **previous,
        "schema_version": FABRICA_DATASET_SCHEMA_VERSION,
        "dataset_name": FABRICA_DATASET_NAME,
        "build_status": "training_shards_verified",
        "training_ready": True,
        "source_selected_target_count": int(previous.get("selected_target_count", len(target_ids))),
        "selected_target_count": int(len(target_ids)),
        "validated_target_count": int(len(target_ids)),
        "represented_part_count": len(catalog_part_names),
        "split_scheme_metadata": {
            **dict(previous.get("split_scheme_metadata", {})),
            "primary": {
                **dict(previous.get("split_scheme_metadata", {}).get("primary", {})),
                "target_counts": {name: int(split_counts.get(name, 0)) for name in ("train", "validation", "test")},
            },
        },
        "artifacts": {
            **{
                key: (repo_relative(value) if isinstance(value, str) and Path(value).is_absolute() else value)
                for key, value in dict(previous.get("artifacts", {})).items()
            },
            "planned_manifest": repo_relative(planned_manifest_path),
            "merged_goal_catalog": repo_relative(source_paths["goal_catalog"]),
            "merged_paths": repo_relative(source_paths["paths"]),
            "merged_rotation_resets": repo_relative(source_paths["rotation_resets"]),
        },
        "parts": [part_metadata[name] for name in catalog_part_names],
        # Retain the original field as the stable single-rank/default layout
        # while schema-v3 indexes expose additional rank-count layouts.
        "shards": ordered_layout_items[str(FABRICA_SHARD_COUNT)],
        "shard_layouts": {
            "default_count": FABRICA_SHARD_COUNT,
            "items": ordered_layout_items,
        },
    }
    index["dataset_sha256"] = canonical_json_sha256(index)
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(
        f"[DONE] dataset={FABRICA_DATASET_NAME} targets={len(target_ids)} "
        f"parts={len(catalog_part_names)} layout={args.shard_count} "
        f"available_layouts={','.join(ordered_layout_items)} index={index_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
