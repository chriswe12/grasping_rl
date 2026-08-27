#!/usr/bin/env python3
"""Merge globally namespaced per-assembly Fabrica grasp manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SCHEMA_VERSION = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260804)
    parser.add_argument("--held-out-part-fraction", type=float, default=0.20)
    parser.add_argument("--held-out-assembly", action="append", default=[])
    return parser.parse_args()


def _hash_order(value: str, seed: int, namespace: str) -> bytes:
    return hashlib.blake2b(
        f"{namespace}:{seed}:{value}".encode(), digest_size=16
    ).digest()


def _assert_unique(records: list[dict[str, object]], field: str) -> None:
    values = [str(record[field]) for record in records]
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate {field} values: {duplicates[:8]}")


def _held_out_parts(
    part_keys: list[str], *, fraction: float, seed: int
) -> set[str]:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("held-out part fraction must be in [0, 1).")
    if not part_keys or fraction == 0.0:
        return set()
    count = max(1, round(len(part_keys) * fraction))
    ordered = sorted(part_keys, key=lambda key: _hash_order(key, seed, "part_holdout"))
    return set(ordered[:count])


def merge_manifests(
    manifest_paths: list[Path],
    *,
    split_seed: int,
    held_out_part_fraction: float,
    held_out_assemblies: set[str],
) -> dict[str, object]:
    loaded: list[tuple[Path, dict[str, object]]] = []
    for path in manifest_paths:
        resolved = path.expanduser().resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) < 5:
            raise ValueError(
                f"'{resolved}' does not use globally namespaced manifest IDs."
            )
        loaded.append((resolved, payload))
    loaded.sort(key=lambda item: str(item[1].get("assembly_name", "")))
    assembly_names = [str(payload["assembly_name"]) for _, payload in loaded]
    if len(set(assembly_names)) != len(assembly_names):
        raise ValueError("Each assembly may appear in exactly one input manifest.")
    unknown_holdouts = held_out_assemblies - set(assembly_names)
    if unknown_holdouts:
        raise ValueError(f"Unknown held-out assemblies: {sorted(unknown_holdouts)}")
    requested_fractions = dict(loaded[0][1].get("split", {}).get("requested_fractions", {}))
    if not requested_fractions:
        requested_fractions = {"train": 0.8, "validation": 0.1, "test": 0.1}

    parts: list[dict[str, object]] = []
    orientations: list[dict[str, object]] = []
    targets: list[dict[str, object]] = []
    alternates: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for path, payload in loaded:
        sources.append(
            {
                "assembly_name": str(payload["assembly_name"]),
                "manifest_path": str(path),
                "schema_version": int(payload["schema_version"]),
            }
        )
        parts.extend(dict(item) for item in payload.get("parts", []))
        orientations.extend(dict(item) for item in payload.get("orientations", []))
        targets.extend(dict(item) for item in payload.get("targets", []))
        alternates.extend(dict(item) for item in payload.get("alternates", []))
        exclusions.extend(dict(item) for item in payload.get("exclusions", []))

    parts.sort(key=lambda item: (str(item["assembly_name"]), int(item["local_part_id"])))
    part_index = {str(part["part_key"]): index for index, part in enumerate(parts)}
    for part in parts:
        part["part_index"] = part_index[str(part["part_key"])]
    orientations.sort(
        key=lambda item: (
            part_index[str(item["part_key"])],
            str(item["local_orientation_id"]),
        )
    )
    orientation_index = {
        str(orientation["orientation_id"]): index
        for index, orientation in enumerate(orientations)
    }
    for orientation in orientations:
        orientation["part_index"] = part_index[str(orientation["part_key"])]
        orientation["orientation_index"] = orientation_index[
            str(orientation["orientation_id"])
        ]
    for collection in (targets, alternates):
        collection.sort(
            key=lambda item: (
                orientation_index[str(item["orientation_id"])],
                int(item.get("orientation_selection_rank", 0)),
                str(item["grasp_id"]),
            )
        )
        for target in collection:
            target["part_index"] = part_index[str(target["part_key"])]
            target["orientation_index"] = orientation_index[
                str(target["orientation_id"])
            ]

    _assert_unique(parts, "part_key")
    _assert_unique(orientations, "orientation_id")
    _assert_unique(targets + alternates, "target_id")
    selected_ids = {str(target["target_id"]) for target in targets}
    alternate_ids = {str(target["target_id"]) for target in alternates}
    if selected_ids & alternate_ids:
        raise ValueError("Selected and alternate target IDs overlap.")

    holdout_parts = _held_out_parts(
        [str(part["part_key"]) for part in parts],
        fraction=held_out_part_fraction,
        seed=split_seed,
    )
    for target in targets + alternates:
        target["part_holdout_split"] = (
            "test" if str(target["part_key"]) in holdout_parts else "train"
        )
        target["assembly_holdout_split"] = (
            "test"
            if str(target["assembly_name"]) in held_out_assemblies
            else "train"
        )

    primary_counts = Counter(str(target["split"]) for target in targets)
    part_holdout_counts = Counter(str(target["part_holdout_split"]) for target in targets)
    assembly_holdout_counts = Counter(
        str(target["assembly_holdout_split"]) for target in targets
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": "fabrica_all_v1",
        "assembly_name": "fabrica_all_v1",
        "assembly_count": len(assembly_names),
        "assemblies": assembly_names,
        "configured_part_count": sum(
            int(payload.get("configured_part_count", len(payload.get("parts", []))))
            for _, payload in loaded
        ),
        "part_count": len(parts),
        "parts": parts,
        "exclusions": exclusions,
        "selection": {
            "selected_target_count": len(targets),
            "alternate_target_count": len(alternates),
        },
        "split": {
            "names": ["train", "validation", "test"],
            "unit": "assembly_name_local_part_id_grasp_id_group",
            "seed": split_seed,
            "requested_fractions": requested_fractions,
            "target_counts": {
                name: int(primary_counts[name])
                for name in ("train", "validation", "test")
            },
        },
        "split_scheme_metadata": {
            "primary": {
                "name": "held_out_grasps",
                "field": "split",
                "unit": "assembly_name_local_part_id_grasp_id_group",
                "seed": split_seed,
                "target_counts": dict(primary_counts),
            },
            "part_holdout": {
                "name": "held_out_parts",
                "field": "part_holdout_split",
                "seed": split_seed,
                "held_out_part_keys": sorted(holdout_parts),
                "target_counts": dict(part_holdout_counts),
            },
            "assembly_holdout": {
                "name": "held_out_assemblies",
                "field": "assembly_holdout_split",
                "held_out_assemblies": sorted(held_out_assemblies),
                "target_counts": dict(assembly_holdout_counts),
            },
        },
        "sources": sources,
        "orientations": orientations,
        "targets": targets,
        "alternates": alternates,
    }


def main() -> None:
    args = parse_args()
    merged = merge_manifests(
        list(args.manifest),
        split_seed=int(args.split_seed),
        held_out_part_fraction=float(args.held_out_part_fraction),
        held_out_assemblies={str(value) for value in args.held_out_assembly},
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(
        f"[DONE] Merged {merged['assembly_count']} assemblies, {merged['part_count']} "
        f"parts, and {merged['selection']['selected_target_count']} selected targets "
        f"into {output}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
