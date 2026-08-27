#!/usr/bin/env python3
"""Build a leakage-safe multi-part grasp manifest for one Fabrica assembly."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_path in (REPO_ROOT, SCRIPT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from build_multigrasp_manifest import build_manifest  # noqa: E402

SCHEMA_VERSION = 4
GLOBAL_ID_SCHEMA_VERSION = 5
SPLIT_NAMES = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-name", default="plumbers_block")
    parser.add_argument("--part-ids", nargs="+", default=("0", "1", "2", "3", "4"))
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=REPO_ROOT / "isaac_rl/data/plumbers_block/sources",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "isaac_rl/data/plumbers_block/assembly_manifest.json",
    )
    parser.add_argument("--targets-per-part-orientation", type=int, default=64)
    parser.add_argument("--alternates-per-part-orientation", type=int, default=256)
    parser.add_argument("--pregrasp-offset", type=float, default=0.10)
    parser.add_argument("--gripper-width-clearance", type=float, default=0.01)
    # 66 mm is the largest final jaw width that still permits the required
    # 5 mm-per-finger approach clearance inside the PDZ 76 mm opening.
    parser.add_argument("--max-training-jaw-width", type=float, default=0.066)
    parser.add_argument("--minimum-pregrasp-height", type=float, default=0.05)
    parser.add_argument(
        "--object-xy-world",
        type=float,
        nargs=2,
        default=(0.4252643585205078, 0.05988234281539917),
        metavar=("X", "Y"),
    )
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=20260804)
    parser.add_argument(
        "--global-ids",
        action="store_true",
        help=(
            "Namespace part, orientation, and target identifiers with the assembly "
            "name. Required when manifests from multiple assemblies will be merged."
        ),
    )
    parser.add_argument(
        "--part-usd-dir",
        type=Path,
        default=None,
        help="Override the directory containing part_<id>_bundle_local.usd assets.",
    )
    parser.add_argument(
        "--allow-excluded-parts",
        action="store_true",
        help=(
            "Record parts that cannot produce a valid manifest instead of aborting "
            "the entire assembly build."
        ),
    )
    return parser.parse_args()


def _part_key(assembly_name: str, part_id: str) -> str:
    return f"{assembly_name}__part_{part_id}"


def _prefixed_orientation(
    part_id: str,
    orientation_id: str,
    *,
    assembly_name: str | None = None,
) -> str:
    prefix = f"part_{part_id}" if assembly_name is None else _part_key(assembly_name, part_id)
    return f"{prefix}__{orientation_id}"


def _prefixed_target(
    part_id: str,
    orientation_id: str,
    grasp_id: str,
    *,
    assembly_name: str | None = None,
) -> str:
    return (
        f"{_prefixed_orientation(part_id, orientation_id, assembly_name=assembly_name)}"
        f"__{grasp_id}"
    )


def _annotate_part_manifest(
    manifest: dict[str, object],
    *,
    assembly_name: str,
    part_id: str,
    part_index: int,
    part_usd_path: Path,
    global_ids: bool = False,
) -> dict[str, object]:
    payload = json.loads(json.dumps(manifest))
    namespace = assembly_name if global_ids else None
    part_key = _part_key(assembly_name, part_id)
    manifest_part_id = part_key if global_ids else part_id
    for orientation in payload["orientations"]:
        local_id = str(orientation["orientation_id"])
        orientation["local_orientation_id"] = local_id
        orientation["orientation_id"] = _prefixed_orientation(
            part_id, local_id, assembly_name=namespace
        )
        orientation["assembly_name"] = assembly_name
        orientation["part_key"] = part_key
        orientation["local_part_id"] = part_id
        orientation["part_id"] = manifest_part_id
        orientation["part_index"] = part_index
    for target in payload["targets"] + payload.get("alternates", []):
        local_orientation = str(target["orientation_id"])
        grasp_id = str(target["grasp_id"])
        target["local_orientation_id"] = local_orientation
        target["orientation_id"] = _prefixed_orientation(
            part_id, local_orientation, assembly_name=namespace
        )
        target["target_id"] = _prefixed_target(
            part_id,
            local_orientation,
            grasp_id,
            assembly_name=namespace,
        )
        target["assembly_name"] = assembly_name
        target["part_key"] = part_key
        target["local_part_id"] = part_id
        target["part_id"] = manifest_part_id
        target["part_index"] = part_index
        target["part_usd_path"] = str(part_usd_path.resolve())
    return payload


def _group_split_assignment(
    targets: list[dict[str, object]],
    *,
    coverage_targets: list[dict[str, object]] | None = None,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> tuple[dict[tuple[str, str], str], int]:
    """Split by (part, grasp), so the same local grasp never crosses a split."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--train-fraction must be between zero and one.")
    if not 0.0 < validation_fraction < 1.0 - train_fraction:
        raise ValueError(
            "--validation-fraction must be positive and leave a positive test fraction."
        )
    groups_by_part: dict[str, set[str]] = defaultdict(set)
    targets_by_orientation: dict[str, list[dict[str, object]]] = defaultdict(list)
    for target in targets:
        part_id = str(target["part_id"])
        groups_by_part[part_id].add(str(target["grasp_id"]))
    for target in coverage_targets if coverage_targets is not None else targets:
        targets_by_orientation[str(target["orientation_id"])].append(target)
    if any(len(groups) < 3 for groups in groups_by_part.values()):
        raise ValueError("Every part needs at least three distinct grasps for train/validation/test.")

    for salt in range(10000):
        assignment: dict[tuple[str, str], str] = {}
        for part_id, grasp_ids in sorted(groups_by_part.items()):
            ordered = sorted(
                grasp_ids,
                key=lambda grasp_id: hashlib.blake2b(
                    f"{seed}:{salt}:{part_id}:{grasp_id}".encode(),
                    digest_size=16,
                ).digest(),
            )
            count = len(ordered)
            validation_count = max(1, round(count * validation_fraction))
            test_count = max(1, round(count * (1.0 - train_fraction - validation_fraction)))
            train_count = count - validation_count - test_count
            if train_count < 1:
                raise ValueError(f"Part {part_id} has too few grasp groups for the requested split.")
            for index, grasp_id in enumerate(ordered):
                if index < train_count:
                    split = "train"
                elif index < train_count + validation_count:
                    split = "validation"
                else:
                    split = "test"
                assignment[(part_id, grasp_id)] = split

        # With a sufficiently large orientation stratum, insist on all three
        # splits. Retry only the deterministic hash salt; never split an exact
        # grasp across orientations to make the table look balanced.
        complete = True
        for orientation_targets in targets_by_orientation.values():
            if len(orientation_targets) < 10:
                continue
            present = {
                assignment[(str(target["part_id"]), str(target["grasp_id"]))]
                for target in orientation_targets
            }
            if present != set(SPLIT_NAMES):
                complete = False
                break
        if complete:
            return assignment, salt
    raise RuntimeError(
        "Could not find a grouped split with train/validation/test coverage in every "
        "orientation after 10,000 deterministic salts."
    )


def build_assembly_manifest(args: argparse.Namespace) -> dict[str, object]:
    if args.targets_per_part_orientation <= 0:
        raise ValueError("--targets-per-part-orientation must be positive.")
    if args.alternates_per_part_orientation < 0:
        raise ValueError("--alternates-per-part-orientation must be non-negative.")

    parts: list[dict[str, object]] = []
    orientations: list[dict[str, object]] = []
    targets: list[dict[str, object]] = []
    alternates: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for part_index, raw_part_id in enumerate(args.part_ids):
        part_id = str(raw_part_id)
        stage1 = args.source_dir / f"part_{part_id}_stage1.json"
        stage2 = args.source_dir / f"part_{part_id}_stage2.json"
        missing = [required for required in (stage1, stage2) if not required.is_file()]
        if missing:
            if bool(getattr(args, "allow_excluded_parts", False)):
                exclusions.append(
                    {
                        "assembly_name": str(args.assembly_name),
                        "local_part_id": part_id,
                        "part_key": _part_key(str(args.assembly_name), part_id),
                        "reason": "missing_planning_source",
                        "detail": ", ".join(str(path) for path in missing),
                    }
                )
                continue
            raise FileNotFoundError(
                f"Missing CPU planning source {missing[0]}. Generate it with "
                "prepare_plumbers_block_catalog.py --stage sources."
            )
        part_usd_dir = getattr(args, "part_usd_dir", None)
        part_usd_path = (
            Path(part_usd_dir) / f"part_{part_id}_bundle_local.usd"
            if part_usd_dir is not None
            else REPO_ROOT
            / "isaac_rl/data"
            / args.assembly_name
            / "usd"
            / f"part_{part_id}_bundle_local.usd"
        )
        part_args = SimpleNamespace(
            stage1_bundle=stage1,
            current_stage2_bundle=stage2,
            target_count=0,
            targets_per_orientation=int(args.targets_per_part_orientation),
            pregrasp_offset=float(args.pregrasp_offset),
            gripper_width_clearance=float(args.gripper_width_clearance),
            max_training_jaw_width=float(args.max_training_jaw_width),
            minimum_pregrasp_height=float(args.minimum_pregrasp_height),
            object_xy_world=tuple(float(value) for value in args.object_xy_world),
            alternates_per_orientation=int(args.alternates_per_part_orientation),
        )
        try:
            raw_part_manifest = build_manifest(part_args)
        except Exception as error:
            if not bool(getattr(args, "allow_excluded_parts", False)):
                raise
            exclusions.append(
                {
                    "assembly_name": str(args.assembly_name),
                    "local_part_id": part_id,
                    "part_key": _part_key(str(args.assembly_name), part_id),
                    "reason": "no_valid_training_targets",
                    "detail": str(error),
                }
            )
            continue
        part_manifest = _annotate_part_manifest(
            raw_part_manifest,
            assembly_name=args.assembly_name,
            part_id=part_id,
            part_index=part_index,
            part_usd_path=part_usd_path,
            global_ids=bool(getattr(args, "global_ids", False)),
        )
        # A stable resting orientation can legitimately have no grasp that
        # passes the gripper/ground filters. It is not a learnable stratum and
        # must not make downstream minimum-per-orientation validation fail.
        part_manifest["orientations"] = [
            orientation
            for orientation in part_manifest["orientations"]
            if int(orientation["selected_target_count"]) > 0
        ]
        parts.append(
            {
                "part_id": (
                    _part_key(str(args.assembly_name), part_id)
                    if bool(getattr(args, "global_ids", False))
                    else part_id
                ),
                "part_key": _part_key(str(args.assembly_name), part_id),
                "local_part_id": part_id,
                "assembly_name": str(args.assembly_name),
                "part_index": part_index,
                "target_mesh_path": str(part_manifest["target_mesh_path"]),
                "mesh_scale": float(part_manifest["mesh_scale"]),
                "source_stage1_bundle": str(stage1.resolve()),
                "source_current_stage2_bundle": str(stage2.resolve()),
                "part_usd_path": str(part_usd_path.resolve()),
                "orientation_count": len(part_manifest["orientations"]),
                "selected_target_count": len(part_manifest["targets"]),
                "alternate_target_count": len(part_manifest["alternates"]),
            }
        )
        orientations.extend(part_manifest["orientations"])
        targets.extend(part_manifest["targets"])
        alternates.extend(part_manifest["alternates"])

    if not targets:
        if not bool(getattr(args, "allow_excluded_parts", False)):
            raise ValueError("The assembly produced no selectable training targets.")
        assignment: dict[tuple[str, str], str] = {}
        split_salt = 0
    else:
        assignment, split_salt = _group_split_assignment(
            targets + alternates,
            coverage_targets=targets,
            train_fraction=float(args.train_fraction),
            validation_fraction=float(args.validation_fraction),
            seed=int(args.split_seed),
        )
    for target in targets:
        target["split"] = assignment[(str(target["part_id"]), str(target["grasp_id"]))]
    for target in alternates:
        target["split"] = assignment[(str(target["part_id"]), str(target["grasp_id"]))]

    split_counts = Counter(str(target["split"]) for target in targets)
    part_split_counts: dict[str, dict[str, int]] = {}
    orientation_split_counts: dict[str, dict[str, int]] = {}
    for part in parts:
        part_id = str(part["part_id"])
        part_split_counts[part_id] = {
            split: sum(
                target["part_id"] == part_id and target["split"] == split
                for target in targets
            )
            for split in SPLIT_NAMES
        }
    for orientation in orientations:
        orientation_id = str(orientation["orientation_id"])
        counts = {
            split: sum(
                target["orientation_id"] == orientation_id and target["split"] == split
                for target in targets
            )
            for split in SPLIT_NAMES
        }
        orientation["split_target_counts"] = counts
        orientation_split_counts[orientation_id] = counts

    return {
        "schema_version": (
            GLOBAL_ID_SCHEMA_VERSION
            if bool(getattr(args, "global_ids", False))
            else SCHEMA_VERSION
        ),
        "assembly_name": str(args.assembly_name),
        "part_count": len(parts),
        "configured_part_count": len(args.part_ids),
        "parts": parts,
        "exclusions": exclusions,
        "selection": {
            "selected_target_count": len(targets),
            "alternate_target_count": len(alternates),
            "targets_per_part_orientation_cap": int(args.targets_per_part_orientation),
            "alternates_per_part_orientation_cap": int(
                args.alternates_per_part_orientation
            ),
            "method": "per_part_orientation_diverse_farthest_point",
            "moveit_validation_required": True,
            "isaac_goal_capture_required": True,
        },
        "split": {
            "names": list(SPLIT_NAMES),
            "unit": "part_id_and_grasp_id_group",
            "seed": int(args.split_seed),
            "salt": split_salt,
            "requested_fractions": {
                "train": float(args.train_fraction),
                "validation": float(args.validation_fraction),
                "test": float(1.0 - args.train_fraction - args.validation_fraction),
            },
            "target_counts": {name: int(split_counts[name]) for name in SPLIT_NAMES},
            "part_target_counts": part_split_counts,
            "orientation_target_counts": orientation_split_counts,
        },
        "orientations": orientations,
        "targets": targets,
        "alternates": alternates,
    }


def main() -> None:
    args = parse_args()
    manifest = build_assembly_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    split_counts = manifest["split"]["target_counts"]
    print(
        f"[DONE] Wrote {len(manifest['targets'])} targets for {len(manifest['parts'])} "
        f"parts to {args.output}.",
        flush=True,
    )
    print(
        f"[SPLIT] train={split_counts['train']} validation={split_counts['validation']} "
        f"test={split_counts['test']} grouped by (part_id, grasp_id).",
        flush=True,
    )
    for part in manifest["parts"]:
        counts = manifest["split"]["part_target_counts"][str(part["part_id"])]
        print(
            f"  part {part['part_id']}: targets={part['selected_target_count']} "
            f"orientations={part['orientation_count']} splits={counts}",
            flush=True,
        )


if __name__ == "__main__":
    main()
