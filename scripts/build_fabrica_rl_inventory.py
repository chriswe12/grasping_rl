#!/usr/bin/env python3
"""Build the deterministic source inventory for a versioned Fabrica RL dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/fabrica_all_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"'{path}' did not load as a single triangle mesh.")
    if loaded.vertices.ndim != 2 or loaded.vertices.shape[1] != 3:
        raise ValueError(f"'{path}' has an invalid vertex array.")
    if loaded.faces.ndim != 2 or loaded.faces.shape[1] != 3:
        raise ValueError(f"'{path}' has an invalid triangle array.")
    if not np.isfinite(loaded.vertices).all():
        raise ValueError(f"'{path}' contains non-finite vertices.")
    if len(loaded.vertices) < 4 or len(loaded.faces) < 4:
        raise ValueError(f"'{path}' is too small to be a supported solid part.")
    return loaded


def build_inventory(config_path: Path) -> dict[str, object]:
    config_path = config_path.expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = dict(payload.get("dataset", {}))
    assemblies = dict(payload.get("assemblies", {}))
    inventory_cfg = dict(payload.get("inventory", {}))
    mesh_scale = float(dataset.get("mesh_scale", 0.01))
    if mesh_scale <= 0.0:
        raise ValueError("dataset.mesh_scale must be positive.")
    mesh_root = Path(str(dataset.get("mesh_root", "assets/obj/fabrica")))
    if not mesh_root.is_absolute():
        mesh_root = (REPO_ROOT / mesh_root).resolve()
    if not mesh_root.is_dir():
        raise FileNotFoundError(mesh_root)

    threshold = int(inventory_cfg.get("simplified_mesh_face_threshold", 50000))
    records: list[dict[str, object]] = []
    exclusions: list[dict[str, str]] = []
    expected_keys: set[str] = set()
    for assembly_name in sorted(str(name) for name in assemblies):
        raw_ids = assemblies[assembly_name]
        part_ids = sorted((str(value) for value in raw_ids), key=lambda value: (int(value), value))
        for local_part_id in part_ids:
            part_key = f"{assembly_name}__part_{local_part_id}"
            if part_key in expected_keys:
                raise ValueError(f"Duplicate configured part key '{part_key}'.")
            expected_keys.add(part_key)
            source = mesh_root / assembly_name / f"{local_part_id}.obj"
            if not source.is_file():
                exclusions.append(
                    {
                        "part_key": part_key,
                        "assembly_name": assembly_name,
                        "local_part_id": local_part_id,
                        "reason": "missing_source_mesh",
                        "source_path": str(source),
                    }
                )
                continue
            try:
                mesh = _load_mesh(source)
            except Exception as error:  # inventory records data failures instead of hiding them
                exclusions.append(
                    {
                        "part_key": part_key,
                        "assembly_name": assembly_name,
                        "local_part_id": local_part_id,
                        "reason": "invalid_source_mesh",
                        "detail": str(error),
                        "source_path": str(source),
                    }
                )
                continue
            bounds_source = np.asarray(mesh.bounds, dtype=float)
            bounds_m = bounds_source * mesh_scale
            extents_m = bounds_m[1] - bounds_m[0]
            records.append(
                {
                    "part_key": part_key,
                    "assembly_name": assembly_name,
                    "local_part_id": local_part_id,
                    "source_path": source.relative_to(REPO_ROOT).as_posix(),
                    "source_sha256": _sha256(source),
                    "source_size_bytes": source.stat().st_size,
                    "mesh_scale": mesh_scale,
                    "vertex_count": int(len(mesh.vertices)),
                    "face_count": int(len(mesh.faces)),
                    "bounds_source": bounds_source.tolist(),
                    "bounds_m": bounds_m.tolist(),
                    "extents_m": extents_m.tolist(),
                    "largest_extent_m": float(np.max(extents_m)),
                    "watertight": bool(mesh.is_watertight),
                    "planning_mesh_path": source.relative_to(REPO_ROOT).as_posix(),
                    "planning_mesh_kind": "source",
                    "simplification_recommended": bool(len(mesh.faces) > threshold),
                }
            )

    discovered = {
        f"{path.parent.name}__part_{path.stem}"
        for path in mesh_root.glob("*/*.obj")
        if path.is_file() and path.stem.isdigit()
    }
    unconfigured = sorted(discovered - expected_keys)
    return {
        "schema_version": 1,
        "dataset_name": str(dataset.get("name", "fabrica_all_v1")),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_config": config_path.relative_to(REPO_ROOT).as_posix(),
        "source_config_sha256": _config_hash(payload),
        "mesh_root": mesh_root.relative_to(REPO_ROOT).as_posix(),
        "mesh_scale": mesh_scale,
        "configured_part_count": len(expected_keys),
        "valid_part_count": len(records),
        "excluded_part_count": len(exclusions),
        "unconfigured_discovered_part_keys": unconfigured,
        "assemblies": sorted(str(name) for name in assemblies),
        "parts": records,
        "exclusions": exclusions,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    inventory = build_inventory(config_path)
    if args.output is None:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        output_root = Path(str(config["dataset"]["output_root"]))
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root
        output = output_root / "inventory.json"
    else:
        output = args.output.expanduser()
        if not output.is_absolute():
            output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(
        f"[DONE] Inventoried {inventory['valid_part_count']}/"
        f"{inventory['configured_part_count']} configured parts in "
        f"{len(inventory['assemblies'])} assemblies: {output}",
        flush=True,
    )
    if inventory["exclusions"]:
        print(f"[EXCLUDED] {len(inventory['exclusions'])} parts; see inventory.json.")


if __name__ == "__main__":
    main()
