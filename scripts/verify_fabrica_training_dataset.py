#!/usr/bin/env python3
"""Verify the portable Fabrica-all dataset using only the Python standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Dataset contains a non-portable absolute path: {value}")
    resolved = (root / path).resolve()
    resolved.relative_to(root)
    return resolved


def _canonical_index_hash(index: dict[str, object]) -> str:
    payload = dict(index)
    payload.pop("dataset_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _shard_layouts(index: dict[str, object]) -> tuple[int, dict[int, dict[str, object]]]:
    legacy = index.get("shards", {})
    if not isinstance(legacy, dict):
        raise ValueError("Fabrica-all shards record is invalid")
    legacy_count = int(legacy.get("count", 0))
    layouts: dict[int, dict[str, object]] = {}
    if legacy_count > 0:
        layouts[legacy_count] = legacy

    group = index.get("shard_layouts", {})
    if group:
        if not isinstance(group, dict) or not isinstance(group.get("items", {}), dict):
            raise ValueError("Fabrica-all shard_layouts record is invalid")
        for count_text, layout in group["items"].items():
            if not isinstance(layout, dict):
                raise ValueError(f"Fabrica-all shard layout {count_text!r} is invalid")
            count = int(count_text)
            if int(layout.get("count", -1)) != count:
                raise ValueError(f"Fabrica-all shard layout {count} has a mismatched count")
            layouts[count] = layout
        default_count = int(group.get("default_count", legacy_count))
    else:
        default_count = legacy_count

    if default_count not in layouts or legacy != layouts[default_count]:
        raise ValueError("Fabrica-all default shards record does not match its named layout")
    return default_count, layouts


def _verify_layout(
    *,
    root: Path,
    index: dict[str, object],
    rank_count: int,
    layout: dict[str, object],
) -> tuple[int, set[str]]:
    shard_records = layout.get("items", [])
    if not isinstance(shard_records, list) or len(shard_records) != rank_count:
        raise ValueError(f"Fabrica-all {rank_count}-rank shard count mismatch")
    all_parts: set[str] = set()
    total_targets = 0
    for expected_index, record in enumerate(shard_records):
        if int(record["shard_index"]) != expected_index:
            raise ValueError(f"Shard records for layout {rank_count} are not in deterministic index order")
        manifest_path = _resolve(root, record["manifest"])
        if _sha256(manifest_path) != record["manifest_sha256"]:
            raise ValueError(f"Shard manifest checksum mismatch: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("shard_count", rank_count)) != rank_count:
            raise ValueError(f"Shard manifest rank count mismatch: {manifest_path}")
        parts = set(str(value) for value in manifest["part_names"])
        if all_parts.intersection(parts):
            raise ValueError(f"A part appears in more than one Fabrica shard for layout {rank_count}")
        all_parts.update(parts)
        total_targets += int(manifest["target_count"])
        for artifact in manifest["artifacts"].values():
            path = _resolve(root, artifact["path"])
            if path.stat().st_size != int(artifact["size_bytes"]) or _sha256(path) != artifact["sha256"]:
                raise ValueError(f"Shard artifact checksum mismatch: {path}")
        for part in manifest["parts"]:
            usd = _resolve(root, part["usd_path"])
            if usd.stat().st_size != int(part["usd_size_bytes"]) or _sha256(usd) != part["usd_sha256"]:
                raise ValueError(f"Part USD checksum mismatch: {usd}")
    if total_targets != int(index["validated_target_count"]):
        raise ValueError(f"Shard target total for layout {rank_count} does not match dataset index")
    if len(all_parts) != int(index["represented_part_count"]):
        raise ValueError(f"Shard part union for layout {rank_count} does not match dataset index")
    return total_targets, all_parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--index",
        default="isaac_rl/data/fabrica_all_v1/dataset_index.json",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    index_path = _resolve(root, args.index)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not index.get("training_ready") or index.get("build_status") != "training_shards_verified":
        raise ValueError("Fabrica-all dataset is not marked as verified/training-ready")
    if index.get("dataset_sha256") != _canonical_index_hash(index):
        raise ValueError("Fabrica-all dataset index checksum is stale")

    default_count, layouts = _shard_layouts(index)
    expected_parts: set[str] | None = None
    total_targets = 0
    for rank_count, layout in sorted(layouts.items()):
        layout_targets, layout_parts = _verify_layout(
            root=root,
            index=index,
            rank_count=rank_count,
            layout=layout,
        )
        total_targets = layout_targets
        if expected_parts is None:
            expected_parts = layout_parts
        elif layout_parts != expected_parts:
            raise ValueError("Fabrica-all shard layouts do not represent the same part set")
    print(
        f"[OK] Fabrica dataset {index['dataset_name']}: targets={total_targets} "
        f"parts={len(expected_parts or ())} default_layout={default_count} "
        f"layouts={','.join(str(count) for count in sorted(layouts))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
