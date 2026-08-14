#!/usr/bin/env python3
"""Convert every assembly part's saved bundle-local mesh to a collision USD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--manifest",
    type=Path,
    default=Path("isaac_rl/data/plumbers_block/planned_manifest.json"),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg  # noqa: E402
from isaaclab.sim.schemas import schemas_cfg  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping.fabrica_grasp_debug import load_grasp_bundle  # noqa: E402
from grasp_planning.mujoco import (  # noqa: E402
    build_bundle_local_mesh,
    write_temporary_triangle_mesh_stl,
)

ISAAC_MIN_CONTACT_OFFSET_M = 1.0e-5


def main() -> None:
    manifest_path = args_cli.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("parts"):
        raise ValueError(f"Manifest contains no part records: {manifest_path}")
    for part in manifest["parts"]:
        bundle_path = Path(part["source_current_stage2_bundle"]).expanduser().resolve()
        output_usd = Path(part["part_usd_path"]).expanduser().resolve()
        bundle = load_grasp_bundle(bundle_path)
        mesh_local = build_bundle_local_mesh(bundle)
        output_usd.parent.mkdir(parents=True, exist_ok=True)
        temporary_stl = write_temporary_triangle_mesh_stl(
            mesh_local,
            prefix=f"part_{part['part_id']}_bundle_local_",
            dir=output_usd.parent,
        )
        try:
            converter = MeshConverter(
                MeshConverterCfg(
                    asset_path=str(temporary_stl),
                    usd_dir=str(output_usd.parent),
                    usd_file_name=output_usd.name,
                    force_usd_conversion=True,
                    make_instanceable=False,
                    scale=(1.0, 1.0, 1.0),
                    mass_props=sim_utils.MassPropertiesCfg(density=1240.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=False,
                        disable_gravity=False,
                        max_depenetration_velocity=5.0,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True,
                        contact_offset=ISAAC_MIN_CONTACT_OFFSET_M,
                        rest_offset=0.0,
                    ),
                    mesh_collision_props=schemas_cfg.ConvexDecompositionPropertiesCfg(),
                )
            )
        finally:
            temporary_stl.unlink(missing_ok=True)
        converted = Path(converter.usd_path).resolve()
        if converted != output_usd:
            raise RuntimeError(
                f"Mesh converter wrote {converted}, expected manifest path {output_usd}."
            )
        print(
            f"[USD] part={part['part_id']} bundle={bundle_path.name} output={output_usd}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
