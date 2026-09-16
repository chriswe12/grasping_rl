#!/usr/bin/env python3
"""Render the validated multipart PDZ goal RGB-D catalog with MuJoCo Filament."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh

# The validated experimental Filament backend runs without an OpenGL context.
# MuJoCo's Python renderer still imports this symbol in that mode on the local
# build, so provide the same harmless sentinel as the renderer proof script.
from mujoco.rendering.classic import gl_context
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

if not hasattr(gl_context, "GLContext"):
    gl_context.GLContext = None

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_path in (REPO_ROOT, SCRIPT_ROOT, REPO_ROOT / "isaac_rl/source/isaac_rl"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from build_reset_trajectory_asset import (  # noqa: E402
    MOVEIT_TO_ISAAC_SIGNS,
    _robot_tcp_transform_link7,
)
from grasp_planning.d405_wrist_camera import (  # noqa: E402
    D405_VISUAL_SERVO_CAMERA_PROFILE,
    D405_VISUAL_SERVO_OBSERVATION_PROFILE,
    VISUAL_SERVO_RENDER_HEIGHT,
    VISUAL_SERVO_RENDER_WIDTH,
    D405WristCameraConfig,
    camera_pose_in_link7,
)
from grasp_planning.grasping.fabrica_grasp_debug import load_grasp_bundle  # noqa: E402
from grasp_planning.isaac_visual_materials import (  # noqa: E402
    VISUAL_SERVO_MATERIAL_PROFILE,
    VISUAL_SERVO_PART_PALETTE,
)
from grasp_planning.isaac_visual_scene import VISUAL_SERVO_SCENE_PROFILE  # noqa: E402
from grasp_planning.mujoco import build_bundle_local_mesh  # noqa: E402
from grasp_planning.rl.goal_catalog_profiles import (  # noqa: E402
    GOAL_FILAMENT_MATERIALS,
    MUJOCO_GOAL_RENDERER_BACKEND,
    MUJOCO_GOAL_RENDERER_PROFILE,
)
from grasp_planning.start_poses import (  # noqa: E402
    PDZ_GRIPPER_APPROACH_PROFILE,
    PDZ_GRIPPER_CLOSED_WIDTH_M,
    PDZ_GRIPPER_TRAVEL_M,
    VISUAL_SERVO_GRIPPER_PROFILE,
)
from grasp_planning.visual_servo_workspace import VISUAL_SERVO_TSLOT_PROFILE  # noqa: E402

# Import this leaf validator directly. Importing it through the ``isaac_rl``
# package eagerly imports Isaac Lab tasks, while this renderer deliberately
# runs in lightweight system Python with MuJoCo/Filament only.
_CATALOG_MODULE_PATH = (
    REPO_ROOT
    / "isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/multigrasp_catalog.py"
)
_catalog_spec = importlib.util.spec_from_file_location(
    "_pdz_multigrasp_catalog", _CATALOG_MODULE_PATH
)
if _catalog_spec is None or _catalog_spec.loader is None:
    raise ImportError(f"Cannot load catalog validator from {_CATALOG_MODULE_PATH}.")
_catalog_module = importlib.util.module_from_spec(_catalog_spec)
_catalog_spec.loader.exec_module(_catalog_module)
load_multigrasp_catalog = _catalog_module.load_multigrasp_catalog

WIDTH = VISUAL_SERVO_RENDER_WIDTH
HEIGHT = VISUAL_SERVO_RENDER_HEIGHT

_PDZ_GRIPPER_VISUAL_MESHES = (
    (
        "pdz_gripper_left_finger_link",
        None,
        "left_finger",
        "pdz_visual_left_finger",
        "left_finger.stl",
    ),
    (
        "pdz_gripper_left_finger_link",
        "left_tpu_pad",
        "left_pad_8mm",
        "pdz_visual_left_pad",
        "left_pad_8mm.stl",
    ),
    (
        "pdz_gripper_right_finger_link",
        None,
        "right_finger",
        "pdz_visual_right_finger",
        "right_finger.stl",
    ),
    (
        "pdz_gripper_right_finger_link",
        "right_tpu_pad",
        "right_pad_8mm",
        "pdz_visual_right_pad",
        "right_pad_8mm.stl",
    ),
)

FILAMENT_FALLBACK_HEAD_LIGHT_INTENSITY = 1000.0
FILAMENT_FALLBACK_ENVIRONMENT_LIGHT_INTENSITY = 6500.0

# Exact dimensions of assets/usd/visual_servo/t_slot_surface_nominal.usda.
# MuJoCo box sizes are half-extents, so keep the physical dimensions named here
# instead of obscuring the parity in hard-coded MJCF half-size literals.
CANONICAL_TSLOT_PITCH_M = 0.0255
CANONICAL_TSLOT_LAND_WIDTH_M = 0.0205
CANONICAL_TSLOT_SLOT_WIDTH_M = (
    CANONICAL_TSLOT_PITCH_M - CANONICAL_TSLOT_LAND_WIDTH_M
)
CANONICAL_TSLOT_LAND_COUNT = 25
CANONICAL_TSLOT_SURFACE_SIZE_M = (0.65, 0.60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths-asset",
        type=Path,
        default=REPO_ROOT / "isaac_rl/data/plumbers_block/paths.npz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "isaac_rl/data/plumbers_block/planned_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "isaac_rl/data/plumbers_block/goal_catalog.npz",
    )
    parser.add_argument(
        "--robot-urdf",
        type=Path,
        default=REPO_ROOT
        / "assets/urdf/kuka_iiwa7_pdz_gripper/urdf/kuka_iiwa7_pdz_gripper.urdf",
    )
    parser.add_argument("--renderer-backend", choices=("filament", "classic"), default="filament")
    parser.add_argument("--maximum-position-error-m", type=float, default=0.00005)
    parser.add_argument("--maximum-rotation-error-deg", type=float, default=0.03)
    parser.add_argument("--minimum-depth-std-m", type=float, default=0.01)
    parser.add_argument("--target-indices", type=int, nargs="+", default=None)
    parser.add_argument("--contact-sheet", type=Path, default=None)
    parser.add_argument(
        "--goal-palette-indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Render true color-conditioned goal variants at 128x72. Recommended balanced subset: "
            "0 2 3 9 19 23. Omit to preserve the legacy single-goal-color artifact size."
        ),
    )
    return parser.parse_args()


def _atomic_savez(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".json", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _select_targets(
    payload: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, np.ndarray]:
    source_count = len(payload["target_ids"])
    return {
        name: value[indices].copy()
        if value.ndim > 0 and value.shape[0] == source_count
        else value.copy()
        for name, value in payload.items()
    }


def _ros_camera_quat_to_opengl(quat_wxyz: np.ndarray) -> np.ndarray:
    rotation_ros = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()
    rotation_gl = rotation_ros @ np.diag([1.0, -1.0, -1.0])
    quat_xyzw = Rotation.from_matrix(rotation_gl).as_quat()
    return quat_xyzw[[3, 0, 1, 2]]


def _write_contact_sheet(
    *, path: Path, rgb: np.ndarray, target_ids: np.ndarray
) -> None:
    columns = min(5, len(rgb))
    tile_width, tile_height, label_height = 384, 216, 30
    rows = int(math.ceil(len(rgb) / columns))
    sheet = Image.new(
        "RGB", (columns * tile_width, rows * (tile_height + label_height)), (20, 23, 28)
    )
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15
        )
    except OSError:
        font = ImageFont.load_default()
    for index, image in enumerate(rgb):
        column, row = index % columns, index // columns
        x, y = column * tile_width, row * (tile_height + label_height)
        tile = Image.fromarray(image).resize(
            (tile_width, tile_height), Image.Resampling.LANCZOS
        )
        sheet.paste(tile, (x, y))
        draw.text(
            (x + 7, y + tile_height + 6),
            str(target_ids[index]),
            fill=(240, 242, 245),
            font=font,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _build_bundle_mesh(part: dict[str, object], output: Path) -> None:
    bundle_path = Path(str(part["source_current_stage2_bundle"])).expanduser()
    if not bundle_path.is_file():
        bundle_path = (
            REPO_ROOT
            / "isaac_rl/data/plumbers_block/sources"
            / f"part_{part['part_id']}_stage2.json"
        )
    mesh = build_bundle_local_mesh(load_grasp_bundle(bundle_path))
    trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices_obj, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    ).export(output)


def _add_material(
    asset: ET.Element,
    *,
    name: str,
    color: tuple[float, float, float],
    metallic: float,
    roughness: float,
    emission: float = 0.0,
) -> None:
    ET.SubElement(
        asset,
        "material",
        name=name,
        rgba=" ".join(str(value) for value in (*color, 1.0)),
        specular="0.05",
        shininess="0.05",
        metallic=str(metallic),
        roughness=str(roughness),
        emission=str(emission),
    )


def _author_canonical_tslot_surface(worldbody: ET.Element) -> None:
    """Author the Isaac T-slot asset at exactly the same metric dimensions."""

    center_x, center_y = 0.425, 0.05
    ET.SubElement(
        worldbody,
        "geom",
        name="tslot_backing",
        type="box",
        size=(
            f"{0.5 * CANONICAL_TSLOT_SURFACE_SIZE_M[0]:.12g} "
            f"{0.5 * CANONICAL_TSLOT_SURFACE_SIZE_M[1]:.12g} 0.002"
        ),
        pos=f"{center_x:.12g} {center_y:.12g} -0.009",
        material="tslot_slot",
        contype="0",
        conaffinity="0",
    )
    for index in range(CANONICAL_TSLOT_LAND_COUNT):
        x = center_x + (index - CANONICAL_TSLOT_LAND_COUNT // 2) * CANONICAL_TSLOT_PITCH_M
        ET.SubElement(
            worldbody,
            "geom",
            name=f"tslot_land_{index:02d}",
            type="box",
            size=(
                f"{0.5 * CANONICAL_TSLOT_LAND_WIDTH_M:.12g} "
                f"{0.5 * CANONICAL_TSLOT_SURFACE_SIZE_M[1]:.12g} 0.0015"
            ),
            pos=f"{x:.12g} {center_y:.12g} -0.003",
            material="tslot_aluminum",
            contype="0",
            conaffinity="0",
        )


def _export_robot_mjcf(robot_urdf: Path, output_mjcf: Path) -> None:
    """Run URDF import without the experimental Filament preload.

    The experimental renderer is only needed while compiling and rendering the
    final scene.  Its current Python binding corrupts MjSpec string iteration,
    so canonicalize the URDF in a short stock-MuJoCo subprocess first.
    """

    environment = os.environ.copy()
    environment.pop("LD_PRELOAD", None)
    environment.pop("MUJOCO_FILAMENT_ACTIVE", None)
    environment.pop("MUJOCO_FILAMENT_ASSETS_DIR", None)
    environment.pop("VK_ICD_FILENAMES", None)
    environment["MUJOCO_GL"] = "disable"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_ROOT / "export_robot_urdf_mjcf.py"),
            str(robot_urdf),
            str(output_mjcf),
        ],
        check=True,
        env=environment,
    )


def _set_custom_numeric(root: ET.Element, name: str, value: float) -> None:
    custom = root.find("custom")
    if custom is None:
        custom = ET.SubElement(root, "custom")
    numeric = custom.find(f"numeric[@name='{name}']")
    if numeric is None:
        numeric = ET.SubElement(custom, "numeric", name=name)
    numeric.set("data", f"{value:.12g}")


def _restore_pdz_gripper_visual_meshes(
    root: ET.Element,
    robot_urdf: Path,
) -> None:
    """Replace collision-only URDF imports with the authored visual meshes."""
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise RuntimeError("Canonical robot MJCF has no asset or worldbody element.")
    visual_mesh_dir = robot_urdf.parent.parent / "meshes" / "visual"
    for body_name, geom_name, imported_mesh, visual_mesh, filename in (
        _PDZ_GRIPPER_VISUAL_MESHES
    ):
        mesh_path = (visual_mesh_dir / filename).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"PDZ gripper visual mesh not found: {mesh_path}")
        ET.SubElement(
            asset,
            "mesh",
            name=visual_mesh,
            file=str(mesh_path),
            scale="0.001 0.001 0.001",
        )
        body = worldbody.find(f".//body[@name='{body_name}']")
        if body is None:
            raise RuntimeError(f"Canonical MJCF has no {body_name} body.")
        matching_geoms = [
            geom
            for geom in body.findall("geom")
            if (
                geom.get("name") == geom_name
                if geom_name is not None
                else geom.get("name") is None and geom.get("mesh") == imported_mesh
            )
        ]
        if len(matching_geoms) != 1:
            raise RuntimeError(
                f"Expected one imported {body_name}/{geom_name or imported_mesh} geom, "
                f"found {len(matching_geoms)}."
            )
        matching_geoms[0].set("mesh", visual_mesh)


def _author_filament_fallback_lighting(root: ET.Element) -> None:
    """Use Filament IBL plus a camera fill, without hard cast shadows."""
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.attrib.update(
        active="1",
        ambient="0 0 0",
        diffuse="1 0.98 0.95",
        specular="0.05 0.05 0.05",
    )
    _set_custom_numeric(root, "filament.ao.enabled", 0.0)
    _set_custom_numeric(
        root,
        "filament.fallback.head_light_intensity",
        FILAMENT_FALLBACK_HEAD_LIGHT_INTENSITY,
    )
    _set_custom_numeric(
        root,
        "filament.fallback.environment_light_intensity",
        FILAMENT_FALLBACK_ENVIRONMENT_LIGHT_INTENSITY,
    )


def _scene_model(robot_mjcf: Path, robot_urdf: Path, part_mesh: Path) -> mujoco.MjModel:
    root = ET.parse(robot_mjcf).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Canonical robot MJCF has no worldbody.")
    link7_xml = worldbody.find(".//body[@name='link7']")
    if link7_xml is None:
        raise RuntimeError("Canonical robot MJCF has no link7 body.")
    camera_cfg = D405WristCameraConfig(enabled=True)
    camera_position, camera_quat_ros = camera_pose_in_link7(camera_cfg)
    camera_quat_gl = _ros_camera_quat_to_opengl(
        np.asarray(camera_quat_ros, dtype=np.float64)
    )
    scale_x = WIDTH / float(camera_cfg.width)
    scale_y = HEIGHT / float(camera_cfg.height)
    for old_camera in link7_xml.findall("camera[@name='d405']"):
        link7_xml.remove(old_camera)
    ET.SubElement(
        link7_xml,
        "camera",
        name="d405",
        pos=" ".join(f"{value:.12g}" for value in camera_position),
        quat=" ".join(f"{value:.12g}" for value in camera_quat_gl),
        resolution=f"{WIDTH} {HEIGHT}",
        sensorsize=f"{WIDTH} {HEIGHT}",
        focalpixel=(
            f"{camera_cfg.fx * scale_x:.12g} {camera_cfg.fy * scale_y:.12g}"
        ),
        principalpixel=(
            f"{camera_cfg.cx * scale_x - WIDTH / 2.0:.12g} "
            f"{HEIGHT / 2.0 - camera_cfg.cy * scale_y:.12g}"
        ),
    )
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str(robot_urdf.parent))
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="selected_part_mesh", file=str(part_mesh))
    for name, material in GOAL_FILAMENT_MATERIALS.items():
        _add_material(
            asset,
            name=name,
            color=material.color,
            metallic=material.metallic,
            roughness=material.roughness,
            emission=material.emission,
        )
    removed_box = removed_base = False
    for geom in list(link7_xml.findall("geom")):
        if geom.get("type") == "box" and geom.get("size") == "0.0115 0.021 0.021":
            link7_xml.remove(geom)
            removed_box = True
        elif geom.get("type") == "mesh" and geom.get("mesh") == "base":
            link7_xml.remove(geom)
            removed_base = True
    if not (removed_box and removed_base):
        raise RuntimeError(
            "Could not remove the camera enclosure surfaces that contain the optical origin."
        )
    _restore_pdz_gripper_visual_meshes(root, robot_urdf)
    for finger_name in (
        "pdz_gripper_left_finger_link",
        "pdz_gripper_right_finger_link",
    ):
        finger = worldbody.find(f".//body[@name='{finger_name}']")
        if finger is None:
            raise RuntimeError(f"Canonical MJCF has no {finger_name} body.")
        for geom in finger.findall("geom"):
            geom.attrib.pop("rgba", None)
            is_pad = geom.get("name") in {"left_tpu_pad", "right_tpu_pad"}
            geom.set("material", "pdz_contact_white" if is_pad else "pdz_finger_black")

    part_body = ET.SubElement(worldbody, "body", name="selected_part", mocap="true")
    ET.SubElement(
        part_body,
        "geom",
        name="selected_part_visual",
        type="mesh",
        mesh="selected_part_mesh",
        material="part_canonical",
        contype="0",
        conaffinity="0",
    )
    _author_canonical_tslot_surface(worldbody)
    _author_filament_fallback_lighting(root)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if model.nlight != 0:
        raise RuntimeError(
            "The canonical Filament scene must have no physical lights so its "
            "environment illumination remains active."
        )
    return model


def _apply_filament_materials(model: mujoco.MjModel) -> None:
    for name, material in GOAL_FILAMENT_MATERIALS.items():
        material_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_MATERIAL, name
        )
        if material_id < 0:
            raise RuntimeError(f"MuJoCo material '{name}' was not compiled.")
        model.mat_rgba[material_id, :3] = material.color
        model.mat_metallic[material_id] = material.metallic
        model.mat_roughness[material_id] = material.roughness
        model.mat_emission[material_id] = material.emission
    model.light_castshadow[:] = 0


def _policy_area_downsample(rgb: np.ndarray) -> np.ndarray:
    """Exactly average the 256x144 render into the policy's 128x72 RGB grid."""

    if rgb.shape != (HEIGHT, WIDTH, 3) or HEIGHT % 2 or WIDTH % 2:
        raise ValueError(f"Unexpected goal render shape for 2x area resize: {rgb.shape}.")
    averaged = rgb.reshape(HEIGHT // 2, 2, WIDTH // 2, 2, 3).mean(axis=(1, 3))
    return np.clip(np.rint(averaged), 0.0, 255.0).astype(np.uint8)


def _tcp_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tcp_position_link7: np.ndarray,
    tcp_rotation_link7: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    link7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    link7_rotation = data.xmat[link7_id].reshape(3, 3)
    return (
        data.xpos[link7_id] + link7_rotation @ tcp_position_link7,
        link7_rotation @ tcp_rotation_link7,
    )


def main() -> None:  # noqa: C901
    args = parse_args()
    if args.renderer_backend == MUJOCO_GOAL_RENDERER_BACKEND and os.environ.get(
        "MUJOCO_FILAMENT_ACTIVE"
    ) != "1":
        raise RuntimeError(
            "Filament catalog capture must be launched through "
            "scripts/run_mujoco_filament.sh so LD_PRELOAD is active before Python starts."
        )
    paths_asset = args.paths_asset.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    robot_urdf = args.robot_urdf.expanduser().resolve()
    for required in (paths_asset, manifest_path, robot_urdf):
        if not required.is_file():
            raise FileNotFoundError(required)
    with np.load(paths_asset, allow_pickle=False) as source:
        payload = {name: source[name].copy() for name in source.files}
    source_count = len(payload["target_ids"])
    if args.target_indices is not None:
        indices = np.asarray(args.target_indices, dtype=np.int64)
        if (
            indices.size < 1
            or len(np.unique(indices)) != len(indices)
            or int(indices.min()) < 0
            or int(indices.max()) >= source_count
        ):
            raise ValueError(f"--target-indices must be unique values in [0, {source_count - 1}].")
        payload = _select_targets(payload, indices)
    target_count = len(payload["target_ids"])
    goal_palette_indices: tuple[int, ...] = ()
    if args.goal_palette_indices is not None:
        goal_palette_indices = tuple(int(value) for value in args.goal_palette_indices)
        if len(goal_palette_indices) < 3 or len(set(goal_palette_indices)) != len(goal_palette_indices):
            raise ValueError("--goal-palette-indices requires at least three unique indices.")
        if min(goal_palette_indices) < 0 or max(goal_palette_indices) >= len(VISUAL_SERVO_PART_PALETTE):
            raise ValueError("A --goal-palette-indices value lies outside the part palette.")
    if str(np.asarray(payload.get("robot_profile", "")).item()) != VISUAL_SERVO_GRIPPER_PROFILE:
        raise ValueError("Path asset was not rebuilt for the active PDZ robot profile.")
    if str(np.asarray(payload.get("approach_gripper_profile", "")).item()) != PDZ_GRIPPER_APPROACH_PROFILE:
        raise ValueError("Path asset does not use the PDZ jaw-width-plus-10-mm profile.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    part_records = list(manifest["parts"])
    urdf_root = ET.parse(robot_urdf).getroot()
    tcp_position_link7, tcp_rotation_link7, _tcp_link = _robot_tcp_transform_link7(
        urdf_root
    )
    rgb = np.empty((target_count, HEIGHT, WIDTH, 3), dtype=np.uint8)
    depth = np.empty((target_count, HEIGHT, WIDTH), dtype=np.float32)
    position_errors = np.empty(target_count, dtype=np.float32)
    rotation_errors = np.empty(target_count, dtype=np.float32)
    goal_rgb_policy_variants = (
        np.empty((target_count, len(goal_palette_indices), HEIGHT // 2, WIDTH // 2, 3), dtype=np.uint8)
        if goal_palette_indices
        else None
    )

    with tempfile.TemporaryDirectory(prefix="mujoco_pdz_goal_catalog_") as temp_name:
        temporary = Path(temp_name)
        robot_mjcf = temporary / "pdz_robot_canonical.xml"
        _export_robot_mjcf(robot_urdf, robot_mjcf)
        for part_index, part in enumerate(part_records):
            selected = np.flatnonzero(payload["part_indices"] == part_index)
            if selected.size == 0:
                continue
            part_mesh = temporary / f"part_{part['part_id']}_bundle_local.stl"
            _build_bundle_mesh(part, part_mesh)
            model = _scene_model(robot_mjcf, robot_urdf, part_mesh)
            _apply_filament_materials(model)
            part_material_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_MATERIAL, "part_canonical"
            )
            if part_material_id < 0:
                raise RuntimeError("MuJoCo material 'part_canonical' was not compiled.")
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
            try:
                for completed, target_index in enumerate(selected.tolist(), start=1):
                    data.qpos[:7] = (
                        payload["reset_joint_trajectories"][target_index, -1]
                        * MOVEIT_TO_ISAAC_SIGNS
                    )
                    finger_travel = np.clip(
                        0.5
                        * (
                            float(payload["approach_gripper_widths_m"][target_index])
                            - PDZ_GRIPPER_CLOSED_WIDTH_M
                        ),
                        0.0,
                        PDZ_GRIPPER_TRAVEL_M,
                    )
                    data.qpos[7:9] = finger_travel
                    data.mocap_pos[0] = payload["object_positions_w"][target_index]
                    object_xyzw = payload["object_orientations_xyzw_w"][target_index]
                    data.mocap_quat[0] = object_xyzw[[3, 0, 1, 2]]
                    mujoco.mj_forward(model, data)
                    actual_position, actual_rotation = _tcp_pose(
                        model,
                        data,
                        tcp_position_link7,
                        tcp_rotation_link7,
                    )
                    desired_position = payload["goal_tcp_positions_w"][target_index]
                    desired_rotation = Rotation.from_quat(
                        payload["goal_tcp_orientations_xyzw_w"][target_index]
                    ).as_matrix()
                    position_errors[target_index] = np.linalg.norm(
                        desired_position - actual_position
                    )
                    rotation_errors[target_index] = np.linalg.norm(
                        Rotation.from_matrix(
                            desired_rotation @ actual_rotation.T
                        ).as_rotvec()
                    )
                    renderer.update_scene(data, camera="d405")
                    rgb[target_index] = renderer.render()
                    if goal_rgb_policy_variants is not None:
                        canonical_color = model.mat_rgba[part_material_id, :3].copy()
                        for variant_slot, palette_index in enumerate(goal_palette_indices):
                            model.mat_rgba[part_material_id, :3] = VISUAL_SERVO_PART_PALETTE[
                                palette_index
                            ].color
                            renderer.update_scene(data, camera="d405")
                            goal_rgb_policy_variants[target_index, variant_slot] = _policy_area_downsample(
                                renderer.render()
                            )
                        model.mat_rgba[part_material_id, :3] = canonical_color
                    renderer.enable_depth_rendering()
                    renderer.update_scene(data, camera="d405")
                    depth[target_index] = renderer.render()
                    renderer.disable_depth_rendering()
                    if completed % 50 == 0 or completed == len(selected):
                        print(
                            f"[MUJOCO CAPTURE] part={part['part_id']} "
                            f"{completed}/{len(selected)} total_index={target_index}",
                            flush=True,
                        )
            finally:
                renderer.close()

    depth = np.nan_to_num(depth, nan=0.50, posinf=0.50, neginf=0.04)
    depth_std = depth.reshape(target_count, -1).std(axis=1)
    rgb_std = rgb.reshape(target_count, -1).std(axis=1)
    position_failure = position_errors > float(args.maximum_position_error_m)
    rotation_error_deg = np.degrees(rotation_errors)
    rotation_failure = rotation_error_deg > float(args.maximum_rotation_error_deg)
    quality_failure = depth_std < float(args.minimum_depth_std_m)
    failure = position_failure | rotation_failure | quality_failure
    passed = ~failure

    payload.update(
        {
            "goal_rgb": rgb,
            "goal_depth": depth.astype(np.float32),
            "goal_tcp_capture_position_error_m": position_errors,
            "goal_tcp_capture_rotation_error_rad": rotation_errors,
            "goal_rgb_std": rgb_std.astype(np.float32),
            "goal_depth_std_m": depth_std.astype(np.float32),
            "capture_validation_passed": passed.astype(np.bool_),
            # Retained for old consumers while schema 4 records the actual backend.
            "isaac_goal_rgbd_captured": passed.astype(np.bool_),
            "goal_rgbd_captured": passed.astype(np.bool_),
            "goal_renderer_backend": np.asarray(args.renderer_backend),
            "goal_renderer_profile": np.asarray(MUJOCO_GOAL_RENDERER_PROFILE),
            "visual_material_profile": np.asarray(VISUAL_SERVO_MATERIAL_PROFILE),
            "visual_scene_profile": np.asarray(VISUAL_SERVO_SCENE_PROFILE),
            "visual_tslot_profile": np.asarray(VISUAL_SERVO_TSLOT_PROFILE),
            "goal_camera_profile": np.asarray(D405_VISUAL_SERVO_CAMERA_PROFILE),
            "goal_observation_profile": np.asarray(D405_VISUAL_SERVO_OBSERVATION_PROFILE),
        }
    )
    if goal_rgb_policy_variants is not None:
        payload["goal_rgb_policy_variants"] = goal_rgb_policy_variants
        payload["goal_variant_palette_indices"] = np.asarray(goal_palette_indices, dtype=np.int16)
    failures = []
    for index in np.flatnonzero(failure).tolist():
        reasons = []
        if position_failure[index]:
            reasons.append("tcp_position_error")
        if rotation_failure[index]:
            reasons.append("tcp_rotation_error")
        if quality_failure[index]:
            reasons.append("goal_depth_std_below_threshold")
        failures.append(
            {
                "target_index": index,
                "target_id": str(payload["target_ids"][index]),
                "reasons": reasons,
                "position_error_mm": float(position_errors[index] * 1000.0),
                "rotation_error_deg": float(rotation_error_deg[index]),
                "goal_depth_std_m": float(depth_std[index]),
                "goal_rgb_std": float(rgb_std[index]),
            }
        )
    output = args.output.expanduser().resolve()
    report_path = output.with_name(f"{output.stem}_validation_report.json")
    _atomic_write_json(
        report_path,
        {
            "schema_version": 2,
            "renderer_profile": MUJOCO_GOAL_RENDERER_PROFILE,
            "target_count": target_count,
            "passed_count": int(passed.sum()),
            "failure_count": int(failure.sum()),
            "worst_position_error_mm": float(position_errors.max() * 1000.0),
            "worst_rotation_error_deg": float(rotation_error_deg.max()),
            "failures": failures,
        },
    )
    if args.contact_sheet is not None:
        _write_contact_sheet(
            path=args.contact_sheet.expanduser().resolve(),
            rgb=rgb,
            target_ids=payload["target_ids"],
        )
    if failure.any():
        diagnostic = output.with_name(f"{output.stem}_failed_validation.npz")
        _atomic_savez(diagnostic, payload)
        raise RuntimeError(
            f"Rejected {int(failure.sum())}/{target_count} MuJoCo goal renders; "
            f"diagnostics are in {diagnostic}."
        )
    _atomic_savez(output, payload)
    load_multigrasp_catalog(output, expected_arm_joint_count=7, require_complete=True)
    print(
        f"[DONE] Wrote {target_count} validated MuJoCo Filament PDZ RGB-D goals to {output}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
