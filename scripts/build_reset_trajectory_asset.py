"""Embed the nominal MoveIt pregrasp-to-grasp path in the compact RL goal asset."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from xml.etree import ElementTree

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
MOVEIT_TO_ISAAC_SIGNS = np.asarray(
    (1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0), dtype=np.float64
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--moveit-plan",
        type=Path,
        default=REPO_ROOT / "artifacts/sim_isaac_pick_attempt_moveit_plan.json",
    )
    parser.add_argument(
        "--asset",
        type=Path,
        default=REPO_ROOT / "isaac_rl/data/fixed_goal_reset.npz",
    )
    parser.add_argument(
        "--robot-urdf",
        type=Path,
        default=REPO_ROOT / "assets/urdf/kuka_iiwa7_y_gripper/urdf/kuka_iiwa7_y_gripper.urdf",
    )
    parser.add_argument("--waypoints", type=int, default=32)
    return parser.parse_args()


def _fixed_joint_translation(urdf_root: ElementTree.Element, joint_name: str) -> np.ndarray:
    joint = urdf_root.find(f".//joint[@name='{joint_name}']")
    if joint is None:
        raise ValueError(f"Robot URDF has no joint named {joint_name}.")
    origin = joint.find("origin")
    if origin is None:
        return np.zeros(3, dtype=np.float64)
    rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
    if not np.allclose(rpy, 0.0, atol=1.0e-12):
        raise ValueError(f"{joint_name} must have zero fixed rotation, got rpy={rpy}.")
    return np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")


def _fixed_link_transform(
    urdf_root: ElementTree.Element,
    *,
    ancestor_link: str,
    descendant_link: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a fixed descendant pose expressed in an ancestor link."""

    child_to_joint: dict[str, ElementTree.Element] = {}
    for joint in urdf_root.findall("joint"):
        child = joint.find("child")
        if child is not None:
            child_to_joint[str(child.get("link"))] = joint
    chain: list[ElementTree.Element] = []
    current = str(descendant_link)
    while current != str(ancestor_link):
        joint = child_to_joint.get(current)
        if joint is None:
            raise ValueError(
                f"No fixed URDF chain from '{ancestor_link}' to '{descendant_link}'."
            )
        if str(joint.get("type")) != "fixed":
            raise ValueError(
                f"Transform chain to '{descendant_link}' crosses non-fixed joint "
                f"'{joint.get('name')}'."
            )
        parent = joint.find("parent")
        if parent is None:
            raise ValueError(f"Joint '{joint.get('name')}' has no parent link.")
        chain.append(joint)
        current = str(parent.get("link"))

    transform = np.eye(4, dtype=np.float64)
    for joint in reversed(chain):
        origin = joint.find("origin")
        xyz = (
            np.zeros(3, dtype=np.float64)
            if origin is None
            else np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        )
        rpy = (
            np.zeros(3, dtype=np.float64)
            if origin is None
            else np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        )
        local = np.eye(4, dtype=np.float64)
        local[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        local[:3, 3] = xyz
        transform = transform @ local
    return transform[:3, 3].copy(), transform[:3, :3].copy()


def _robot_tcp_transform_link7(
    urdf_root: ElementTree.Element,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Resolve the active KUKA gripper planner-TCP transform."""

    links = {str(link.get("name")) for link in urdf_root.findall("link")}
    tcp_link = "pdz_gripper_tcp" if "pdz_gripper_tcp" in links else "gripper_tcp"
    position, rotation = _fixed_link_transform(
        urdf_root,
        ancestor_link="link7",
        descendant_link=tcp_link,
    )
    return position, rotation, tcp_link


def _sample_reference_joint_path(trajectory: np.ndarray, progress: float) -> np.ndarray:
    scaled = float(np.clip(progress, 0.0, 1.0)) * (trajectory.shape[0] - 1)
    lower = int(np.floor(scaled))
    upper = min(lower + 1, trajectory.shape[0] - 1)
    return (1.0 - (scaled - lower)) * trajectory[lower] + (scaled - lower) * trajectory[upper]


def _straight_cartesian_joint_path(
    *,
    raw_moveit_trajectory: np.ndarray,
    plan: dict[str, object],
    robot_urdf: Path,
    waypoint_count: int,
    maximum_position_error_m: float = 2.0e-5,
    maximum_rotation_error_rad: float = 2.0e-4,
) -> tuple[np.ndarray, float, float]:
    """Solve a fixed-orientation straight TCP path using the generated robot URDF."""

    if waypoint_count < 2:
        raise ValueError("--waypoints must be at least 2.")
    model = mujoco.MjModel.from_xml_path(str(robot_urdf))
    data = mujoco.MjData(model)
    link7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    if link7_id < 0:
        raise ValueError("Robot URDF did not produce a link7 MuJoCo body.")
    urdf_root = ElementTree.parse(robot_urdf).getroot()
    tcp_offset_link7, tcp_rotation_link7, tcp_link = _robot_tcp_transform_link7(
        urdf_root
    )

    # The stored MoveIt paths were planned with a TCP whose axes matched
    # ``link7``.  The PDZ TCP is rotated -90 degrees about link7 Z.  KUKA A7 is
    # the same local-Z rotation, so compensate the old path before using it as
    # the numerical IK seed.  This leaves the requested world TCP pose
    # unchanged and avoids starting the solver a quarter-turn from the valid
    # branch.  Keep this generic for the identity legacy TCP and any other
    # pure-Z tool rotation.
    moveit_payload = plan.get("moveit", {})
    planned_pose_link = (
        str(moveit_payload.get("pose_link", "gripper_tcp"))
        if isinstance(moveit_payload, dict)
        else "gripper_tcp"
    )
    seed_compensation = (
        np.zeros(3, dtype=np.float64)
        if planned_pose_link == tcp_link
        else Rotation.from_matrix(tcp_rotation_link7.T).as_rotvec()
    )
    if np.linalg.norm(seed_compensation[:2]) > 1.0e-8:
        raise ValueError(
            "The reset-path seed correction supports only a TCP rotation about "
            f"link7 Z, got rotvec={seed_compensation.tolist()}."
        )
    reference_trajectory = raw_moveit_trajectory.copy()
    reference_trajectory[:, 6] += float(seed_compensation[2])
    reference_trajectory[:, 6] = (
        reference_trajectory[:, 6] + np.pi
    ) % (2.0 * np.pi) - np.pi

    selected_grasp = plan["selected_world_grasp"]
    if not isinstance(selected_grasp, dict):
        raise ValueError("MoveIt plan is missing selected_world_grasp.")
    pregrasp_position = np.asarray(selected_grasp["pregrasp_position_w"], dtype=np.float64)
    grasp_position = np.asarray(selected_grasp["position_w"], dtype=np.float64)
    target_rotation = Rotation.from_quat(
        np.asarray(selected_grasp["orientation_xyzw"], dtype=np.float64)
    ).as_matrix()
    joint_lower = model.jnt_range[:7, 0]
    joint_upper = model.jnt_range[:7, 1]

    def pose_and_jacobian(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        link7_rotation = data.xmat[link7_id].reshape(3, 3).copy()
        position = data.xpos[link7_id].copy() + link7_rotation @ tcp_offset_link7
        rotation = link7_rotation @ tcp_rotation_link7
        jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(
            model,
            data,
            jacobian_position,
            jacobian_rotation,
            position,
            link7_id,
        )
        jacobian = np.vstack((jacobian_position[:, :7], jacobian_rotation[:, :7]))
        return position, rotation, jacobian

    progress_values = np.linspace(0.0, 1.0, waypoint_count)
    target_positions = (
        (1.0 - progress_values[:, None]) * pregrasp_position
        + progress_values[:, None] * grasp_position
    )
    solved: list[np.ndarray] = []
    maximum_position_error = 0.0
    maximum_rotation_error = 0.0
    for progress, target_position in zip(progress_values, target_positions, strict=True):
        reference = _sample_reference_joint_path(reference_trajectory, float(progress))
        q = reference.copy()
        for _ in range(200):
            position, rotation, jacobian = pose_and_jacobian(q)
            position_error = target_position - position
            rotation_error = Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec()
            if np.linalg.norm(position_error) < 1.0e-7 and np.linalg.norm(rotation_error) < 1.0e-6:
                break
            error = np.concatenate((position_error, rotation_error))
            damping = 2.0e-3
            inverse = jacobian.T @ np.linalg.inv(
                jacobian @ jacobian.T + damping**2 * np.eye(6)
            )
            delta = inverse @ error
            # Keep the redundant joint close to MoveIt's path without letting
            # the damped null-space projector hold the Cartesian task tens of
            # micrometres away from its target near singular configurations.
            delta += 0.002 * (np.eye(7) - inverse @ jacobian) @ (reference - q)
            maximum_step = float(np.max(np.abs(delta)))
            if maximum_step > 0.08:
                delta *= 0.08 / maximum_step
            q = np.clip(q + delta, joint_lower + 1.0e-4, joint_upper - 1.0e-4)
        position, rotation, _ = pose_and_jacobian(q)
        position_error = float(np.linalg.norm(target_position - position))
        rotation_error = float(
            np.linalg.norm(Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec())
        )
        maximum_position_error = max(maximum_position_error, position_error)
        maximum_rotation_error = max(maximum_rotation_error, rotation_error)
        if (
            position_error > maximum_position_error_m
            or rotation_error > maximum_rotation_error_rad
        ):
            raise RuntimeError(
                "Straight-path IK did not converge: "
                f"progress={progress:.3f}, position_error={position_error:.6g} m, "
                f"rotation_error={rotation_error:.6g} rad."
            )
        solved.append(q)
    return np.asarray(solved), maximum_position_error, maximum_rotation_error


def main() -> None:
    args = parse_args()
    plan = json.loads(args.moveit_plan.read_text(encoding="utf-8"))
    joint_names = tuple(str(name) for name in plan["joint_names"])
    expected_names = tuple(f"lbr_A{index}" for index in range(1, 8))
    if joint_names != expected_names:
        raise ValueError(f"Expected KUKA MoveIt joints {expected_names}, got {joint_names}.")
    raw_trajectory = np.asarray(plan["trajectories"]["grasp"], dtype=np.float64)
    if raw_trajectory.ndim != 2 or raw_trajectory.shape[1] != 7 or raw_trajectory.shape[0] < 2:
        raise ValueError(f"Expected a grasp trajectory with shape (N>=2, 7), got {raw_trajectory.shape}.")
    straight_moveit_trajectory, maximum_position_error, maximum_rotation_error = (
        _straight_cartesian_joint_path(
            raw_moveit_trajectory=raw_trajectory,
            plan=plan,
            robot_urdf=args.robot_urdf,
            waypoint_count=args.waypoints,
        )
    )
    # The generated Isaac USD inverts only the physical A4 coordinate.
    reset_joint_trajectory = (
        straight_moveit_trajectory * MOVEIT_TO_ISAAC_SIGNS[None, :]
    ).astype(np.float32)

    with np.load(args.asset, allow_pickle=False) as source:
        payload = {name: source[name].copy() for name in source.files}
    payload["reset_joint_trajectory"] = reset_joint_trajectory
    payload["reset_path_progress"] = np.linspace(
        0.0, 1.0, reset_joint_trajectory.shape[0], dtype=np.float32
    )
    payload["reset_source_grasp_id"] = np.asarray(str(plan["selected_grasp_id"]))
    payload["reset_path_type"] = np.asarray("straight_cartesian_fixed_orientation")
    payload["reset_path_max_position_error_m"] = np.asarray(
        maximum_position_error, dtype=np.float32
    )
    payload["reset_path_max_rotation_error_rad"] = np.asarray(
        maximum_rotation_error, dtype=np.float32
    )

    args.asset.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.asset.stem}-", suffix=".npz", dir=args.asset.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(temporary_path, **payload)
        os.replace(temporary_path, args.asset)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(
        f"Wrote {reset_joint_trajectory.shape[0]} straight Cartesian reset waypoints "
        f"from grasp {plan['selected_grasp_id']} to {args.asset}; "
        f"max_position_error={maximum_position_error * 1.0e6:.3f} um, "
        f"max_rotation_error={np.degrees(maximum_rotation_error):.6f} deg."
    )


if __name__ == "__main__":
    main()
