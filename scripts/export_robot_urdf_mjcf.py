#!/usr/bin/env python3
"""Import a robot URDF with MuJoCo and write its canonical MJCF."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot_urdf", type=Path)
    parser.add_argument("output_mjcf", type=Path)
    args = parser.parse_args()

    robot_urdf = args.robot_urdf.expanduser().resolve()
    output_mjcf = args.output_mjcf.expanduser().resolve()
    if not robot_urdf.is_file():
        raise FileNotFoundError(robot_urdf)
    output_mjcf.parent.mkdir(parents=True, exist_ok=True)
    output_mjcf.write_text(
        mujoco.MjSpec.from_file(str(robot_urdf)).to_xml(), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
