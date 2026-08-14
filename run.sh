#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_ROOT="${ISAACLAB_ROOT:-/media/pdz/Elements1/IsaacLab-2.3.2}"
PYTHON="${ISAACLAB_ROOT}/isaaclab.sh"
TASK="Grasp-Visual-Servo-RGBD-Direct-v0"
PLAY_TASK="Grasp-Visual-Servo-RGBD-Direct-Play-v0"

case "${1:-help}" in
  setup)
    TERM=xterm "${PYTHON}" -p -m pip install -e source/isaac_rl
    ;;
  list)
    TERM=xterm "${PYTHON}" -p scripts/list_envs.py --keyword Grasp
    ;;
  smoke)
    TERM=xterm "${PYTHON}" -p scripts/smoke_env.py --headless --enable_cameras
    ;;
  train)
    env_count="${2:-64}"
    iterations="${3:-2000}"
    TERM=xterm "${PYTHON}" -p scripts/rl_games/train.py \
      --task "${TASK}" --num_envs "${env_count}" --max_iterations "${iterations}" \
      --headless --enable_cameras
    ;;
  play)
    checkpoint="${2:?usage: ./run.sh play CHECKPOINT}"
    TERM=xterm "${PYTHON}" -p scripts/rl_games/play.py \
      --task "${PLAY_TASK}" --num_envs 1 --checkpoint "${checkpoint}" \
      --video --video_length 450 --headless --enable_cameras
    ;;
  *)
    echo "usage: ./run.sh {setup|list|smoke|train [envs] [iterations]|play CHECKPOINT}"
    ;;
esac
