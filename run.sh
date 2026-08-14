#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_ROOT="${ISAACLAB_ROOT:-/media/pdz/Elements1/IsaacLab-2.3.2}"
PYTHON="${ISAACLAB_ROOT}/isaaclab.sh"
TASK="Grasp-Visual-Servo-RGBD-Direct-v0"
PLAY_TASK="Grasp-Visual-Servo-RGBD-Direct-Play-v0"
SAC_TASK="Grasp-Visual-Servo-RGBD-MultiPart-Direct-v0"
SAC_PLAY_TASK="Grasp-Visual-Servo-RGBD-MultiPart-Direct-Play-v0"

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
  sac-smoke)
    TERM=xterm "${PYTHON}" -p scripts/asymmetric_sac/train.py \
      --task "${SAC_TASK}" --smoke --headless --enable_cameras
    ;;
  sac-train)
    env_count="${2:-64}"
    transitions="${3:-2000000}"
    TERM=xterm "${PYTHON}" -p scripts/asymmetric_sac/train.py \
      --task "${SAC_TASK}" --num_envs "${env_count}" \
      --total_transitions "${transitions}" --headless --enable_cameras
    ;;
  sac-play)
    checkpoint="${2:?usage: ./run.sh sac-play CHECKPOINT [episodes]}"
    episodes="${3:-20}"
    TERM=xterm "${PYTHON}" -p scripts/asymmetric_sac/play.py \
      --task "${SAC_PLAY_TASK}" --checkpoint "${checkpoint}" \
      --episodes "${episodes}" --headless --enable_cameras
    ;;
  play)
    checkpoint="${2:?usage: ./run.sh play CHECKPOINT}"
    TERM=xterm "${PYTHON}" -p scripts/rl_games/play.py \
      --task "${PLAY_TASK}" --num_envs 1 --checkpoint "${checkpoint}" \
      --video --video_length 450 --headless --enable_cameras
    ;;
  *)
    echo "usage: ./run.sh {setup|list|smoke|train [envs] [iterations]|play CHECKPOINT|sac-smoke|sac-train [envs] [transitions]|sac-play CHECKPOINT [episodes]}"
    ;;
esac
