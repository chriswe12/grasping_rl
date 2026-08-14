# Grasp Visual Servo RL

Official Isaac Lab 2.3.2 external-project workspace for the KUKA iiwa7/Y-gripper
RGB-D grasp-alignment task. The sibling Isaac Lab checkout is expected at
`/media/pdz/Elements1/IsaacLab-2.3.2` and is pinned to tag `v2.3.2` for Isaac Sim 5.1.

The actor receives the same deployment images used by behavior cloning:

- live RGB-D, 72 x 128;
- target-grasp RGB-D, 72 x 128, selected independently per environment.

The PPO policy is hybrid: six bounded camera-frame TCP twist values plus one
Bernoulli completion decision. The twist is not added to a nominal controller.
During training only, the centralized critic receives joint position and
velocity, TCP pose error, and the previous six motion actions.

The actor uses a shared ImageNet-pretrained ResNet-18 RGB encoder through layer
3, a trainable depth CNN, and spatial live/goal feature fusion. The RGB stem and
first two residual stages remain frozen; RGB layer 3, depth, fusion, policy,
and auxiliary-head parameters train together. A learned geometric trunk is
shared by the motion, completion, and six-axis pose-error heads, so pose and
completion supervision shape features available to motion without feeding
ground-truth pose into the actor. The motion head additionally receives the
previous six executed motion actions, which are available at deployment. The
fusion retains live and goal features together with signed difference,
absolute difference, and elementwise agreement maps.

During simulator training, the flattened visual observation is followed by the
previous six motion actions, six normalized camera-frame pose-error labels, a
completion label, and a supervision mask. The network slices the final eight
privileged values away before its visual/action path. A pose head uses weighted
smooth-L1 loss, while a completion head uses masked, positive-weighted binary
cross entropy. Strict completion
positives are at most 4 mm and 3 degrees and collision-free; clear negatives are
at least 6 mm or 4 degrees, or in unsafe contact. Samples in between are ignored
by the completion loss. These privileged labels are never needed for deployed
inference. Tests verify that changing only the labels leaves both motion and
completion predictions exactly unchanged.

All pretrained ResNet BatchNorm running statistics stay frozen during PPO,
including trainable layer 3. The actor applies `tanh` to its Gaussian action
mean before sampling. Completion confidence begins attenuating the motion mean
and standard deviation at `p(done)=0.70`, reaching a 0.25 floor at certainty.
The environment also limits normalized action changes to 0.25 per step and
commands at most 0.04 m/s and 0.24 rad/s. PPO uses a linearly decayed `5e-5`
learning rate, two mini-epochs, `0.1` policy clipping, and a `0.5`
gradient-norm limit.

The reward pays only for step-to-step position and rotation potential change.
It adds `+50` only when the policy explicitly declares completion inside the
strict ground-truth region, applies `-50` for a premature declaration or unsafe
hand contact, `-15` for timeout, and `-25` for divergence, plus the existing
`-0.02` step and `-0.002 * ||motion||^2` costs. Synthetic already-ready resets
receive only 20% of the terminal success reward so they teach the classifier
without dominating PPO return. A squared contact-risk cost ramps from 0.05 N
to the 1 N unsafe-contact boundary. Reaching the geometric goal never ends an
episode by itself. Stochastic PPO must emit four consecutive sampled stop
actions; deterministic playback requires `p(done) >= 0.95`. Both paths also
require TCP speed below 0.005 m/s and 0.03 rad/s.

The active target asset is `data/multigrasp_50_catalog.npz`. It contains 50
goal-conditioned tasks: five part orientations with ten diverse grasps each.
Every target has its own object pose, desired TCP pose, 144 x 256 Isaac goal
RGB-D observation, and 32 x 7 straight Cartesian reset path. The catalog is only
accepted after ground/gripper filtering, MoveIt pregrasp-and-grasp planning,
sub-millimetre Isaac goal-pose validation, and a goal-image information check.
Ranked alternates replace targets that fail MoveIt or produce an almost
constant-depth camera view. The goal RGB-D is downsampled exactly once to the
same 72 x 128 input used by the live camera.
Goal capture and the live task both derive field of view from the calibrated
848 x 480 D405 intrinsics while rendering a 256 x 144 buffer. Treating that
render buffer as the intrinsic reference would incorrectly narrow the goal
view to about 30 degrees. Catalogs are labeled with camera,
observation-preprocessing, material, and visual-scene profiles and are rejected
if they do not match the task. The shared canonical scene uses a low-level dome
fill plus one angled distant key, explicit DLAA, four direct-light samples per
pixel, DL denoising, and shadows. The distant key gives every cloned
environment the same lighting without creating one light per environment. The shared material
profile renders the part as matte brown PLA, the two fingers as readable matte
black, and the work surface dark charcoal, so the exact goal visibly includes
both fingers around the part. Both live and goal RGB-D are area-filtered once
from 256 x 144 to the 128 x 72 policy input instead of nearest-neighbor sampled.

During training only, domain randomization is applied to the live RGB-D tensor;
the selected catalog goal remains the deterministic canonical reference. Each
environment samples episode-stable exposure, contrast, gamma, white balance,
vignetting, blur, depth scale/bias, and occasional small RGB/depth patches.
Occluded RGB patches use the live-frame mean and missing depth uses maximum
range rather than an artificial black rectangle. RGB noise, metric depth
noise, millimetre quantization, and edge-weighted missing depth vary per frame.
The physical live scene also changes key-light direction/intensity/temperature,
shadow position, part hue/brightness/roughness, and ground hue/brightness/
roughness. All appearance strength ramps with the curriculum. Playback,
evaluation, and composite debug recording disable these augmentations. This
matches deployment: a real D405 live observation is compared with a fixed
synthetic goal catalog.
The active extrinsic is the CAD-derived camera pose expressed directly in the
flange/link7 frame: `t = (55.667, 9.000, 70.776) mm` and
`R = [[0, -sqrt(3)/2, -1/2], [1, 0, 0], [0, -1/2, sqrt(3)/2]]`.
No legacy 35 mm `lbr_link_ee` offset is added.

The companion `data/multigrasp_50_rotation_resets.npz` contains 16 validated,
position-preserving rotational reset paths per grasp. Every path covers the
same 32 approach points as its nominal target while rotating the TCP about a
feasible world-space axis. Offline IK validation limits residuals to 0.05 mm
and 0.0005 rad before the asset can be loaded by training.
Candidate axes are deterministically farthest-point ordered over a dense
Fibonacci sphere, so a short 16-axis prefix covers both hemispheres rather
than taking the +Z-heavy first entries of the raw Fibonacci ordering. At each
reset, one of those validated directions is still drawn uniformly at random.

Multipart resets use a fixed mixture: 55% continuous path cases, 15% exact
authored path states without pose noise, 15% ready-region cases, and 15%
completion-boundary cases. Ready-region position error is continuous from zero
to 3.5 mm; 25% of those cases are the exact nominal successful pose. Boundary
cases use either one of the last three non-final nominal waypoints or the fully
collision-validated final five-degree rotation. Other path states use
continuous XY error down to zero and fully authored rotation variants. The code
does not interpolate unvalidated joint-space rotations.

Target IDs remain part-balanced. A step-based curriculum holds ordinary path
resets in the close 70--94% region for 16,000 simulation steps, then expands
path distance, Cartesian/rotation perturbation, appearance variation, and hard
target replay linearly until step 192,000. Episode time is reset-dependent:
ordinary path budgets interpolate from 4 s close to 12 s far, ready cases get
1.5 s, and boundary cases get 2.5 s. Terminal failures update a per-target EMA;
up to 25% of later resets are resampled by failure score within the same part,
so hard-target replay cannot destroy part balance.

Every position displacement shifts the active object and TCP goal together and
is capped by the exact nominal/rotation waypoint clearance so the authored 1 mm
minimum plus a 0.1 mm guard remains. TensorBoard reports the reset mixture,
timeouts, curriculum phase, hard-replay rate/scores, initial/final errors,
completion behavior, contact risk, and far/mid/close and per-part results.

The actor observation is now 73,742 values and the shared-head architecture
changed. Start a fresh run; checkpoints produced before this revision,
including earlier seven-action completion policies, are not load-compatible.

Any catalog captured with an older visual, material, or observation profile is
intentionally rejected. Regenerate it end to end before starting a fresh
training run. The normal Python orchestrator starts mock MoveIt,
validates/replaces targets, builds reset paths, renders the goals in batches,
and stops mock MoveIt again:

```bash
cd /media/pdz/Elements1/Grasp_Planning_grasping_rl
python3 isaac_rl/scripts/prepare_multigrasp_catalog.py
```

Start a fresh training run with the standard Python entry point:

```bash
cd /media/pdz/Elements1/Grasp_Planning_grasping_rl
/media/pdz/Elements1/IsaacLab-2.3.2/isaaclab.sh -p \
  isaac_rl/scripts/rl_games/train.py \
  --task Grasp-Visual-Servo-RGBD-Direct-v0 \
  --num_envs 64 --max_iterations 5000 --headless --enable_cameras
```

Catalog debug images and the numerical report can be regenerated without
launching Isaac:

```bash
python3 isaac_rl/scripts/render_multigrasp_catalog_debug.py
```

For explicit playback conditions, call the normal Python entry point through
Isaac Lab. Progress and noise are independent overrides; without a noise
override, the configured path-conditioned playback noise remains active.

```bash
cd /media/pdz/Elements1/Grasp_Planning_grasping_rl/isaac_rl

# Far, high-noise stress test.
/media/pdz/Elements1/IsaacLab-2.3.2/isaaclab.sh -p scripts/rl_games/play.py \
  --task Grasp-Visual-Servo-RGBD-Direct-Play-v0 --num_envs 1 \
  --checkpoint logs/rl_games/grasp_visual_servo_rgbd/<run>/nn/<checkpoint>.pth \
  --target_id orientation_002__g1875 \
  --reset_progress 0.0 --reset_noise_rad 0.08 --reset_rotation_deg 15 \
  --video --video_length 450

# Mid-distance sample.
/media/pdz/Elements1/IsaacLab-2.3.2/isaaclab.sh -p scripts/rl_games/play.py \
  --task Grasp-Visual-Servo-RGBD-Direct-Play-v0 --num_envs 1 \
  --checkpoint logs/rl_games/grasp_visual_servo_rgbd/<run>/nn/<checkpoint>.pth \
  --target_index 27 \
  --reset_progress 0.5 --reset_noise_rad 0.03 --video --video_length 450

# Close sample; 0.85 is preferable to spawning exactly inside success tolerance.
/media/pdz/Elements1/IsaacLab-2.3.2/isaaclab.sh -p scripts/rl_games/play.py \
  --task Grasp-Visual-Servo-RGBD-Direct-Play-v0 --num_envs 1 \
  --checkpoint logs/rl_games/grasp_visual_servo_rgbd/<run>/nn/<checkpoint>.pth \
  --reset_progress 0.85 --reset_noise_rad 0.005 --video --video_length 450
```

Training logs and checkpoints are written below `logs/rl_games/`. Playback with
the Python entry point records an actual Isaac video below the checkpoint run's
`videos/` directory. Explicit target/progress/noise runs use condition-specific video
subdirectories so far, mid, and close recordings do not overwrite one another.
The play task uses a 15-second episode, matching the 450-step examples at 30 Hz;
the training task retains its 4-second horizon.
Each playback prints the off-path spawn offset plus realized initial/final
position and rotation errors and
writes `play_metrics.json` beside the MP4. The JSON includes per-step errors,
termination reason, success, and the minimum position and rotation errors.

For indefinite live GUI playback, omit `--headless` and `--video`, enable the
camera explicitly, and use `--real-time`. `--random_targets` independently
draws a new catalog target after every reset (repeats are allowed), while
`--reset_rotation_range_deg MIN MAX` redraws the authored rotation magnitude.

The evaluator also supports a stratified initial-pose grid. `far`, `mid`, and
`close` use path progress 0.0, 0.5, and 0.85 respectively; their default joint
noise is 0.040, 0.016, and 0.005 rad. For example:

```bash
/media/pdz/Elements1/IsaacLab-2.3.2/isaaclab.sh -p \
  isaac_rl/scripts/rl_games/evaluate_multigrasp.py \
  --task Grasp-Visual-Servo-RGBD-Direct-Play-v0 \
  --checkpoint <checkpoint.pth> --catalog_split all \
  --runs_per_target 3 --episode_seconds 15 \
  --conditions far mid close --rotation_deg 15 --headless
```

`record_debug_videos.py` records a composite 1600x900 MP4 for each requested
condition. Each frame contains an external side view, the exact downsampled
live/goal RGB inputs seen by the policy, current/initial/final pose errors,
`p(done)`, and an error-history plot. `--target_indices` is optional; without
it, targets are drawn independently at random.
Add the `exact` condition to record a zero-action final-path reference. It uses
zero joint noise and zero authored rotation and reports the live-versus-goal
RGB MAE, making camera/material/catalog mismatches directly visible.
Video playback stops when the policy declares success, declares prematurely,
hits an unsafe collision, times out, or diverges, before the automatic reset can
append a misleading frame from a second episode.

Evaluate every catalog target with three independent 15-second attempts at the
maximum training reset noise and three attempts at the stronger playback noise:

```bash
cd /media/pdz/Elements1/Grasp_Planning_grasping_rl
/media/pdz/Elements1/IsaacLab-2.3.2/isaaclab.sh -p \
  isaac_rl/scripts/rl_games/evaluate_multigrasp.py \
  --checkpoint logs/rl_games/grasp_visual_servo_rgbd/<run>/nn/<checkpoint>.pth \
  --runs_per_target 3 --episode_seconds 15 --rotation_deg 15 --headless
```

The evaluator runs one Isaac environment per grasp and writes exact episode,
per-target, per-orientation, and aggregate results under the checkpoint run's
`evaluations/` directory. The current 50-target training setup has no held-out
target split, so its `stress` condition measures generalization to stronger
reset perturbations, not to unseen grasps.

## Five-part plumbers-block curriculum

After multi-part training, run the complete held-out benchmark and create one
far-start debug video per part plus an exact-goal reference with one command:

```bash
python3 isaac_rl/scripts/benchmark_multipart_policy.py
```

The command automatically selects the newest stable multi-part checkpoint,
evaluates every validation and test target from far/mid/close initial states for
15 seconds, and writes a combined Markdown/JSON report and composite MP4s below
that checkpoint run's `evaluations/multipart_full_<timestamp>/` directory. Use
`--checkpoint PATH` to select a specific checkpoint, or `--runs_per_target 3`
for a more statistically reliable but slower benchmark.
Use `--video_split validation --videos_per_part 2 --skip_benchmark` to add two
different validation examples per part to an existing benchmark output.

The separate multi-part task expands the same controller to parts `0` through
`4` of the `plumbers_block` assembly. It keeps as many diverse grasps as are
both geometry-valid and MoveIt-reachable, up to 64 per stable part/orientation
group. Each episode activates one selected part at its catalog pose and parks
the other four rigid objects outside the workspace. Training samples parts
uniformly first and targets uniformly within a part, so parts with many valid
grasps do not drown out harder parts with a smaller catalog.

The catalog is split 80/10/10 into train, validation, and test. Splitting is by
the exact `(part_id, grasp_id)` group, not by rendered target: if one local
grasp is valid in multiple stable orientations, all of those occurrences stay
in the same split. MoveIt reachability filtering recomputes the split over only
surviving targets. This prevents a geometrically identical grasp from leaking
between training and held-out evaluation.

The current checked-in, Isaac-validated catalog contains 1,256 usable targets:
1,012 train, 125 validation, and 119 test. Every target has a 32-waypoint nominal path and
eight distinct position-preserving rotation paths reaching 15 degrees at
pregrasp and tapering toward the grasp. The checked-in rotation file uses the
required schema-2 unbiased axis ordering and exactly matches those 1,256
targets.

Catalog preparation is resumable and deliberately separated into CPU and
Isaac stages. The CPU stage uses the normal pipeline for each part, filters
with MoveIt, creates the straight Cartesian reset paths, and builds the
15-degree rotation-reset pool. Rotation results are cached per target.

```bash
cd /media/pdz/Elements1/Grasp_Planning_grasping_rl
python3 isaac_rl/scripts/prepare_plumbers_block_catalog.py --stage cpu
```

When the GPU is free, one Isaac launch builds the five bundle-local collision
USDs and a second batched launch captures and validates every goal RGB-D image:

```bash
python3 isaac_rl/scripts/prepare_plumbers_block_catalog.py --stage isaac
```

If the full MoveIt path asset includes targets rejected by an earlier Isaac
pose-validation pass, this stage automatically uses the active target subset
from `rotation_resets.npz`. The freshly rendered goal catalog therefore keeps
the exact target order required by multipart training.

If only some rendered targets fail Isaac TCP-pose validation, the stage keeps
the full diagnostic NPZ, promotes only passing RGB-D rows, and filters the
rotation-reset asset to the identical target order. A capture produced before
this automatic recovery was added can be finalized without rerendering:

```bash
python3 isaac_rl/scripts/prepare_plumbers_block_catalog.py --stage finalize
```

Start a fresh multi-part policy. Do not resume any checkpoint from before the
shared geometry/context revision because its observation and parameter shapes
are different:

```bash
/media/pdz/Elements1/IsaacLab-2.3.2/isaaclab.sh -p \
  isaac_rl/scripts/rl_games/train.py \
  --task Grasp-Visual-Servo-RGBD-MultiPart-Direct-v0 \
  --num_envs 64 --max_iterations 10000 --headless --enable_cameras
```

The multi-part task loads only the train split. Playback defaults to test; the
split may be changed explicitly to inspect validation or training targets:

```bash
/media/pdz/Elements1/IsaacLab-2.3.2/isaaclab.sh -p \
  isaac_rl/scripts/rl_games/play.py \
  --task Grasp-Visual-Servo-RGBD-MultiPart-Direct-Play-v0 \
  --checkpoint logs/rl_games/grasp_visual_servo_rgbd_multipart/<run>/nn/<checkpoint>.pth \
  --catalog_split test --num_envs 1 \
  --reset_progress 0.0 --reset_noise_rad 0.0 --reset_rotation_deg 15 \
  --video --video_length 450

/media/pdz/Elements1/IsaacLab-2.3.2/isaaclab.sh -p \
  isaac_rl/scripts/rl_games/evaluate_multigrasp.py \
  --task Grasp-Visual-Servo-RGBD-MultiPart-Direct-Play-v0 \
  --checkpoint logs/rl_games/grasp_visual_servo_rgbd_multipart/<run>/nn/<checkpoint>.pth \
  --catalog_split validation --runs_per_target 3 \
  --episode_seconds 15 --rotation_deg 15 --headless
```

Evaluation reports aggregate, per-part, per-orientation, and per-target errors
and success. Use validation while selecting a checkpoint and run test only for
the final held-out result.

## Asymmetric SAC experiment

The SAC implementation is separate from the RL-Games PPO path. Its actor sees
only the deployable `72x128x8` live/goal RGB-D tensor and the previous 6D
motion command. Twin Q critics see the privileged 26D state (`q`, `qd`, 6D
pose error, previous motion) plus the current 6D motion action. Pose error and
completion are supervised actor heads; completion probability is appended as
the environment's seventh, deterministic stop-gate input and is not treated as
a continuous SAC action.

Replay stays in CPU memory. Live RGB is `uint8`, normalized live depth is
offset-quantized 16-bit values, low-dimensional tensors are `float16`, and canonical goal images are
recovered from one shared catalog using stored target indices. The default
65,536-transition replay allocates about 5.64 GiB instead of hundreds of GiB for
full float32 live/goal observations.

The default update-to-data ratio is 1.0. At 64 environments and batch size 256,
one SAC update runs every four vector steps, keeping visual-encoder work per
collected transition close to the current two-mini-epoch PPO setup.

```bash
cd isaac_rl

# Four transitions, one gradient update, and a checkpoint.
./run.sh sac-smoke

# 64 environments and two million collected transitions.
./run.sh sac-train 64 2000000

# Deterministic held-out playback.
./run.sh sac-play logs/asymmetric_sac/<run>/checkpoints/final.pt 20
```

SAC writes independent logs and checkpoints under `logs/asymmetric_sac/`; it
does not change the PPO task configuration, checkpoint format, or entry point.
On the current local 570-series NVIDIA driver, `run.sh` automatically disables
the incompatible cuDNN path. It leaves cuDNN enabled on Euler's 580-series
driver unless `ISAAC_RL_DISABLE_CUDNN` is explicitly set.
