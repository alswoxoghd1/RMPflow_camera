# Diffusion Policy / TG-RMP tuning profiles

These profiles keep one experiment's values together and are loaded after
`config/params.yaml` and the launch defaults with `tuning_params_file`.

They are experimental settings, not a safety layer.  First replay them with a
fake/dry-run backend.  For later real-hardware checks, use reduced speed, an
empty guarded workspace, and a reachable stop.  Do not test around people.

## Profiles

- `tg_rmp_diffusion_off.yaml`: TG-RMP contribution disabled.  Collision RMP,
  proximity input, and goal conditioning remain enabled.
- `tg_rmp_diffusion_observe.yaml`: evaluates and logs the old TG settings but
  contributes zero TG metric/force to the solve.
- `tg_rmp_diffusion_conservative.yaml`: first low-authority live candidate.

Run the same recorded Diffusion Policy episode in the order above.  This keeps
the controller-facing goal filter constant while isolating the TG contribution.

## Launch and save a run

Replace the profile name and run label for each run.  The synchronized input
velocity filter is enabled explicitly because the launch default is `false`,
even though the base YAML says `true`.

```bash
source /opt/ros/humble/setup.bash
source /home/song/ros2_ws/install/setup.bash

TG_PROFILE=/home/song/ros2_ws/src/RMP_Proximity-Sensor/config/tuning_profiles/tg_rmp_diffusion_off.yaml
TG_RUN=tg_off_01
TG_OUTPUT=/home/song/ros2_ws/log/tg_rmp/$TG_RUN
mkdir -p "$TG_OUTPUT/rmp_goal" "$TG_OUTPUT/rmpflow_trace" "$TG_OUTPUT/params"

ros2 launch rb10_rmpflow_rviz rb10_rmpflow.launch.py \
  cb_simulation:=false \
  use_proximity_bridge:=true \
  tuning_params_file:="$TG_PROFILE" \
  use_synced_input_velocity_filter:=true \
  synced_input_velocity_filter_alpha:=0.35 \
  synced_input_velocity_filter_beta:=0.02 \
  measured_velocity_feedback_blend:=0.35 \
  use_rmp_goal_logger:=true \
  rmp_goal_log_rate:=200.0 \
  rmp_goal_log_directory:="$TG_OUTPUT/rmp_goal" \
  rmp_goal_log_prefix:="$TG_RUN" \
  use_rmpflow_trace_logger:=true \
  rmpflow_trace_log_rate:=200.0 \
  rmpflow_trace_log_directory:="$TG_OUTPUT/rmpflow_trace" \
  record_joint_velocity:=true \
  joint_velocity_log_directory:="$TG_OUTPUT"
```

In another sourced terminal, save the effective runtime values rather than
relying only on the intended YAML:

```bash
ros2 param dump /rmpflow_controller --output-dir "$TG_OUTPUT/params"
ros2 param dump /rmpflow_bridge --output-dir "$TG_OUTPUT/params"
ros2 param dump /proximity_obstacle_bridge --output-dir "$TG_OUTPUT/params"
```

Also save the Diffusion Policy checkpoint/config, episode identifier, random
seed, observation horizon, action horizon, inference rate, and whether the pose
is absolute or delta.  `/RMP_goal` must be expressed in `base_link`; the bridge
does not transform a camera/tool-frame pose.

## Analyze goal discontinuities

The goal logger records the raw `/RMP_goal` as `external_goal` and the filtered
controller input `/goal_pose` as `controller_goal` in the same CSV.

```bash
GOAL_CSV=/home/song/ros2_ws/log/tg_rmp/tg_off_01/rmp_goal/tg_off_01_YYYYMMDD_HHMMSS.csv

ros2 run rb10_rmpflow_rviz analyze_goal_jitter.py "$GOAL_CSV" \
  --pose-prefix external_goal \
  --json-output /home/song/ros2_ws/log/tg_rmp/tg_off_01/raw_goal.json \
  --spike-csv /home/song/ros2_ws/log/tg_rmp/tg_off_01/raw_goal_spikes.csv

ros2 run rb10_rmpflow_rviz analyze_goal_jitter.py "$GOAL_CSV" \
  --pose-prefix controller_goal \
  --json-output /home/song/ros2_ws/log/tg_rmp/tg_off_01/filtered_goal.json
```

With the supplied 200 Hz filter and a 0.30 m/s cap, a controller-facing
translation update should be no larger than roughly 1.5 mm per normal 5 ms
tick.  Investigate raw spikes separately instead of compensating for them by
raising TG damping or metric authority.

## Tuning order

Change only one parameter family per repeated episode:

1. Fix goal semantics and discontinuities: frame, absolute/delta convention,
   quaternion sign/normalization, inference cadence, and action-chunk handoff.
2. Stabilize obstacle lifetime: keep `trigger_distance` beyond
   `tg_rmp_alpha_off_distance`; do not create/remove an obstacle where TG alpha
   is still nonzero.
3. Stabilize selection and derivatives: owner hysteresis, obstacle-motion use,
   clearance-rate filter, and deadband.
4. Increase authority last: acceleration gain/saturation, closing gain, metric
   `mu`, and tangent damping.  Stop increasing when escape is adequate.

The conservative profile starts with obstacle-motion compensation and closing
gain off.  Re-enable them independently only after the off/observe comparison
shows that raw goal, owner selection, timing, and velocity feedback are stable.
