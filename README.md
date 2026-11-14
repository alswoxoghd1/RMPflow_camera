# rb10_rmpflow_rviz

`rb10_rmpflow.launch.py`를 실행해서 RB10 RMP controller와 RViz를 함께 띄우는 ROS 2 Humble 패키지입니다.

현재 구현은 아래 조합을 사용합니다.

- `Eigen`: 수치 계산
- `Pinocchio`: 실제 제어 주기의 RB10 운동학과 explicit RMP2 root 누적
- `CasADi`: Algorithm-3 DAG 참조 구현, task-map 미분, backend 동등성 검증

실제 제어 기본값은 `solver_backend: pinocchio_direct`입니다. CasADi 전체
solver는 `solver_backend:=casadi_rmp2`를 명시한 검증 실험에서만 사용합니다.

TG-RMP의 센서 이력 기반 방향 선택과 단순 접선 leaf 비교는
[TG 방향 비교 실행 방법](TANGENT_ESCAPE_ODE.md#센서-이력-방향과-단순-접선-leaf-비교)을 참고하세요.
기본값은 `tg_direction_mode:=simple`이며, 관절속도로 계산한 센서 위치의 속도에
수직인 방향으로 조향합니다. `sensor_history`는 기존 이력 기반 TG,
`fixed`는 이전 고정 각도 접선 leaf, `legacy`는 기존 복합 스코어입니다.

이 문서는 처음 받은 사람이 GitHub에서 내려받아 바로 실행하는 기준으로 다시 정리한 사용 가이드입니다.

## 1. 권장 환경

- Ubuntu `22.04`
- ROS 2 `Humble`
- 작업공간 경로: `~/rb10_rl_sim`

중요:
- 현재 [`CMakeLists.txt`](/home/song/rb10_rl_sim/rb10_rmpflow_rviz/CMakeLists.txt) 는 CasADi 설치 경로를 `~/rb10_rl_sim/third_party/casadi/install` 로 가정합니다.
- 따라서 이 README대로 하면 가장 덜 헷갈립니다.

## 2. 작업공간 받기

`~/rb10_rl_sim` 경로에 작업공간을 받습니다.

```bash
cd ~
git clone <YOUR_GIT_REPO_URL> rb10_rl_sim
cd ~/rb10_rl_sim
```

이미 다른 이름으로 clone 했다면:
- `CMakeLists.txt`의 `CASADI_ROOT`를 수정하거나
- 아래 설치 경로를 직접 맞춰야 합니다.

## 3. 시스템 패키지 설치

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  libeigen3-dev \
  libboost-filesystem-dev \
  libboost-serialization-dev \
  libtinyxml2-dev \
  python3-colcon-common-extensions \
  python3-rosdep \
  ros-humble-rclcpp \
  ros-humble-std-msgs \
  ros-humble-sensor-msgs \
  ros-humble-geometry-msgs \
  ros-humble-visualization-msgs \
  ros-humble-tf2-ros \
  ros-humble-interactive-markers \
  ros-humble-pinocchio \
  ros-humble-robot-state-publisher \
  ros-humble-joint-state-publisher \
  ros-humble-rviz2
```

`rosdep`를 아직 초기화하지 않았다면:

```bash
sudo rosdep init
rosdep update
```

## 4. CasADi 설치

이 패키지는 `CasADi`를 로컬로 빌드해서 사용합니다.

```bash
mkdir -p ~/rb10_rl_sim/third_party
git clone --depth 1 --branch 3.7.2 https://github.com/casadi/casadi.git ~/rb10_rl_sim/third_party/casadi

cmake -S ~/rb10_rl_sim/third_party/casadi \
  -B ~/rb10_rl_sim/third_party/casadi/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=~/rb10_rl_sim/third_party/casadi/install \
  -DWITH_PYTHON=OFF \
  -DWITH_PYTHON3=OFF \
  -DWITH_EXAMPLES=OFF

cmake --build ~/rb10_rl_sim/third_party/casadi/build -j"$(nproc)"
cmake --install ~/rb10_rl_sim/third_party/casadi/build
```

정상 설치되면 이 파일이 있어야 합니다.

```bash
ls ~/rb10_rl_sim/third_party/casadi/install/lib/libcasadi.so
```

## 5. 빌드

```bash
source /opt/ros/humble/setup.bash
cd ~/rb10_rl_sim
colcon build --packages-select rb10_rmpflow_rviz --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source ~/rb10_rl_sim/install/setup.bash
```

참고:
- 실행 파일에는 CasADi `RPATH`가 들어가 있으므로 보통 `LD_LIBRARY_PATH`를 따로 잡지 않아도 됩니다.

## 6. 실행

### 전체 실행

`robot_state_publisher`, `rmpflow_controller`, `interactive_goal`, `obstacle_manager`, `tof_ray_visualizer`, `rviz2`를 함께 띄웁니다.

```bash
source /opt/ros/humble/setup.bash
source ~/rb10_rl_sim/install/setup.bash
ros2 launch rb10_rmpflow_rviz rb10_rmpflow.launch.py
```

CasADi 참조 backend를 명시적으로 비교할 때만 다음을 추가합니다.

```bash
solver_backend:=casadi_rmp2 casadi_rmp2_jit:=true
```

### 옵션

RViz 없이 실행:

```bash
ros2 launch rb10_rmpflow_rviz rb10_rmpflow.launch.py use_rviz:=false
```

장애물 매니저 없이 실행:

```bash
ros2 launch rb10_rmpflow_rviz rb10_rmpflow.launch.py use_obstacles:=false
```

모델만 확인:

```bash
ros2 launch rb10_rmpflow_rviz rb10_model_only.launch.py
```

## 7. 실행 후 조작

- RViz에서 goal marker를 움직이면 end-effector 목표가 바뀝니다.
- obstacle manager가 켜져 있으면 obstacle marker를 움직여 회피 동작을 볼 수 있습니다.
- controller는 [`config/params.yaml`](/home/song/rb10_rl_sim/rb10_rmpflow_rviz/config/params.yaml) 을 읽어 동작합니다.

## 8. 처음 실행이 안 될 때 확인할 것

### `libcasadi.so`를 못 찾는 경우

아래 파일이 있는지 확인:

```bash
ls ~/rb10_rl_sim/third_party/casadi/install/lib/libcasadi.so
```

만약 작업공간를 `~/rb10_rl_sim`가 아닌 다른 경로에 두었다면:
- [`CMakeLists.txt`](/home/song/rb10_rl_sim/rb10_rmpflow_rviz/CMakeLists.txt)의 `CASADI_ROOT`를 수정하고 다시 빌드해야 합니다.

### 빌드는 되는데 launch가 안 되는 경우

아래 순서로 다시 실행:

```bash
source /opt/ros/humble/setup.bash
source ~/rb10_rl_sim/install/setup.bash
ros2 launch rb10_rmpflow_rviz rb10_rmpflow.launch.py
```

## 9. 핵심 파일

- launch: [`rb10_rmpflow.launch.py`](/home/song/rb10_rl_sim/rb10_rmpflow_rviz/launch/rb10_rmpflow.launch.py)
- controller 파라미터: [`params.yaml`](/home/song/rb10_rl_sim/rb10_rmpflow_rviz/config/params.yaml)
- controller node: [`rmpflow_controller_node.cpp`](/home/song/rb10_rl_sim/rb10_rmpflow_rviz/src/rmpflow_controller_node.cpp)
- 실시간 solver: [`pinocchio_direct_solver.cpp`](/home/song/rb10_rl_sim/rb10_rmpflow_rviz/src/pinocchio_direct_solver.cpp)
- 참조 solver: [`casadi_rmp2_solver.cpp`](/home/song/rb10_rl_sim/rb10_rmpflow_rviz/src/casadi_rmp2_solver.cpp)
- backend 동등성 테스트: [`test_solver_backend_equivalence.cpp`](/home/song/rb10_rl_sim/rb10_rmpflow_rviz/test/test_solver_backend_equivalence.cpp)

## 10. 빠른 실행 요약

```bash
cd ~
git clone <YOUR_GIT_REPO_URL> rb10_rl_sim
cd ~/rb10_rl_sim

sudo apt update
sudo apt install -y \
  build-essential cmake git \
  libeigen3-dev \
  python3-colcon-common-extensions python3-rosdep \
  ros-humble-rclcpp ros-humble-std-msgs ros-humble-sensor-msgs \
  ros-humble-geometry-msgs ros-humble-visualization-msgs \
  ros-humble-tf2-ros ros-humble-interactive-markers \
  ros-humble-robot-state-publisher ros-humble-joint-state-publisher \
  ros-humble-rviz2

mkdir -p ~/rb10_rl_sim/third_party
git clone --depth 1 --branch 3.7.2 https://github.com/casadi/casadi.git ~/rb10_rl_sim/third_party/casadi
cmake -S ~/rb10_rl_sim/third_party/casadi -B ~/rb10_rl_sim/third_party/casadi/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=~/rb10_rl_sim/third_party/casadi/install \
  -DWITH_PYTHON=OFF -DWITH_PYTHON3=OFF -DWITH_EXAMPLES=OFF
cmake --build ~/rb10_rl_sim/third_party/casadi/build -j"$(nproc)"
cmake --install ~/rb10_rl_sim/third_party/casadi/build

source /opt/ros/humble/setup.bash
colcon build --packages-select rb10_rmpflow_rviz --symlink-install
source ~/rb10_rl_sim/install/setup.bash
ros2 launch rb10_rmpflow_rviz rb10_rmpflow.launch.py
```
# RMP_Proximity-Sensor
# RMP_Proximity-Sensor

### TG direction decision audit (offline)

Run `ros2 launch rb10_rmpflow_rviz tg_direction_audit.launch.py` after sourcing
`install/setup.bash`. This separate viewer selects the latest history JSONL,
uses ROS domain 214, and starts paused. It launches only replay visualization.
Use `dataset:=/absolute/path/to/file.details.jsonl` to choose a recording.

Choose a TG metric-on transition (`metric > 1e-9`, observe-only off), then
**TG 켜진 순간으로** to return to its first saved contributing sample.
Direction arrows and direction-update graphs are hidden while TG is off.
C0–C7 are shown in RViz only when an initial selection is recorded while TG is on. The highlighted candidate maximizes
the recorded weighted total (with the controller's tie tolerance). The table
contains weighted goal alignment, weighted current-motion alignment, negative
history cost and total. Initial candidate scores are labelled with their own
time; they are not reevaluated at subsequent cursor positions. The table may
reference an earlier, pre-activation choice, explicitly labelled as a reference.

The five-second timeline starts at TG metric-on and shows the three local score derivatives, raw/limited/
filtered angular rates, the selected angle, block and whether TG contributes.
Positive derivative increases local theta; it is not necessarily global left
or right. Sensor changes and resets are shown separately; local angles across
different sensor bases are not comparable. TG-off direction arrows are hidden; orange selected arrows mean nonzero metric
with observe-only off. Off intervals remain gaps rather than joining across them.
Candidate arrows have equal length and encode direction, not force magnitude.

**이 5초를 논문용 파일로 저장** exports CSV, JSON, PDF, SVG and PNG under
`~/ros2_ws/data/tg_direction_audit/<recording>/metric_on/`. Offline batch export:

```bash
python3 ~/ros2_ws/src/RMP_Proximity-Sensor/scripts/tg_direction_audit.py \
  /absolute/path/to/file.details.jsonl --output /absolute/path/to/results
```

Quantitative outputs include recorded candidate score margins, score-sum errors,
initial argmax agreement, angular filter/integration residuals, world direction
turn rates on consecutive same-owner samples, model coverage and missing cycles.
These are implementation diagnostics, not proof of globally optimal avoidance.
The current-motion term is velocity alignment, not agreement with the previous
chosen direction. The mounting-axis plane is not a measured obstacle surface.
Missing initial scores and incomplete history models are explicitly marked.
Five-second windows may overlap and must not be counted as independent trials.
Actual avoidance efficacy requires repeated matched experiments (goal completion,
progress, stalls, clearance proxies, and motion smoothness), with the sensing
limitations stated. Recorded reference joint states are not independent physical
pose measurements.

Initialization events are now captured separately on
`/tg_rmp/initial_direction_events` with their original control cycle, ROS/steady
clocks, owner, sensor plane, reference joint pose and all eight weighted scores.
A preallocated 64-event SPSC queue keeps the control thread free of DDS/file I/O;
a non-control ROS timer drains at most 64 events per tick. Reliable transient
local history retains the last 64 publications for late subscribers. Recorder
caches are bounded and events are deduplicated on disk. Queue overflow and decode
errors are reported in metadata; this is bounded retention, not unlimited storage.

**초기 8후보 정지화면** freezes the robot at the recorded initialization pose,
including when the normal snapshot containing the event was dropped. Event-based
windows start at the original selection time. Events received before recording
retain their original timestamps and can be displayed as earlier references.
Old recordings without the new event stream remain readable; missing candidate
scores in old files cannot be recovered retrospectively.

Detailed recording now defaults to coherent snapshots plus initialization events,
RMP enable flags and parameter changes. Duplicate joint/command/goal streams and
individual raw sensor topics are omitted from the detailed JSONL; the normal
CSV remains available. Root metrics/forces, sensor histories and provenance needed
for policy analysis remain in coherent snapshots/metadata. The launch option
`record_detail_raw_topics:=true` restores the additional raw topic capture.
The JSONL writer uses compact JSON and buffered writes (flushed during metadata
updates and on stop). Five-second audit exports still mark missing telemetry
rather than inventing direction updates between saved samples.
