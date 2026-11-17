# Current neighbor-direction revision — 2026-11-17

This section supersedes the 2026-11-15 observed-neighbor regression cost below.
Only the neighbor score changes; goal/execution scores, observed stall/rotation,
authority boost and removal of alpha_f/alpha_h are retained.

For fresh currently detected sensors j other than owner i, in common world coordinates:

- `c_j = clip(1 - rho_j / d_N,j, 0, 1)` (current calibrated sensor input, not clearance).
- `a_ij(theta) = max(0, n_j^T t_i(theta))`, with unit mounting axis and candidate tangent.
- `R_i(theta) = max_j c_j * a_ij(theta)`; empty set gives zero.
- `S_i = S_g + S_t - w_N R_i`, where S_g and S_t already include their weights.

There is no displacement magnitude, candidate joint prediction, regression model,
directional support, noise band or model-validation gate in this neighbor cost.
A first fresh detection contributes immediately; its own TG block need not be active.
Invalid, lost, masked and stale detections are excluded by the input path/freshness check.
The mounting-axis bearing is a proxy, especially for omnidirectional capacitive sensing.
This is a soft directional preference, not predicted link clearance or collision probability.
A penalized candidate can still win on total score. Different links need not move in
parallel to the owner's tangent; this simple cost intentionally does not predict that motion.

Main YAML: weight remains 0.3; `tg_rmp_neighbor_distances` uses sensor-control-point
order, 0.25 m except tof_N (index 6), 0.15 m. All entries must be positive and finite.
Old regression diagnostics remain for recording compatibility/analysis; they do not
enter the current candidate risk score. Actual response history still drives rotation.

Trace v12 keeps the v11 suffix length but changes its seven arrays to weighted cost,
alignment, closeness, detection count, critical sensor, current range, distance limit.
Alignment, closeness, critical sensor, range and distance limit describe the
maximum-cost sensor; no winner has id -1 and zero values. Detection count includes
fresh in-range non-owner sensors regardless of alignment. Old v11 remains decoded
with its original field names; archived parameters identify per-sensor cutoffs.
Tests cover first detection, owner exclusion, perpendicular/opposite directions,
common-frame rotation, maximum aggregation, cutoff, zero range and stale/lost input.
No physical experiment or latency validation has been performed for this revision.

---

# Current activation revision — 2026-11-15

This section supersedes older pocket/handoff explanations retained below as design history.

- Metric activation is now `alpha_distance * alpha_block`. `alpha_free` and
  `handoff_gain` remain in legacy trace slots at one, but do not suppress TG.
- No fade-to-zero when the owner changes. Existing distance hysteresis (5 mm),
  within-plane rate limiting, joint acceleration/command limits remain. Removing
  the fade does not prove smooth acceleration across different sensor Jacobians;
  inspect owner transitions in subsequent recordings before claiming improvement.
- In history mode, choose the nearest **eligible** sensor: fresh input, within TG
  activation range, positive goal-progress angle gate (40 degrees by default).
  A closer nonblocking sensor cannot steal ownership. An ineligible owner yields
  immediately to an eligible blocker. With no blockers the TG metric stays zero.
  The gate still uses the mounting axis as a proxy, not a measured obstacle bearing.
- Other detecting sensors participate through a candidate-specific soft cost,
  not an all-direction metric reduction. Both initial 8 candidates and later
  72-candidate scans use it. Existing observed-response rotation conditions remain;
  a new best score alone does not jump the applied direction.

## Observed-neighbor cost and its limits

Each candidate uses its existing bounded joint displacement prediction. Exclude
its owner. Require a current detection, a fresh valid local distance-response fit,
local pose/displacement trust radius, and a recent previous-model validation without
a residual above 3 sigma. Require directional support >= 0.7. These thresholds
are provisional engineering safeguards, not experimentally established guarantees.

For supported sensor j, use the learned scalar response `change_j = gradient_j * dq`.
Compute `sigma_j = noise_j * sqrt(2 + dq^T information_inverse_j dq)` and penalize
only `closing_j = max(-change_j - 2 sigma_j, 0)`. The contribution is
`support_j * exp(-max(range_j-clearance,0)/risk_length) * clamp(closing_j/risk_length,0,1)`.
Use the maximum across eligible non-owner detections. No observed closing means
no directional evidence, NOT certified free space. Unknown/unexcited/disappeared
sensors add no fabricated bearing-based penalty; unknown counts are recorded.
An unobserved initial direction therefore cannot be ruled out from scalar range alone.

The full score adds `-weight_neighbor_risk * cost` to existing goal/execution terms.
Main params.yaml enables this at 0.3 (provisional; no physical validation yet).
The old risk weight remains zero. No model is assumed to describe a permanent
obstacle; the existing expiry/discontinuity handling and new validation gates bound
its use locally, but slow obstacle changes can still violate that approximation.

## Recording and compatibility

History trace v11 appends the active neighbor-risk weight plus seven arrays of
72 values: weighted cost, support, valid model count, unknown count, critical sensor,
predicted distance change, and noise sigma. They describe the most recent candidate
scan, not new evaluations every control tick. The original shadow arrays retain
their original diagnostic meaning. Initial event negative-risk components include
the new cost; exact configuration/source is archived as before. v1–v10 remain readable.
Deprecated pocket/handoff parameters may still be accepted for old YAML compatibility
but no longer affect the metric. No Collision RMP changes are part of this revision.

---

# Escape RMP v2 — `tangent_ode`

## 센서 이력 방향과 단순 접선 leaf 비교

단일 정지 장애물에서 먼저 비교하기 위한 모드들이다. **Collision RMP의 정책,
입력 구체 생성, 가중치는 수정하지 않았다.** 거리 기반 push·metric 크기·거리
활성화·owner 선택은 공유한다. 기본 `simple`은 속도에 수직인 Cartesian 조향
leaf이며, `sensor_history`와 `fixed`는 센서 기준 접선 평면의 진행 leaf이다.

| `tg_direction_mode` | 방향 결정 |
|---|---|
| `sensor_history` | 초기 8개 후보에서 선택한 뒤 목표 진행, 현재 운동과의 연속성, 센서 이력 위험으로 각도를 연속 조정 |
| `simple` (기본) | 관절속도로 구한 센서 위치의 현재 속도에 수직인 조향. 장착 축을 장애물 방향의 대용으로 사용하며 스코어 탐색 없음 |
| `fixed` | 이전 단순 leaf. `tg_initial_angle`로 시작하고 같은 owner 안에서는 센서 기준 각도를 고정 |
| `legacy` | 기존 이웃 센서·I1·축 선호·stall 스코어 보존 |

`simple`의 별도 진입점은 `simple_tangent_leaf::evaluate()`이다. 이전 고정 방향
구현은 `evaluate_fixed()`로 보존하며 이력 모드의 후보 예측도 이 함수를 사용한다.
모드를 바꿀 때 controller를 다시 시작해야 한다. `tg_direction_mode`의 ROS 파라미터,
두 YAML 및 launch 기본값은 `simple`이며,
`enable_tangent_escape_rmp`는 기존 YAML 값을 따른다. SoC YAML은 현재 false이므로
비교할 때 명시적으로 true로 설정한다.

### 센서값이 의미하는 것

정전용량 센서는 장애물 방향이나 표면의 특정 점을 제공하지 않는다. 새 방향 위험은
센서 앞에 그린 가상 구체를 실제 장애물 위치로 해석하지 않는다. 센서별로 최근의
**실측 관절각과 융합된 스칼라 거리** 사이의 국소 관계를 학습한다.
ToF 융합 자체와 Collision 입력 경로는 그대로다.

`sensor_history`와 `fixed`의 접선 평면은 **URDF 센서 장착 축에 수직인 기준 평면**이다.
`simple`의 기준 평면은 **현재 속도에 수직인 평면**이다. 어느 모드도 실제 장애물
표면 법선을 측정했다고 해석하지 않는다. 기존 blockage/pocket 활성화 조건은 공유한다.

### 기본 simple: 속도에 수직인 조향

센서 위치의 속도는 거리 변화율이 아니라 관절 상태로 구한다.

```text
v_i = J_i(q_measured) * qdot_measured
u = v_i / |v_i|
raw_direction = -(I - u*u^T) * n_mount
              = u x (u x n_mount)
```

`n_mount`는 기존 센서 장착 축을 장애물 방향의 대용으로 쓰는 **가정**이다.
센서가 실제 장애물 방향을 측정했다는 뜻이 아니다. controller에서는 위치/속도를
명령 상태와 혼합하기 전의 실측 상태를 전달한다. 관절각으로 속도를 추정하는 설정도
사용할 수 있다. 직접 solver를 호출하여 실측 관측을 제공하지 않는 테스트에서는
solver 상태의 제어점 속도를 사용한다.

움직이는 동안 조향 방향을 현재 속도의 수직 평면에 투영하고, 그 안에서 이전 방향을
이어받아 장착 축 대용 방향에 따라 서서히 회전한다. 평면 안의 회전에는 기존 각속도
상한과 필터를 사용한다. 속도 방향 자체가 바뀔 때 필요한 투영 회전까지 같은 상한을
보장하는 것은 아니다. owner 변경은 기존처럼 영향력을 낮춘 뒤 이어받는다.

- `|v_i| >= tg_rmp_velocity_epsilon`이면 `t · v_i = 0`인 방향을 사용한다.
- 정지/저속에서는 해당 제어점의 목표 정책 가속도를 진행 방향으로 참고한다.
  그것도 없으면 센서 장착 축을 평면 기준으로 사용한다. 속도가 임계값에 가까워질수록
  이 기준을 실측 속도 평면으로 부드럽게 연결한다. 이 구간은 정지 탈출용
  확장이며, 작은 실제 속도에 대한 정확한 수직성을 요구하지 않는다.
- 속도와 장착 축이 평행하면 이전 방향을 투영해 유지한다. 첫 시작이면
  `tg_initial_angle`로 평면 안의 방향을 정한다. 거의 평행한 구간에서는 이 규칙을
  부드럽게 섞어 작은 속도 잡음에 우회 방향이 뒤집히지 않도록 한다.

따라서 기본값 π가 매번 -e1 방향을 강제하지 않는다. 초기 각도는 정면 정렬 등
방향이 정해지지 않는 상황의 선택 기준이다.

속도 수직 방향에는 기존 1차원 진행 좌표의 `d(t^T J)/dt`를 그대로 적용하지 않는다.
`t(v)^T v = 0`을 미분하면 의도한 조향까지 상쇄할 수 있으므로 Cartesian 위치
task map의 pullback을 사용한다.

```text
x = p_i(q),  J = J_i
M_x = m * t*t^T,  f_x = m * a * t
M_q = J^T M_x J
f_q = J^T (f_x - M_x * Jdot_i*qdot)
```

`m`과 거리/접근 기반 push는 기존 설정을 공유한다. 속도 수직 조향에는 기존
센서 평면의 진행 적분을 사용하지 않으며 progress attenuation을 적용하지 않는다.
이 성질은 leaf의 Cartesian 힘에 대한 것이고, 다른 RMP와 command 제한까지 합친
최종 로봇 가속도가 반드시 속도에 수직이라는 의미는 아니다.

로그 `direction_mode`는 legacy=0, sensor_history=1, fixed=2, simple=3이다.
이전 simple 로그의 2는 fixed로 해석한다. 새 로그에는 `steering_speed`,
`steering_velocity_dot`(t·v), `steering_velocity_source`(0=solver, 1=실측),
`steering_fallback`(비트 1=정지 목표 기준, 2=정지 센서 기준, 4=평행 근처),
`steering_turn_rate`를 추가했다. simple의 basis/θ는 현재 속도 수직 평면 기준이다.

### sensor_history의 방향 스코어

```text
S(theta) = w_goal * dot(t(theta), goal_motion)
         + w_continuity * dot(t(theta), current_motion)
         - w_risk * R_history(theta)

theta_dot = low_pass(clamp(direction_rate * dS/dtheta, -max_rate, max_rate))
```

`goal_motion`은 기존 목표 정책이 해당 센서 제어점을 움직이려는 방향이다.
정체를 기다리거나 학습을 위해 별도로 흔들지 않고, 평소 움직임으로 이력을 쌓는다.

1. 센서의 실제 수신 샘플과 그 취득 시각에 보간한 실측 관절각으로
   `d_i(q + dq) ≈ d_i(q) + g_i · dq`를 ridge 회귀한다. 같은 샘플의 재사용은
   학습 횟수로 세지 않는다. 정지 상태에서는 방향 응답을 알아냈다고 판단하지 않는다.
2. 각 후보 접선 leaf를 기존 root 정책(Collision 포함)에 합성하여 짧은 시간의
   관절 이동을 근사한다. 후보 동안 방향과 정책을 고정하는 근사이며, 이후 controller의
   command 제한·energy guard·실제 추종 지연까지 재현하는 예측은 아니다.
3. 학습된 센서의 거리 감소와 방향별 모델 불확실성을 비용으로 둔다.
   여유가 작을수록 비용이 커지고, 다른 센서가 멀어지는 것으로 접근 비용을 상쇄하지 않는다.
4. 이력이 없는 센서에는 장애물 위치나 예상 거리를 만들지 않는다. 가까운 감지가 있을 때
   그 센서 제어점의 예상 이동량을 약한 대칭 비용으로 둔다. 이는 미관측 장애물에 대한
   접촉 예측이 아니며, 장애물 방향을 모르는 상태에서 움직임을 줄이려는 보조항이다.

새 센서가 감지했다는 사실 자체에는 벌점을 주지 않는다. 충분한 거리에서 다른 센서로
감지가 옮겨가는 정상 우회를 허용한다. 최초 선택 이후에는 각속도 제한과 저역통과
필터를 사용한다. owner 교체 시에는 TG의 영향력을 낮춘 다음 기존 세계 접선을 새
기준 평면에 투영하고 영향력을 다시 높인다. 두 모드에 같은 handoff를 적용한다.

`proximity_obstacle_bridge`가 `/tg_rmp/sensor_ranges`에 `sensor_msgs/Range`를
추가 발행한다. 단위는 m, frame은 실제 센서 프레임, stamp는 취득 시각이다
(입력 stamp가 0이면 수신 시각). 기존 Collision용 최소 거리 clamp와 trigger를
적용하기 전의 값이다. NaN은 무효, +inf는 설정 범위 밖이다.
`tg_observation_max_range` 기본값 0.30 m는 관측 스트림의 설정 한계이며 센서의
물리적 정확도 보장이 아니다. 실제 보정 범위에 맞게 bridge YAML에서 조정한다.
스트림이 전혀 없으면 기하 시뮬레이션용 기존 활성화 입력을 쓰지만 센서 이력 학습은
이뤄지지 않는다. 아래 비교에서는 반드시 bridge와 실제 센서 입력을 사용한다.

### 실행과 비교 로그

같은 초기 관절 자세, 고정 목표, 단일 정지 장애물, 속도 및 TG metric 설정으로
두 실행을 따로 기록한다. 아래는 SoC test launch 예시이며 기존에 사용하던
하드웨어/시뮬레이션 옵션을 동일하게 붙인다. 표준 모델은 파일명에서 `SoC_`를 뺀다.

```bash
# A: 수정한 TG-RMP
ros2 launch rb10_rmpflow_rviz SoC_rb10_rmpflow_test.launch.py \
  use_proximity_bridge:=true enable_tangent_escape_rmp:=true \
  publish_tangent_escape_rmp_data:=true tg_direction_mode:=sensor_history

# 별도 터미널에서 기록. 실험 목표 설정 후 한 번의 접근만 포함하고 Ctrl-C로 저장 종료.
ros2 run rb10_rmpflow_rviz log_tg_rmp_observation.py \
  -t /tangent_escape_rmp_data -o history.csv

# A 종료 및 같은 시작 조건 복원 후 B: 기본 속도 수직 조향 leaf
ros2 launch rb10_rmpflow_rviz SoC_rb10_rmpflow_test.launch.py \
  use_proximity_bridge:=true enable_tangent_escape_rmp:=true \
  publish_tangent_escape_rmp_data:=true tg_direction_mode:=simple \
  tg_initial_angle:=3.141592653589793

ros2 run rb10_rmpflow_rviz log_tg_rmp_observation.py \
  -t /tangent_escape_rmp_data -o simple.csv

ros2 run rb10_rmpflow_rviz compare_tangent_direction_runs.py history.csv simple.csv
```

비교 도구는 첫 목표 도달까지의 시간·TCP 경로 길이·TG 활성 중 저속 시간·최소
센서 등가 거리·owner 전환·세계 접선 각속도 95백분위를 출력한다. 기록 중 목표를
바꾸지 않는다. 로깅 시작 전의 동작이나 0.1초 이상 기록 공백은 복원하지 못하며,
저속 시간은 정체의 보조 지표이다. 최소 등가 거리는 실제 표면 간격이 아니다.
`history_confidence`, 센서별 학습 샘플 수와 age도 기록하므로 이력 부족과 잘못된
방향 선택을 구분할 수 있다. 기본 비교는 모드 1/3을 요구하므로 이전 fixed 로그와
섞이지 않는다. 센서 평면을 동일하게 유지하며 스코어 유무만 비교하려면 B를
`tg_direction_mode:=fixed`로 실행하고 비교 도구에 `--baseline-mode fixed`를 붙인다.
`simple`과의 비교에는 방향 평면과 pullback 방식의 차이도 포함된다.

### 우선 조정할 값

현재 실행의 방향·metric을 CSV와 오프라인 HTML로 함께 관측하려면 별도 터미널에서
다음을 실행한다. controller 설정이나 목표를 변경하지 않는 구독 전용 관측기다.
기존 `record_data`의 관절·거리 기록과 함께 사용할 수 있다.

```bash
ros2 run rb10_rmpflow_rviz tg_rmp_html_monitor.py --duration 600
# 계속 기록하다 Ctrl-C로 종료하려면 --duration 0
# 기존 TG observation CSV에서 보고서만 생성:
ros2 run rb10_rmpflow_rviz tg_rmp_html_monitor.py --csv observation.csv --output report.html
```

`~/ros2_ws/data/tg_rmp_monitor/시각/`에 observation.csv, metadata.json,
report.html, report.summary.json을 저장한다. HTML은 5초마다 갱신되며 브라우저에서
새로고침하여 볼 수 있다. 보고서 생성은 낮은 우선순위의 별도 프로세스에서 처리한다.
Plotly가 필요하며 HTML 자체에는 JavaScript를 포함하므로 오프라인에서 열 수 있다.
관측하는 항목은 속도 수직성, owner 변경, 세계 방향 변화율, 거리/활성화 gate,
scalar metric과 joint 공간 metric norm의 관계 `||M_TG||_F = m ||J_t||²`이다.
전체 root 대비 비중이나 최종 가속도 변화의 확정에는 별도 root 진단이 필요하다.
데이터가 없거나 필요한 진단 필드가 없으면 해당 값은 미확인으로 표시한다.


| 파라미터 | 기본값 | 용도 |
|---|---:|---|
| `tg_initial_angle` / `tg_rmp_initial_angle` | π rad | simple 특이 상황의 방향 선택, fixed 방향, TG 초기 후보 기준 |
| `tg_rmp_history_weight_goal` | 0.5 | 목표 진행 선호 |
| `tg_rmp_history_weight_risk` | 1.0 | 거리 이력 위험 비용 |
| `tg_rmp_history_weight_continuity` | 0.1 | 현재 운동과의 방향 일치 |
| `tg_rmp_unknown_motion_weight` | 0.1 | 미학습 센서 이동량 보조 비용 |
| `tg_rmp_history_clearance` | 0.08 m | 센서 등가 거리의 여유 기준 |
| `tg_rmp_history_horizon` | 0.12 s | 짧은 후보 이동 근사 시간 |
| `tg_rmp_history_window` | 1.0 s | 국소 학습 이력 길이 |
| `tg_rmp_max_direction_rate` | 1.5 rad/s | 기준 평면 안에서 각도 조정 속도 상한 |
| `tg_rmp_direction_filter_tau` | 0.1 s | 각속도 필터 |
| `tg_rmp_handoff_time` | 0.15 s | owner 교체 시 fade 단계 시간 |

각속도 상한은 센서 평면 안의 θ에 대한 것이며 로봇 회전과 owner 교체까지 포함한
세계 접선 각속도의 상한은 아니다. 모델은 최소 6개 샘플과 실제 관절 변화가 있어야
유효하다. 기본 freshness 0.15초, 마지막 관측 모델 만료 0.6초, 후보 관절 이동량
상한 0.05 rad를 사용한다. 다른 방향으로 움직여 본 이력이 없으면 그 방향의
응답을 정확히 알 수 없다. 여러 장애물/동적 장애물 성능은 이번 단계의 검증 대상이 아니다.

기존 `tg_rmp_weight_i1`, 이웃 방향 스코어 등은 `legacy`에만 적용된다.
가상 구체를 고정 장애물로 가정했던 `tg_rmp_weight_predicted_risk`는 두 YAML에서
0으로 두었다. `tg_rmp_direction_rate`는 새 TG에도 각도 ODE의 gain으로 사용한다.

---

아래는 기존 스칼라 leaf와 legacy 방향 모델의 설계 기록이다. 접선/법선 관련 설명은
센서 기준 축에 대한 것이며, 새 모드의 방향 선택과 handoff는 위 설명을 따른다.

## Runtime selection

```yaml
enable_tangent_escape_rmp: true
tg_rmp_direction_mode: legacy
```

The current runtime uses the shared `tangent_ode` scalar policy with the
direction modes described above. Older acceleration-model variants mentioned
in historical experiments below are not the direction comparison switches.

## What it is

A one-dimensional leaf laid out beside the Collision leaf.

| | Collision | **Escape v2** |
|---|---|---|
| coordinate | \(x=d(\mathbf q)\) | \(\dot y=\hat{\mathbf t}^\top\mathbf v\) |
| Jacobian | \(-\hat{\mathbf o}^\top\mathbf J_p\) | \(\hat{\mathbf t}^\top\mathbf J_p\) |
| holonomic | yes | **no** — \(y\) is a path integral |
| metric | \(m_c(d)\) | \(m_e(d)\) |
| potential | infinite barrier | **finite well** |
| \(\boldsymbol\xi\) | \(\ne 0\) | **\(0\)** |
| direction | fixed \(\hat{\mathbf o}\) | **\(\theta\) ODE** |
| active count | per sensor | **one for the whole robot** |

Since \(\hat{\mathbf t}\perp\hat{\mathbf o}\), their reference axes are
orthogonal at the owner control point. The resulting joint motion is still
coupled through the root solve and the other leaves.

## Leaf

The leaf is designed in **canonical** `(a, M)` form. `f = M a` is only the
representation the pullback transports.

```
J_e   = t^T Jp                                    (1 x 6)
ydot  = J_e qdot

gamma(r) = 1 - sigmoid(r / v)                     closing-rate gate, r = ddot

a_e   = sat_[0,a_sat]( k_p e^{-d/l_p} + k_v [-gamma_v ddot]_+ / (d/l_v + eps_v) )
        - b_e ydot                                                 [m/s^2]
m     = alpha_TG gamma_m(ddot) mu / (d/l_m + eps_m)

f_e   = m a_e                                     (representation change)
f_q   = J_e^T (f_e - m Jdot_e qdot)
M_q   = m J_e^T J_e                                (PSD, rank <= 1)
```

### Activation

```
alpha_TG = alpha_d * alpha_block * alpha_free      metric only, never a_e
```

| factor | input | question |
|---|---|---|
| `alpha_d` | owner's own clearance | is the leaf near enough to have a say |
| `alpha_block` | `o . normalize(J_i qdd_nominal + cdot_i)` | is this point being driven into its obstacle |
| `alpha_free` | the owner's neighbours' clearances | is there any way out left in the tangent plane |

`alpha_d` is a quintic smoothstep from 1 at `activation_distance` to 0 at
`release_distance`. Those two numbers used to be the edges of a Schmitt
trigger, which never released once engaged -- measured active on 11291 of
11291 cycles. The latch also had to go before the metric can become hyperbolic
in `d`, since such a metric does not vanish at the release edge on its own.

`qdd_nominal` is the root solve over the goal-driven leaves with the obstacle
list emptied, so `alpha_block` asks where the goal *wants* this control point
to go, not where it is going. Both halves matter. It is an acceleration, so it
survives `qdot -> 0` at a local minimum -- which is what killed the old
velocity gate. And the obstacles are removed, so the target pull has nothing
to cancel against at exactly the configuration the leaf exists to handle.
Below `nominal_motion_epsilon` the direction is noise and the gate fails
**open**: an unreadable signal must not switch the leaf off.

The straight line from a sensor to the TCP goal was measured as the
alternative reference and rejected. It disagrees in sign with the nominal
motion on 52% of the cycles `tof3_1_E` owns the leaf and on all 139 that
`tof6_1_L` owns it -- and the disagreement is per sensor, not per link, so a
link-level average hides it.

Activation multiplies the metric and nothing else. Gating `a_e` as well put
alpha into `f = M a` twice, and since the damping term `-b_e ydot` was never
gated, a shrinking alpha collapsed the desired acceleration onto pure damping:
the leaf braked tangential motion instead of yielding it.

### Why canonical and not natural

Designing `f` directly was a real bug. With the force gate wider than the
metric gate, the implied desired acceleration is

```
a_e = f_e / m  ~  exp( +d (1/sigma_m - 1/sigma_a) )
```

which **grows with distance** -- 22x at 0.15 m for sigma_m=0.03, sigma_a=0.08.
The pullback hides most of it because `m -> 0` at the same time, so it is a
0/0 that barely reaches the command, but the policy is meaningless, the logged
`a_e` is nonsense, and the behaviour is hypersensitive to `sigma_m`.

`sigma_m << sigma_a` was the right conclusion; attaching it to the force was
the mistake. Gating the acceleration keeps `a_e` bounded by `a_max` at every
distance while the metric stays narrow:

| d | a_push [m/s^2] | m / m_max | state |
|---|---|---|---|
| 0.15 | 7.7 | 0.007 | suggests only |
| 0.08 | 18.4 | 0.069 | starts pushing |
| 0.04 | 30.3 | 0.264 | gaining authority |
| 0.02 | 38.9 | 0.513 | owns the tangent |

`t = cos(theta) e1 + sin(theta) e2`, where `e1`, `e2` come from the sensor's own
frame. **`t . o == 0` holds to machine precision by construction** — measured
`1.7e-16` in a closed-loop run — so there is no world-frame latch and no
re-projection anywhere.

### Why the curvature term vanishes

`m` is a function of clearance only. The direction state never enters the
metric, so `dm/dy = dm/dydot = 0` and both GDS curvature terms are identically
zero. This is the structural reason the direction search was moved entirely
into the force. (The residual approximation from `m`'s dependence on `q`
through `d` is the same one the Collision leaf already makes.)

### Analytic `Jdot_e qdot`

```
tdot        = theta_dot * t_perp  +  omega x t
Jdot_e qdot = tdot . v  +  t . (Jdot_p qdot)
```

The tangent changes for exactly two reasons — the ODE turns it, or the link
rotates — and both are closed form. No numerical differentiation. Both terms
are logged separately: a `theta` term much larger than the `omega` term means
`direction_rate` is too high.

### Finite well

The tangential field is rotational, so an unbounded well could inject energy
without limit. In canonical form the reachable energy is
`m_max * a_max * budget_scale`, since `f = M a`:

```
sdot = max(0, ydot) - s / tau
```

Only forward progress spends the budget; being pushed back does not refund it.
When the budget is spent the leaf goes quiet and the goal leaf grows relatively
larger, which is what lets a different solution emerge.

In natural form the push becomes

```
f_push = m_max a_max e^{-d/sigma_f} e^{-s/s0} alpha_free^2
1/sigma_f = 1/sigma_a + 1/sigma_m   =>  sigma_f = 0.022 m
```

which is the same shape as before -- only the parameterisation changed, and
`sigma_f` is now *derived* from the two gates instead of being set by hand. So
the finite-well argument carries over unchanged:

```
Phi_e = m_max a_max s0 e^{-d/sigma_f} e^{-s/s0}
integral of f_push ds  <=  m_max a_max s0  <  infinity
```

`alpha_free` appears squared because it gates both `a` and `m`.

The storage-function cancellation is still approximate: the budget drains at
least as fast as energy goes in, so the bound errs conservative.

### Neighbourhood table — all 20 ToF sensors

Derived from the sensor geometry at startup, not hand-maintained. Sensors on
the same link and within 100 mm along the link axis form one neighbourhood:
the two stations on links 2 and 3 are 200 mm apart and stay separate cross
rings, while the wrist's lateral trio and its forward sensor are 60 mm apart
and merge into one group of four.

A fixed `(next, previous, antipodal)` table cannot describe the wrist, which is
three lateral sensors plus one axial sensor. So each neighbour is instead
reduced to the angle at which its normal enters this sensor's tangent plane,
and contributes a dipole plus a quadrupole:

```
S(theta) -= w_j [ cos(theta - phi_j) + cos(2(theta - phi_j)) ]
```

which expands into the `a`, `b`, `c`, `e` coefficients. A neighbour whose
normal is antiparallel has no tangent-plane direction at all; it only says
"the far side is blocked too", which argues for leaving along the link axis,
and is kept as the separate `axial_block` term. A neighbour pointing the same
way says nothing and is dropped.

**On a regular cross ring (phi in {0, +-90, 180}) this collapses exactly to the
hand-written table, and `e` vanishes.** The 16 ring sensors are unchanged; the
generalisation only adds the four wrist sensors. A unit test asserts both:
20 sensors covered, 16 still reducing to the +-90 / 180 pattern.

`eps_i = sign(e1 . z_link)` makes `eps * e1` point toward the distal end.
The wrist's forward sensor has no axial component in `e1`, so it takes no
axial preference at all. `axial_preference` flips the sign globally.

### Direction ODE

```
S(theta)     = a cos(theta) + b sin(theta) + c cos(2 theta) + e sin(2 theta)
theta_dot    = k_theta dS/dtheta

a = w_g (e1 . dg) + w_v (e1 . vhat)
      + w_r ( eps * sum_axial w_j  -  sum_inplane w_j u_j cos(phi_j) ) + beta_e1
b = w_g (e2 . dg) + w_v (e2 . vhat)
      + w_r (                      -  sum_inplane w_j u_j sin(phi_j) ) + beta_e2
c = w_c ( -sum_inplane w_j u_j cos(2 phi_j)  -  sum_axial w_j )
e = w_c ( -sum_inplane w_j u_j sin(2 phi_j) )

w_j = exp(-d_j / risk_sigma)        u_j = |P_T o_j|
```

A coarse 8-seed argmax runs **only at activation** — the gradient vanishes at
minima too, so the ODE must not be seeded near one. The metric is ~0 at that
instant, so the discrete jump has no effect on the commanded acceleration.
There is no argmax during continuous operation.

### Pocket guard

`alpha_free = 1 - sigmoid((geomean_j w_j - pocket_threshold) / pocket_epsilon)`

The geometric mean runs over however many neighbours the sensor actually has,
so a three-neighbour wrist sensor and a four-neighbour ring sensor are judged
on the same scale.

When every neighbour is close there is no useful
direction left in the tangent plane; the way out is `-o`, which belongs to
Collision. `alpha_free` multiplies **both** the force and the metric, so the
leaf fades out together rather than leaving a live metric with no push.

### Ownership

One leaf for the whole robot: `i* = argmin d` over all 20 ToF sensors. `M_q` is
rank-one, so N leaves would tie up N degrees of freedom, and independent
per-sensor optima cancel in the pullback and deadlock. Sensor hand-off uses a
`switch_hysteresis` band; on switch, `theta` is re-expressed in the new basis so
the world direction is preserved.

## Parameters

All under `tangent_escape_ode_*` as node parameters.

| Parameter | Meaning | Default |
|---|---|---|
| `accel_max`, `accel_sigma` | `a_max` [m/s^2], gate kept **wide** | 50.0, 0.08 |
| `metric_max`, `metric_sigma` | leaf metric, gate kept **narrow** | 1e5, 0.03 |
| `budget_scale`, `budget_recovery_time` | `s0` [m], `tau` [s] | 1.0, 3.0 |
| `tangent_damping` | `b_e` as a **rate [1/s]**; `1/b_e` is the settling time and must stay far above the 5 ms control period | 40.0 |
| `direction_rate`, `risk_sigma` | `k_theta`, risk decay | 3.0, 0.06 |
| `weight_goal/velocity/risk/curvature` | score weights | 1.0/0.3/1.0/1.5 |
| `e1_bias`, `e2_bias` | signed preference for the sensor-frame tangent basis directions | 0.0, 0.0 |
| `axial_preference` | flips `eps` globally | 1.0 |
| `pocket_threshold`, `pocket_epsilon` | pocket guard | 0.6, 0.1 |
| `activation_distance`, `release_distance` | ownership band | 0.30, 0.40 |
| `switch_hysteresis` | sensor hand-off | 0.005 |

`metric_sigma < accel_sigma` is deliberate: the metric is a constraint on
tangential acceleration, so a wide metric would cap the goal leaf inside the
tangent plane. The *acceleration* is what should reach far.

`accel_max` is a real acceleration, so it can be read against the Collision
leaf directly: 50 m/s^2 is about 10% of Collision's 500 m/s^2 at contact.

## Diagnostics — schema 9

31-value fixed record, decoded by `scripts/rmpflow_trace_logger.py` into
`tangent_escape_ode_*` columns. Slot 26 is the schema id, shared with schemas
6-8; nothing else is shared.

Key columns: `theta_rad`, `theta_rate_rad_s`, `budget_remaining`,
`tangent_rate_m_s`, `score_a/b/c`, `push_accel_m_s2`, `metric`, `alpha_free`,
`injected_power`, `tangent_normal_residual`, `jdot_theta_term`,
`jdot_omega_term`.

## Verified

`test/test_tangent_escape_ode.cpp` (target `tangent_escape_ode`):

- ring table: 16 ring sensors, neighbours share link and station, antipodal
  normal is opposite, next/previous orthogonal, `eps * e1` distal, stored basis
  orthogonal
- score gradient matches finite difference
- inactive leaf contributes exactly zero
- `|t . o| <= 1e-12`, `|t| = 1`, `t . t_perp = 0` every cycle
- `M_q` symmetric PSD with rank <= 1; force lies along that same axis
- budget never negative, never grows without forward progress
- pocket guard tightens monotonically and gates the metric with the same factor
- `axial_preference` flip accepted

Closed-loop run (fake proximity, `tof3_1_S`, 40 s):

```
schema_id                    9
sensor_index                 8   (tof3_1_S)
clearance                    0 .. 0.399
theta                    0.248 .. 0.849 rad     continuous
budget_remaining         0.736 .. 0.989
tangent_normal_residual  max 1.665e-16          <- structural orthogonality
jdot_theta_term         -0.022 .. 0.095
jdot_omega_term         -0.263 .. 0.021         <- omega dominates: k_theta OK
```

## Not claimed

- Standard GDS closure. The task map is not holonomic (§ leaf), so the theorem
  is not cited; stability rests on the finite budget instead.
- Collision-safe-set forward invariance.
- That the budget-recovery term (`tau`) preserves the storage bound during
  recovery intervals.

## 상세 재생 데이터 기록

표준/SoC main 및 test launch에서 `record_data:=true`이면
`record_rmp_details:=true`가 기본 적용된다. `auto_start_recording`과 기존
recorder start/stop 서비스를 함께 따르며 CSV 옆에 `.details.jsonl`과
`.details.metadata.json`, 실행 YAML/URDF 사본을 저장한다.
`rmp_details_rate:=30.0`으로 큰 행렬/geometry의 기록 주기를 조절한다.
상태·명령·TG compact 패킷은 제어 주기마다 발행하며 같은 sequence/제어 시각을 사용한다.
필요 없으면 `record_rmp_details:=false`로 끈다.

JSONL은 제어 스냅샷, TG 원본 진단, 센서 원본 및 TG 정규화 Range,
관절 상태와 reference/encoder 디버그 스트림, 목표, 파라미터 변경을
각각 수신 시각과 함께 저장한다. 상세 기록 시 reference/encoder 위치
디버그 발행도 활성화한다. 제어 스냅샷에는 주기 번호, ROS/steady 시각,
backend 입력 상태와 solver 상태, 제한 후 qdd, controller가 backend에
넘긴 명령, guard 상태, 목표 자세, 장애물 proxy, 그룹별 root M/f,
전체 root M/f와 raw qdd, solver-native 제어점/TCP Jacobian·곡률,
동일 계산의 TG 진단이 들어간다. 스키마는 sidecar metadata에 명시하며
`rmp_detail_capture.decode_snapshot()`으로 읽을 수 있다.

그룹별 frozen-policy 제거 결과는 기존 ablation 계산을 재사용한다.
따라서 상세 진단에는 추가 root resolve 비용이 있다. 발행 주기 감소만으로
solver의 진단 계산 비용이 줄어드는 것은 아니며 제어 timing은 별도로 확인한다.
여러 장애물 pair의 원래 task 정책을 개별 저장하는 형태가 아니라
Target/Collision/TG/다른 정책 그룹의 joint 공간 기여를 저장하는 구조다.
센서 기반 구체는 실제 장애물 방향을 측정한 좌표가 아닌 proxy다.
명령은 backend 입력 단계이며 backend 내부 변환/네트워크 송신 이후의
실제 로봇 추종 결과와 구분한다. snapshot sequence는 실제 command cycle 기준이다. 대기 gate 구간에는 command cycle이
없으며, compact 패킷의 비어 있는 행렬은 세부 기록 주기 제한을 뜻한다. 수신 누락은
sequence와 steady 시각을 함께 사용하여 분석한다.

### RViz 기록 재생

`sensor_history` 모드와 `record_rmp_details:=true`를 함께 사용하면
기존 `.details.jsonl`의 동일 제어 스냅샷 안에 확장 진단이 자동으로 포함된다.
`simple`, `fixed`, `legacy`에는 이 확장을 붙이지 않는다.

- 매 제어 주기: 갱신 전/평가/갱신 후 각도, 목표·연속성·history 항의 실제
  점수 기울기, 제한 전/후 및 필터 후 각속도, 센서 전환·초기화 이벤트,
  실제 사용한 센서 거리·획득 시각·hold 여부, nominal 가속도,
  현재 방향 평가에 사용한 예측 관절 변위와 제한 전 크기를 저장한다.
- 최초 방향 선택: 평가한 후보 8개의 각도와 항별 점수를 저장한다.
- 모델이 바뀔 때: 기준 관절 자세·거리, 기울기 6성분, information inverse
  6×6 행렬, 유효 여부·표본 수·noise·rate를 저장한다. 설정된 적분 시간 기준
  0.5초마다 모든 센서 모델을 다시 기록해 수신 누락 후 복원을 가능하게 한다.
- 실제 history 설정에는 코드 안의 estimator 상수도 포함한다. ROS 파라미터
  스냅샷, 복사한 YAML/URDF, 컴파일된 소스 식별자와 설치 바이너리 해시도 보존한다.
  소스 사본은 `_sources.zip`에 저장하고 빌드 당시 해시와의 일치 여부를 표시한다.

파싱은 `rmp_detail_capture.decode_snapshot(data)['history']`를 사용한다.
확장은 TG schema-9 기존 필드 뒤 `45045,1,payload_length`로 시작하며,
필드명·이벤트 코드·모델 갱신 사유는 `.details.metadata.json`에 기록된다.
`theta_evaluated`와 `risk_gradient_before`는 실제 갱신에 사용한 값이다.
기존 TG 진단의 history gradient는 갱신 후 재평가 값이므로 서로 구분한다.
`filtered_rate_before`는 센서 전환 시 필터 리셋을 적용한 후, 이번 적분 직전 값이다.

메타데이터의 `recording_quality`에서 제어 sequence 누락, history 확장 누락,
해독 오류, 모델 복원이 불완전한 프레임 수를 확인한다.
`parameter_snapshot_complete`와 `runtime_build_matches_manifest`도 확인한다.
모델 delta를 읽을 때 누락이 있으면 다음 full keyframe까지 완전 복원을 주장하지 않는다.
기록 시작 직후에도 첫 full keyframe 수신 전 구간은 미완전으로 표시한다.
history 확장 전체가 들어 있는 `/tangent_escape_rmp_data`의 중복 사본은 생략한다.
동일한 TG 패킷은 `/rmp_recording_snapshot` 안에 보존되며 생략 건수는
`deduplicated_history_messages`로 남긴다. 다른 모드의 기존 TG 기록은 유지한다.
기록된 `dt`는 설정된 적분 간격이고 실제 실행 간격은 스냅샷의 steady 시각 차이다.
저장 후 `ros2 run rb10_rmpflow_rviz rmp_history_record.py /경로/기록.details.jsonl`로
파일을 다시 검사할 수 있다. `direction_update_mismatches`는 기록된 입력으로
재계산한 각속도 필터·각도 적분과 저장 출력이 일치하지 않는 프레임 수다.

이 확장은 진단만 추가하며 TG 방향 선택·Collision 정책을 변경하지 않는다.
시뮬레이션 검증은 파일 구조와 진단 정합성을 확인하며 실기 제어 주기 성능은
해당 장비의 timing 데이터로 별도 평가한다.

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch rb10_rmpflow_rviz rmp_recording_replay.launch.py
```

최신 저장 파일을 선택하고 일시정지 상태로 시작한다. 재생 창에서 Play / Pause,
시간 슬라이더, 속도, TG peak(가장 큰 TG metric 시각)를 선택한다.
특정 기록은 `dataset:=/절대경로/파일.details.jsonl`로 지정한다.
별도 ROS domain 213에서 로봇 모델과 시각화만 실행하며 하드웨어 명령은 발행하지 않는다.

로봇 모델의 alpha는 0.4이며 감지 센서는 주황 구체로 강조한다.
상세 JSONL이 있으면 동일 root metric으로 분해한 Target(초록)은 TCP에서,
Collision(빨강)은 감지 센서에서, TG(파랑)는 TG 담당 센서에서 표시한다.
Collision은 전체 Collision 그룹이 해당 센서 움직임에 주는 기여이며 센서별 독립 leaf 힘은 아니다.
흰색/노란색 합성 가속도의 기준점은 TCP/센서 선택 상자로 바꾼다. 주황 화살표는 TG 선택 방향,
하늘색은 TG 요청 가속도다. 화살표 길이는 화면에 표시된 배율을 사용하고 0.6m에서 잘린다.
기존 관절 CSV와 TG observation만 있는 기록은 로봇 자세와 TG만 재생하며,
저장하지 않은 Target/Collision 기여는 표시할 수 없다.

2026-11-11 15:16 및 15:32 실행은 snapshot 파일 경로의 launch 파라미터 타입 오류로
저장 노드가 시작 직후 종료되어 기록이 남지 않았다. 해당 오류는 URDF/설정 경로를
각각 명시적 문자열 파라미터로 전달하도록 수정하고 simulation launch에서 저장을 검증했다.

2026-11-11: `sensor_history`의 목표 방향 항은 TCP 위치 오차를 감쇠 역기구학
`J_tcp^T (J_tcp J_tcp^T + 0.01^2 I)^-1`로 관절 진행 방향에 연결한 뒤,
담당 센서의 Jacobian으로 변환해 정규화한다. 속도 감속과 곡률은 이 방향 항에
포함하지 않는다. TCP 목표에 도달했거나 해당 센서의 진행 성분이 0이면 목표 항은 0이다.
막힘 판단에는 기존 기준 가속도를 계속 사용한다. `goal_from_nominal`은 이 모드에서
false이며 `goal_e1`, `goal_e2`에는 새 진행 방향의 접선축 투영이 저장된다.
센서 전환과 감지 공백의 방향 초기화, metric 파라미터는 이번 수정 범위에 포함하지 않는다.

이후 활성화 수정: `sensor_history`의 block도 위 목표 진행 기준을 사용한다.
`v_goal_i = J_i J_tcp^T (J_tcp J_tcp^T + 0.01^2 I)^-1 (goal - tcp)`와
센서 장착축의 정규화 내적을 기존 `smoothstep(0, 0.3)`에 넣는다.
기준 크기가 `tg_rmp_nominal_motion_epsilon` 이하이면 block은 0이다.
정지나 후퇴 중에도 목표 진행 기준이 유효하면 block을 계산한다. 장착축은
장애물 방향의 근사이며 정전용량 센서가 측정한 장애물 법선이 아니다.
후보 방향을 평가하는 고정 방향 trial도 동일한 block을 사용한다.

기록의 history 확장은 버전 2로 갱신했다. 기존 nominal 진단 필드는 보존하고,
끝에 `block_reference_source`(1: 목표 진행, 0: 기준 가속도), 유효 여부, 기준
크기의 임계값, 관절 진행 벡터 6개, 담당 센서 진행 벡터 3개를 저장한다.
디코더는 이전 버전 1 기록도 읽는다. 18:58:44 기록의 11/24/53초 회귀 사례는
block=1, 19:33:16 기록의 29.68/29.82/29.995초 사례는 block=0을 요구한다.

2026-11-11 angle gate update: history activation now uses
`theta_deg = acos(clamp(n_i dot normalize(v_goal_i), -1, 1)) * 180/pi`,
`alpha_block = clamp(1 - theta_deg / tg_rmp_block_start_angle_deg, 0, 1)`.
The start angle defaults to 40 degrees: 0/10/20/30/40 degrees give
1/0.75/0.5/0.25/0. Invalid angles or unavailable progress close the gate.
The fixed history candidate trials share this rule. Other modes retain their
dot smoothstep thresholds. The mounting axis remains an unmeasured bearing proxy.
History trace v2 reference source 2 identifies the angle-linear goal-progress
gate (source 1 identifies older goal-progress dot smoothstep recordings).
The controller parameter snapshot records the configured start angle.
Previously eligible recorded blocked cases must retain positive activation,
rather than the old saturated value of 1.

History direction scheduling update: sensor/pose acquisition and local model
updates continue independently of TG authority. Activation (including the
40-degree angle gate, distance, pocket, handoff and rate metric factors) is
computed before the history direction objective. With metric <= 1e-9, ownership
is tracked but candidate search and direction integration are skipped; the
angular filter is reset to zero. The first positive metric selects eight seeds.
Current sensor velocity alignment participates in that initial selection.
Temporary zero authority on the same detection preserves its selected angle;
a sensor ownership change requires a new initial selection when authority
returns. Losing observations or leaving the release range still resets state.
Candidate trial predictions retain the same activation factors as production.
The existing history trace records initialization events at the new time and
zero direction rates during pauses; the metric-on audit viewer uses these events.

### History escape effectiveness (2026-11-12)

History direction score adds `weight_escape * alpha_block * tanh(tangent.dot(J * delta_q) / escape_length)`.
`delta_q = H*qd + .5*H*H*qdd` uses the combined base+trial TG root solve,
the configured global joint acceleration cap, and the existing .05 rad local
trust radius. This is a frozen-policy kinematic prediction, not a simulation
of backend guards, servo tracking, or the obstacle geometry. Defaults: weight
0.2, scale .01 m, H .12 s. Weight zero disables both this score and global probes.
Initial seeds and continuous finite-difference ascent use the same score.
Every .1 s a coarse 8-candidate search supplies a bounded, held angular bias
when another basin has higher score; theta is never reset by that search.
Existing angular rate limit and filtering apply. This does not guarantee C2
will beat C0, nor identify which direction an unobserved obstacle occupies.

Trace 45045 v3 preserves all v2 fields, then appends weight_escape,
escape_length, reselect_interval, escape_gradient (including global probe),
escape_before, escape_after, seed_escape_count, weighted seed escape scores.
Retained event 46046 v2 has 85 doubles: original 77 plus 8 escape scores.
Decoders retain support for old recordings. Candidate plots/CSV include the
new term separately; risk fields remain risk only.

The existing detailed recorder subscribes to `/rb10/measured_joint_states`
even in the minimal profile, without saving a duplicate raw stream. It writes
`/tg_rmp/escape_feedback` derived records (max 50 Hz). It uses independent
encoder increments and the interval-start sensor Jacobian/tangent, not backend
reference increments. This is a local linear displacement audit, not exact FK.
Same-owner/unchanged-goal rolling intervals span .12 s, with at most .05 s
encoder age. Missing encoders, invalid ranges, and short episodes stay unknown.
Default movement deadband is .5 mm; pressure thresholds propagate 3*sqrt(2)
times a 2 mm equivalent-range noise floor through the existing softplus
pressure function, independently per sensor. These are initial thresholds,
not experimentally calibrated stationary-noise estimates. Encoder positions,
thresholds, checked/unknown sensors, motion prediction errors, supported
range-model prediction errors and available linearized TCP goal progress are
saved for auditing. Local progress is not proof of complete obstacle bypass.
These outcome labels do not introduce a stall-triggered activation or override
direction selection; incoming distance history still updates the original
response model. Physical clock alignment and servo/model error remain limits.

### Measured response direction update (2026-11-12, supersedes continuous escape ascent)

History initial 8-candidate scoring retains its goal/motion/risk/escape terms.
After initialization, continuous score ascent and periodic global-maximum
pulling are replaced by TgResponseDirection. The controller subscribes to
independent `/rb10/measured_joint_states`; backend reference state is not used
as a substitute for missing encoder observations. Existing range regression
still uses its original backend timeline; this change adds a separate measured
response input. Collision and distance/block activation remain unchanged.

While the owner/goal remains fixed, fresh encoder samples form .12s windows.
The start Jacobian and world tangent project encoder increments into signed
progress. Progress below .5mm/window or half the recent decaying best speed
allows a small trial when block > .1. Goal error below 1cm suspends trials.
The first trial side is chosen using scores at current theta +/- 5deg; its
quality is not presumed. Rotation follows bounded filtered proportional angle
tracking. The trial window starts after angle error < .005rad and angular
speed < .1rad/s. Progress improvement > .5mm/.12s with no observed near-range
worsening accepts the trial. Sufficient progress holds the direction; continued
poor progress may request another step. No improvement (including ambiguity)
returns to the last accepted angle, and the next attempt uses the opposite
side. After two failed attempts further probing stops until encoder joint
motion > .01rad from the failure pose, or the episode/owner/goal resets.
Returning an angle does not command a return to an old robot pose.

Near-range worsening: a fresh paired range below .08m decreases by more than
3*sqrt(2)*.002m. This is an initial conservative equivalent-range rule, not an
obstacle-bearing measurement or physical collision boundary. It also aborts
during the trial turn. Missing/stale encoder (> .05s), no paired range evidence,
and incomplete windows do not establish improvement. Owner/goal changes,
metric-off pauses and observation loss invalidate the comparison. Thresholds
are compiled initial settings awaiting experimental calibration. This is
local observed improvement, not a proof that obstacles caused a slowdown or
that the globally best bypass direction was found. In particular controller
limits and servo tracking can also affect observed progress.

Trace 45045 v4 adds 10 values after the v3 tail: phase,result,target,accepted,
speed,baseline,failures,encoder_stamp,best_speed,seeded. Phases: 0 observe,
1 turn-to-trial,2 evaluate-trial,3 return-to-accepted. Results: 2 trial requested,
1 improved,0 no clear improvement,-2 range worsened,-3 unavailable. The former
escape_gradient field now records the correction needed to replace the score
gradient with the feedback angular command; it is not a physical score
partial derivative. Retained initialization events keep the v2 wire layout.
The existing recorder-side encoder audit remains independently available.

Direct hardware backend automatically publishes independent measured joint
states in sensor_history mode, including when detailed recording is disabled.
Other backends must supply the measured topic explicitly; missing physical
feedback holds the initialized direction rather than substituting reference.
Range freshness uses the controller observation time separately from encoder
sample time, since a proximity sample can be newer than the latest encoder.
In v4, weight_escape=0 removes predicted mobility from initialization and trial
side ranking only; measured response trials remain active. The old
reselect_interval parameter is retained for trace compatibility but is unused
by the new direction controller. Outcome diagnostics are visible in the RViz
audit status: observation, turning, evaluating, returning, measured speeds and
accept/reject/unavailable verdicts. The main recorder's standalone outcome
labels and the controller's trial verdict are distinct diagnostics.

### Bounded response exploration (2026-11-12, supersedes single-window verdicts)

Direction feedback now evaluates three fresh 0.12 s windows (at least 0.36 s)
per verdict. At least two windows and the mean must support improvement or
sustained degradation. Positive progress above 4.17 mm/s holds the direction;
comparison with the historical maximum is diagnostic only. Neutral trials may
continue in 5 degree steps up to 30 degrees from the exploration anchor.
A rejected trial returns to the accepted angle and tries the other side.
Two failed sides inhibit further exploration until 0.05 rad joint displacement.
The existing near-range deterioration check remains active during turning.
A block change above 0.2 restarts comparison evidence. These thresholds are
initial settings, not calibrated guarantees of successful avoidance. Scalar
capacitive ranges do not establish obstacle bearing or certify a free path.
Goal progress is not a separate trial score in this revision; the actual
signed tangent progress detects persistent reverse motion. Metric changes
alone are not yet included in comparison invalidation.

Trace schema v4 stays compatible: phase/result/target/accepted/speed/baseline
and failure count retain their meanings. During evidence collection speed is
the latest window value; at a verdict it is the three-window mean. Internal
window counts are not exported in v4.

### Wide same-state response search (2026-11-12, supersedes 5-degree trial rollback)

A three-window measured stall triggers a 37-candidate search at 5-degree
spacing over anchor +/-90 degrees. Candidates use the current pre-TG root,
fixed-direction TG pullback, velocity damping, curvature, regularization and
uniform acceleration limit. The objective is projected point acceleration
`t dot (J qdd + curvature)` in m/s². It is not a prediction of obstacle geometry
or successful escape. At least 0.05 m/s² improvement over the current direction
is required; absent/invalid models hold the direction. Search runs on decision
events, not each 200 Hz tick. Trace records the complete last sweep and its cost.

The chosen angle is approached through the existing angular rate/filter.
Lack of forward progress during this turn no longer causes rollback at 5 degrees.
Three windows with majority positive progress and mean >4.17 mm/s accept the
CURRENT angle and stop the search early. An unproductive settled endpoint
excludes that side and searches the opposite half of the original interval.
After an empirically failed endpoint, an untried opposite-side candidate must
predict positive acceleration; it need not exceed the failed endpoint model.
Both exhausted sides/no viable candidate hold until a 0.05 rad posture change.
The initial/search anchor is not an empirically accepted return angle.

Near-range decrease >8.49 mm below 80 mm aborts turning, monitored both against
recent paired ranges and the search-start ranges. With an accepted angle the
controller returns the direction (not robot pose); otherwise it holds the
current direction. Stale encoders freeze steering. Missing paired range data,
range changes >20 mm, block changes >0.2, or metric ratios >4 across a window
invalidate comparisons and hold turning; three comparable windows resume the
saved target unless forward progress already accepts the current direction.
These are initial heuristic thresholds. They can inhibit exploration when
observations remain variable; actual recordings must validate their effect.

History trace 45045 v5 appends 11 scalars after v4: anchor, has_accepted, sign,
sweep_sequence, current_score, best_score, chosen_angle, sweep_us, scan_metric,
tried (side mask), bound_clamped; then 37 scores at offsets -90:5:+90 degrees.
Phase 4=hold, 5=wait for comparable observations. Result -4=incomparable window,
-5=no improved untried candidate. Explicit angular-bound clipping is recorded
so audit checks distinguish constrained integration from an ODE mismatch.

### Continuous full-circle guidance (v6, supersedes v5 bounded trials)

Initialization retains eight coarse seeds for immediate activation and retained
initial-event compatibility. It now uses the SAME physical objective as the
72-candidate (0:5:355 degrees) periodic refinement. Candidate state is frozen;
TG direction, damping and pullback are recomputed with the pre-TG root. The
existing regularization, uniform acceleration cap, 0.12 s prediction horizon
and joint trust radius apply. This is a controller-local approximation, not a
geometric free-path prediction or an actual future robot rollout.

For each candidate, dq = h*qd + 0.5*h²*qdd (trust-radius limited):
 goal = weight_goal * tanh(goal_unit dot J_TCP*dq / escape_length)
 execution = weight_escape * tanh(t dot J_sensor*dq / escape_length)
 risk = weight_risk * History.evaluate(dq).value
 physical_score = goal + execution - risk.
The existing risk value includes valid local range models and explicitly
symmetric unknown-motion exposure; absent models are not evidence of clearance.
Initial seeds use physical_score. Subsequent choices add
 weight_continuity*(1+progress_evidence/3)*cos(angle-retained_target).
The original weights remain configurable. These changes intentionally replace
post-initialization acceleration-only ranking and geometric goal alignment.

Scores update no faster than max(0.24 s, history.reselect_interval), using the
existing fresh independent encoder windows. Score EWMA gain is 0.35*confidence.
Confidence is 1/(1+range_change/.02+abs(log(metric_ratio))+5*abs(block_change)),
bounded to [0.1,1]. Metric/range variability reduces confidence; it does NOT
stop turning. Stale encoders or unavailable paired ranges still freeze steering.
The goal-progress measurement uses J_TCP at the window start times measured
encoder displacement; it is local linear kinematics, not exact measured FK.

Forward tangent progress >4.17 mm/s with goal retreat no worse than 4.17 mm/s
adds confidence to good-progress evidence; otherwise poor-progress evidence.
Evidence is bounded to [0,3] and opposite evidence subtracts confidence. Good
progress reinforces continuity and records an accepted heading, but does not
cancel the current destination halfway through a turn. Temporary goal retreat
is permitted; persistent retreat allows reconsideration rather than forcing
an immediate reversal.

A small same-side target adjustment requires score advantage >0.03. A target
change >45 degrees or reversal requires advantage >0.08, two consecutive
preferred headings within 15 degrees, and poor-progress evidence >=1.5.
Angles use shortest circular differences; the +/-90 degree bound is removed.
The existing rate/filter limits still apply to every angular update. Thresholds
are initial design settings, not validated universal constants.

Near-range deterioration (>8.49 mm decrease below80 mm) holds the current
heading immediately. A rolling 0.5 s anchor also detects cumulative approach.
Three fresh windows with range change <=2 mm release the hazard hold and clear
old ranking evidence. There is no automatic return to an old robot pose or to
a previously accepted heading in a changed environment.

Trace 45045 v6 retains the v4 response scalars and v5 compatibility scalars,
then has 72 filtered PHYSICAL scores (no continuity yet), six new scalars
(observation_confidence, goal_speed, poor_evidence, progress_evidence,
pending_angle, pending_count), and four arrays of72: raw weighted goal,
execution, positive risk cost, continuity. Decision score = filtered physical
score + continuity. current_score denotes filtered retained-target score;
best_score denotes best filtered candidate plus continuity. Both dimensionless.
The v6 suffix is387 doubles. The initial-event queue still preserves its eight
actual initialization candidates. RViz audit plots show the full-circle scores
and evidence, and exports add *_full_circle_candidates.csv.

## v7: observed-response cumulative steering (2026-11-12)

This section supersedes v6's post-initialization destination selection and
hazard hold. It applies to `sensor_history`. Target, Collision, block activation,
TG magnitude/metric parameters and the simple mode are not changed.

The initial eight-seed choice remains a **model prior**, not a claim about an
obstacle surface. Afterwards the 72-angle score is used to suggest a rotation
sign. The controller no longer commands the score-maximizing remote angle.
It advances the tangent-plane angle cumulatively when observed execution fails.

1. Observe independent encoder displacement and paired fresh range samples in
   0.12 s windows. Repeated acquisition stamps do not create evidence. Missing
   feedback freezes steering at the applied angle; it does not prove success.
2. Tangent speed above 0.5 mm / 0.12 s, without the near-range deterioration
   below, supplies good-motion evidence. Goal retreat is allowed temporarily.
   Retreat faster than 4.17 mm/s accumulates confidence-weighted time; after
   0.6 s it counts as failure unless near-sensor readings improve (>2 mm opening
   and no >2 mm closing). This is a provisional response criterion, not a
   geometric-clearance or global-progress guarantee. Goal-speed diagnostics
   use the frozen TCP Jacobian and independent encoder displacement.
3. Poor evidence >=1.5 starts a rotation. The positive and negative angular
   half-planes compare the existing filtered goal+execution-risk score plus
   continuity minus observed failure cost; the current +/-15 degrees and
   antipode do not vote. A margin <=0.03 retains the previous sign (initially
   positive), explicitly a tie-break rather than evidence of a better side.
4. Failed windows advance the retained target by
   `min(max_direction_rate, pi/3) * window_dt * (0.5 + 0.5*poor_evidence/3)`.
   Target lead is limited to 20 degrees ahead of the **applied** direction.
   Existing ODE rate limits and filtering still apply. No fixed 30/90-degree
   endpoint or automatic return to the seed is used. One successful window
   pauses additional advance; good evidence >=2 retains the working angle.
5. Failure memory is a fixed 72-element array. Only directions within 30 degrees
   of the **evaluated window-start angle** receive a triangular weighted cost,
   capped at0.6. Good observations reduce local cost and decay memory with a
   10 s time constant. Untried directions have no failure evidence; this is not
   evidence of safety. Owner/goal resets discard this episode's memory.
6. A reversal requires at least30 degrees of actual angular travel in the
   current rotation leg, >=0.6 s of observations, poor evidence >=1.5 and an
   opposite-side advantage >0.08 on two consecutive model scans. A completed
   360-degree leg with continuing failure also reverses. These are experimental
   persistence settings, not a theorem or proof of an optimal escape direction.

The existing near-range criterion (>8.49 mm decrease below80 mm in a paired
window or against a rolling0.5 s reference) contributes failure evidence. It
**does not freeze candidate evaluation or the steering ODE**. Phase4 now means
steering while a worsening range response is observed. The previous requirement
for three windows with absolute change <=2 mm is removed. A stable small range
alone neither proves successful escape nor releases a separate safety state.
Collision RMP continues to supply its normal response independently.

Model solves remain limited to72 per scan, no faster than
`max(0.24 s, history.reselect_interval)`. Evidence and angular targets update in
observation windows. Memory and ranking are fixed-size O(72), without unbounded
history or additional candidate solves. All force/clearance predictions remain
conditional on the existing model; the sensor mounting axis is not promoted to
a newly measured obstacle bearing or normal by this change.

Trace45045 v7 retains the entire v6 suffix, then appends17 scalars:
`reason, turning, turn_travel, turn_elapsed, reverse_count, range_closing,
min_range, adverse_evidence, retreat_evidence, positive_score, negative_score,
sign_margin, window_dt, paired_samples, turn_rate_limit, failure_weight,
evaluated_angle`, followed by72 **already weighted** failure costs.
The complete suffix is476 doubles. `chosen_angle` is the best model proposal;
`target` is the cumulative steering target and may differ. `current_score` now
refers to the applied angle. Guidance = filtered physical + continuity - failure.
`turn_travel` accumulates applied angular travel in the current sign, not target
travel or robot displacement. `min_range` is an equivalent sensor range.

Reason codes:0 observe,1 measured motion,2 start for stall,3 accumulate for
stall,4 start for range worsening,5 accumulate during range worsening,
6 persistent alternative reversal,7 failed full-circle reversal,
8 block inactive,9 missing feedback,10 invalid model candidates.
The recorder and audit retain v1--v6 decoding. RViz distinguishes v7 phase4
from the old hazard hold and plots failure costs separately from model scores.

Related design sources (adaptations, not imported guarantees):
- Kim et al., *Escaping Local Minima: Hybrid Artificial Potential Field with
  Wall-Follower for Decentralized Multi-Robot Navigation*,
  https://arxiv.org/html/2409.10332v1 — cumulative rotation and persistent side.
  Its 2D directional LiDAR and rotated attraction are not available/used here.
- Becker et al., *Informed circular fields*,
  https://doi.org/10.3389/frobt.2024.1447351 — coherent bypass direction and
  candidate continuity. Its geometric point-cloud rollout is not reproduced.

Validation must distinguish regression tests (continued updates after a noisy
range drop, cumulative turn, observed recovery, reversal, bounded rate, trace
compatibility) from real-robot escape success. Old recorded trajectories cannot
establish the closed-loop performance of the new policy.

## v8: sensor-evidence shadow recording (2026-11-14)

This is the observation-only stage: v7 control costs, activation, failure memory,
and steering decisions remain unchanged. `record_rmp_details:=true` captures the
new evidence in `sensor_history` recordings; no new control opt-in is enabled.
Simple mode is unchanged. These diagnostics must be validated before using them
to alter steering or reduce uncertain-risk penalties.

Each periodic72-angle scan records seven unweighted diagnostic arrays:
nominal_cost (predicted closing alone), uncertainty_cost (extra conservative
model cost), unknown_exposure, support, critical_sensor, critical_change,
critical_sigma. The original risk satisfies
`value = nominal_cost + uncertainty_cost + unknown_exposure` and remains the
control input. The critical sensor is the freshly observed, modeled sensor with
the greatest predicted closing; its signed change is not an obstacle bearing.
Sigma is an approximate noise/model scale, not a calibrated probability bound.
Support is the minimum freshness-weighted excitation support among freshly
observed sensors, zero for an unmodeled observed sensor or a candidate outside
the local joint-space trust radius. No fresh observations also means zero.
Unobserved space is not certified safe by any of these values.

Before each newly accepted training sample enters fitting, the previous model
is evaluated at the sample's interpolated pose. Recorded fields per sensor:
stamp, model_stamp, predicted, observed, error=observed-predicted, sigma, support,
valid. Repeated acquisitions never create independent validation observations.
Abrupt sensor changes are tested before the model is reset. Missing/out-of-trust
models produce invalid checks, not zero error. The pose is the existing model's
configured pose source (which may be reference); this is not a claim of encoder
ground truth or validation of an unexecuted candidate rollout. It measures
held-out, next-observation model error, not closed-loop escape success.

Trace45045 v8 appends seven arrays of72, three observation-window fields
(shadow_window_stamp, shadow_stalled, shadow_worsening), sensor count, eight
fields per PredictionCheck, and the extension length including that last value.
Length =509+8*N doubles, after the complete476-double v7 suffix. Stall and
near-range deterioration are separately labeled even when both occur. Existing
v1--v7 recordings remain readable. The initial-event queue preserves eight seed
scores using the appended length; candidate diagnostics do not change seeds.

For an offline report, run (absolute or relative recording/output paths):

```bash
python3 /home/song/ros2_ws/src/RMP_Proximity-Sensor/scripts/tg_sensor_shadow_audit.py \
  recording.details.jsonl --output sensor_evidence --start 0 --duration 60
```

Outputs: candidates.csv, prediction_checks.csv, observed_outcomes.csv,
summary.json and sensor_evidence.png. Scans and validation samples are deduplicated
by their recorded identifiers/stamps. RViz audit also labels the observation-only
side comparison. Comparisons use supported candidates in each angular half-plane;
support<0.6 is excluded. Overlapping nominal-to-conservative cost envelopes are
reported as no resolved difference. A side without supported candidates is
insufficient support, not safe. These are experimental diagnostic thresholds,
not confidence tests or claims about unobserved alternatives. The PNG names its
selected scan time and sensor; the CSV contains every captured scan.

### Rotation-triggered TG metric boost (history mode, trace v10)

The optional `tg_rmp_authority_boost_enabled` feature starts its metric ramp when
observed-response direction search starts rotation. Insufficient command duration
alone does not trigger it. No booster waiting or settlement gate delays rotation
or failure evidence. Existing search conditions and score weights stay unchanged.

While rotating without successful measured progress, gain rises at 1/s to at most
2. Successful progress or ending rotation reduces gain at 1/s. Range danger also
reduces gain. Missing feedback, inactive activation gates and context changes reset
it. Collision, distance/block gates and desired TG acceleration remain unchanged.
Both TG metric and natural force use the same gain. Frozen candidate solves reuse
current gain. This scheduled multiplier is not an additional direction score or a
new stability guarantee.

Parameters: `tg_rmp_authority_boost_{enabled,maximum,rise_rate,decay_rate}`.
The main params.yaml enables it; node/library defaults remain disabled.

Trace v10 retains its 14-field extension for compatibility. Base metric, applied
and state gains, phase and command coverage remain recorded. Former deficit/cap
times, exhausted flag and trigger/settle times are now reserved zero fields.
Old phase/reason IDs remain readable; new control uses idle, rising, recovering
and danger. Use base_metric * applied_gain to reproduce the applied leaf metric.
