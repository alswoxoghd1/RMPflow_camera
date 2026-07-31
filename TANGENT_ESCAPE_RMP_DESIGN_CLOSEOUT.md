# Tangent Escape RMP Design Closeout

## 결론

현재 Tangent Escape RMP 설계는 다음 범위에서 마무리한다.

1. Escape leaf는 scalar natural-form RMP로서 기존 root natural form에
   symmetric PSD rank-one metric contribution을 더한다.
2. Escape가 꺼진 NORMAL production discrete map은 인증된 평형점 근방에서
   locally exponentially stable하다.
3. Escape를 포함한 switching 실행은 명령 유계성, 유한 episode 종료,
   양의 dwell, NORMAL 감소 근방 재진입, 이후 무재활성화라는 조건 아래
   NORMAL 안정성으로 연결한다.
4. 한 continuously active Escape episode는 12초 hard timeout 뒤
   smooth lambda-down으로 종료되며, 1초 동안 재진입이 억제된다.

Score, memory, timer, hard switching을 포함하는 supervisor 전체를
structured GDS라고 주장하지 않는다. 모든 장애물 환경에 대한 global
asymptotic stability와 collision-safe-set forward invariance도 이 결과의
범위가 아니다.

## Natural-Form Compatibility

고정된 Escape mode에서 scalar tangent task의 metric과 natural force는

```text
m_y = m_t lambda B >= 0
f_y = m_y (a_y - Jdot_y qdot)
```

이고 root contribution은

```text
DeltaM = J_y^T m_y J_y
Deltaf = J_y^T f_y
```

이다. 따라서 `DeltaM`은 symmetric PSD이고 `rank(DeltaM) <= 1`이다.
이 성질은 supervisor의 GDS 여부와 별개로 RMPflow natural-form
pullback/resolve 인터페이스를 보존한다.

Production solver를 사용한 synthetic blocking probe 결과:

- 20/20 ToF 활성화
- 1,060 active samples, 실패 0
- maximum rank: 1
- minimum eigenvalue of `DeltaM`: `-2.324e-11`
  (수치 허용오차 안의 roundoff)
- maximum symmetry residual: `5.145e-12`
- maximum `|n^T t|`: `3.331e-16`
- minimum root metric eigenvalue: `2900.878`
- maximum regularized resolve residual: `9.531e-12`
- clear input에서 `DeltaM=0`, `Deltaf=0`

## NORMAL Certificate

고정 goal에서 NORMAL command map을 선형화한 이산 Jacobian을 `A`라 하면

```text
rho(A) = 0.994964007 < 1
A^T P A - P = -I, P > 0
```

이다. 비선형 production map을 평형점 주변에서 샘플링한 결과
`||x-x*||_2 <= 0.01` 범위에서 `Delta V < 0`이 관측됐다.

또한 모든 sensor clearance가 `d_i >= d_off=0.24 m`인 clear set에서는
거리 activation과 blockage가 0이므로 Escape entry guard가 열리지 않는다.

## Conditional Hybrid Result

마지막 Escape episode 종료 시각을 `K`라고 한다. 다음 조건을 둔다.

1. Escape 동안 software command bounds가 유지된다.
2. 각 continuously active Escape episode는
   `T_E,max=12 s`와 handoff 시간 안에 종료된다.
3. 유효 branch 전환 사이에 양의 dwell time이 유지된다.
4. 마지막 Escape 뒤 상태가 샘플링된 NORMAL 감소 근방 `Omega_N`에 들어간다.
5. 그 뒤 Escape가 다시 활성화되지 않는다.

그러면 `k >= K`에서는 인증된 NORMAL map이 적용되므로 command state는
NORMAL 평형점으로 수렴한다. 실제 RB10의 inner-loop tracking disturbance가
bounded라면 물리 상태에 대해서는 평형점 근방으로의 practical convergence를
주장한다.

## Validation Result

다섯 fake 시나리오를 자동 판정했다.

- 세 접근-후퇴 반복: command bounds, finite release, dwell, NORMAL re-entry,
  post-reentry no-reactivation 모두 PASS
- clear environment: Escape activation 0, command bounds PASS
- persistent obstacle: goal convergence는 판정 대상에서 제외하고
  `|qdot| <= 1 rad/s`, `|qddot| <= 10 rad/s^2` PASS
- hard-timeout 단위 시험: reason 7, smooth lambda-down, episode OFF,
  1초 재진입 cooldown 경로 PASS

세 positive 반복의 minimum post-Escape state distance는 각각
`3.826e-5`, `3.679e-8`, `5.180e-9`이다.

## Figure Guide

- `(a)` NORMAL discrete eigenvalues: 단위원 내부 여부
- `(b)` nonlinear Lyapunov samples: `Delta V < 0` 여부
- `(c)` active Escape root metric: ToF별 root SPD와 rank-one contribution
- `(d)` episode timeline: 실제 Escape 활성 구간
- `(e)` hybrid checks: bounds, finite exit, dwell, re-entry, no reactivation
- `(f)` claim boundary: formal, conditional experimental, not claimed 분리

## Reproduction

Local certificate:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch rb10_rmpflow_rviz \
  rb10_rmpflow_production_stability_validation.launch.py \
  params_file:=/home/song/ros2_ws/src/RMP_Proximity-Sensor/config/params.yaml \
  goal_x:=0.5347335338592529 \
  goal_y:=-0.6670728921890259 \
  goal_z:=0.5349549651145935 \
  enable_tangent_escape_rmp:=true
```

Final dossier:

```bash
cd ~/ros2_ws/src/RMP_Proximity-Sensor
python3 scripts/build_tangent_escape_design_dossier.py \
  --local-yaml /home/song/ros2_ws/log/rmpflow_production_stability/production_stability_20260728_235726.yaml \
  --params-file /home/song/ros2_ws/src/RMP_Proximity-Sensor/config/params.yaml \
  --manifest /home/song/ros2_ws/log/rmpflow_production_stability/tangent_escape_closeout_manifest_20260728.yaml \
  --output-prefix /home/song/ros2_ws/log/rmpflow_production_stability/tangent_escape_design_closeout_20260728
```

## Paper Wording

사용 가능한 표현:

> The Tangent Escape leaf preserves the natural-form RMP composition by
> contributing a symmetric positive-semidefinite rank-one metric. The NORMAL
> production map is locally exponentially stable in the certified
> neighborhood. Conditional post-Escape convergence is established when each
> bounded Escape episode terminates, the state re-enters the NORMAL
> Lyapunov-decrease neighborhood, and no subsequent Escape activation occurs.

사용하면 안 되는 표현:

- The stateful Escape supervisor satisfies structured-GDS closure.
- The controller is globally asymptotically stable for arbitrary obstacles.
- Collision-safe-set forward invariance is formally guaranteed.
