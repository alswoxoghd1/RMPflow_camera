# RMPflow 전체 제어기 검증 보고서

검증일: 2026-07-23

## 1. 검증 범위

이번 검증은 다음 전체 경로를 대상으로 했다.

```text
Target + Axis + Collision + Damping + JointLimit + Tangent Escape
    -> natural-form pullback / root resolve
    -> joint acceleration limit
    -> velocity integration
    -> joint/velocity command guard
```

물리 RB10에는 명령을 보내지 않았다. 모든 폐루프 실행은 fake simulation backend와
격리된 ROS domain에서 수행했다.

현재 운용 설정과 논문 가정을 만족시키기 위한 증명용 설정은 서로 다른 목적의
설정이다. 두 결과를 하나의 보장으로 합치지 않는다.

## 2. 최종 판정

| 검증 항목 | 현재 운용 설정 | 증명용 `paper_gds` 설정 |
|---|---|---|
| 전체 leaf의 수치적 유한성 및 root solve | PASS | PASS |
| 고정 mode structured-GDS closure 조건 | NOT COVERED | PASS |
| 고정 mode 조건부 Lyapunov 비증가 | NOT COVERED | PASS |
| Escape metric 대칭/PSD 및 tangent 구조 | PASS | PASS |
| Escape ON/OFF 동일 상태 비교 | PASS | PASS |
| risk-damped zero-effect handoff | PASS | PASS |
| 전체 hybrid switching 안정성 | NOT PROVEN | NOT PROVEN |
| 장애물 안전집합의 formal forward invariance | FAIL | FAIL |
| `paper_gds` joint open-domain invariance | 해당 없음 | CONDITIONAL PASS |
| 운용 command guard의 joint-limit 유지 | EMPIRICAL PASS | 해당 없음 |
| fake 폐루프 수치 회귀 | PASS | 구조 검증용 실행 PASS |

따라서 정확한 결론은 다음과 같다.

1. Tangent Escape leaf는 RMP natural form으로 전체 root solve에 정상 결합된다.
2. 증명용 고정 mode에서는 structured-GDS closure와 조건부 Lyapunov 비증가가
   자동 검증됐다.
3. 현재 운용 설정은 안정성 정리의 가정을 만족하지 않으므로 같은 formal
   certificate를 주장할 수 없다.
4. 현재 Collision RMP만으로 장애물 거리 집합의 hard forward invariance는
   보장되지 않는다.
5. zero-effect handoff는 검증됐지만, memory와 mode switching을 포함한 전체
   hybrid 시스템의 안정성까지 증명된 것은 아니다.

## 3. 왜 현재 운용 설정은 certificate 대상이 아닌가

현재 운용 설정에는 다음 요소가 있다.

- `collision_policy=lula_canonical`
- `joint_limit_policy=lula_canonical`
- `root_solve_offset=0.001`
- 상태 의존 Target/Axis metric boost
- joint velocity cap leaf
- root solve 이후 acceleration/command guard

이 요소들은 유용한 운용 기능이지만, 현재 구현된 structured-GDS certificate의
가정 밖에 있다. 따라서 운용 로그가 안정적으로 보인다는 사실과 논문 정리에 따른
formal stability는 구분해야 한다.

증명용 설정은 다음과 같이 제한했다.

- Collision과 JointLimit을 `paper_gds` policy로 사용
- Target/Axis metric을 상수로 고정
- joint velocity cap leaf 비활성화
- `root_solve_offset=0`
- Escape energy tank 활성화
- 연속시간 모델을 보기 위해 acceleration clamp를 사실상 비활성화

## 4. 자동 단위 및 구조 검증

추가한 전체 그래프 검증은 다음을 확인한다.

- 운용 설정과 증명용 설정을 certificate에서 혼동하지 않음
- 활성 Escape를 포함한 전체 proof graph의 closure 조건
- 40개 결정론적 random interior state에서 metric의 유한성, 대칭성, 양의 정부호성
- root natural-form residual
- 운용 graph의 유한성 및 certificate 거부
- 장애물 거리 안전집합의 경계 반례

결과:

- 일반 CTest: 9/9 targets PASS
- gtest: 40 cases PASS
- Tangent Escape custom integration: 2 scenarios PASS
- loopback socket test: sandbox 밖에서 1/1 PASS
- Python 진단 스크립트 문법 검사 PASS

관련 테스트:

- `test/test_full_rmp_properties.cpp`
- `test/test_tangent_escape_canonical.cpp`

## 5. 증명용 전체 폐루프 결과

입력 로그:

`/home/song/ros2_ws/log/rmpflow_trace/rmpflow_trace_20260723_183020.csv`

시각화:

`/home/song/ros2_ws/log/rmpflow_trace/full_verify_20260723/proof_full_20260723.png`

주요 결과:

- 1514 rows, 21.999869 s
- structural GDS assumptions: 100%
- static environment assumption: 89.76%
- external RMP empty: 100%
- energy guard valid: 100%
- conditional nonincrease: 89.76%
- energy-bound violation: 0
- tank identity residual: 최대 `7.67e-13`
- root solve power residual: 최대 `2.60e-9`
- clamp power: 0
- Escape ON/OFF 동일 상태 `delta qdd`: 최대 `1.08018 rad/s^2`
- Escape ON/OFF control-point acceleration 차이: 최대 `0.485143 m/s^2`

조건부 비증가 비율이 89.76%인 이유는 실패가 아니라, 나머지 row에서 goal을
움직여 environment-static 가정이 성립하지 않았기 때문이다. 가정이 성립한
row에서는 위반이 없었다.

이 실행은 증명 구조를 보기 위해 acceleration clamp를 제거했으므로 raw joint
acceleration이 크게 나타난다. 실제 로봇용 운용 profile로 사용하면 안 된다.

## 6. 현재 운용 설정의 fake 폐루프 결과

입력 로그:

`/home/song/ros2_ws/log/rmpflow_trace/full_verify_20260723/operational_full_gen_20260723_183132.csv`

시각화:

`/home/song/ros2_ws/log/rmpflow_trace/full_verify_20260723/operational_full_20260723.png`

주요 결과:

- 896 rows, 15.464 s
- Escape active: 81 rows, 7 intervals, 1.289 s
- Escape 동일 상태 `delta qdd`: 최대 `0.0448911 rad/s^2`
- Escape 동일 상태 TCP acceleration 차이: 최대 `0.0182287 m/s^2`
- Escape 동일 상태 control-point acceleration 차이: 최대 `0.0126646 m/s^2`
- joint-acceleration 방향 변화: 중앙값 `0.0554 deg`, 최대 `7.161 deg`
- control-point acceleration 방향 변화: 중앙값 `0.04977 deg`, 최대 `8.835 deg`
- `|normal dot tangent|`: 최대 `0.0234945`
- 외부 장애물 최소 clearance: `0.0100433 m`, 음수 row 0
- geometry 최소 clearance: `0.0100004 m`, 음수 row 0
- joint buffered margin: 최소 `0.178842 rad`, 위반 row 0

Escape active 중 acceleration saturation은 3 rows였지만, 같은 상태에서 Escape가
추가한 최대 `delta qdd`는 `0.0449 rad/s^2`였다. 이 로그의 포화는 큰 Escape kick
때문이 아니라 이미 큰 base RMP 조합이 제한에 도달한 경우다.

이 결과는 해당 fake trajectory에서의 수치 회귀 통과이며, 모든 초기조건과 모든
장애물에 대한 안전성 증명은 아니다.

## 7. Hybrid handoff 검증

`risk_damped` supervisor에 대해 다음 순서를 결정론적으로 만들었다.

1. 기존 `(control point, tangent)` branch의 `lambda` 감소
2. root에 추가되는 Escape metric과 force가 모두 0인 cycle에서 branch 교체
3. 새 branch의 `lambda` 증가

테스트와 새 generation diagnostic에서 확인된 결과:

- 실제 generation change: 1
- 같은 generation 내부의 비정상 tangent mutation: 0
- zero-effect handoff violation: 0

이는 전환 순간의 force/metric jump를 막는 구조 검증이다. 최소 dwell time,
Zeno 배제, 공통 Lyapunov 함수 또는 mode별 energy jump 조건까지 포함하는
전체 hybrid stability 증명은 아니다.

## 8. 불변성 판정

### 장애물 거리

안전집합을 `h(q) = d(q) - d_safe >= 0`으로 두면, 경계 `h=0`에서 장애물 쪽
속도 `hdot<0`를 가진 2차 시스템은 유한한 acceleration만으로 즉시 집합 밖으로
나갈 수 있다. 자동 테스트가 이 상태에서 다음 step의 clearance가 음수가 되는
반례를 확인했다.

따라서 현재 smooth finite Collision RMP는 충돌 회피 성능을 제공하지만,
거리 안전집합의 formal forward invariance를 보장하지 않는다. 이를 hard
보장하려면 higher-order barrier, braking-safe set 또는 동등한 제약 계층이
필요하다.

### 관절 제한

`paper_gds` joint-limit task는 open boundary에서 발산하는 potential을 사용한다.
정확한 연속시간 GDS, 유한 초기 에너지, 정확한 integration 가정에서는 open
joint domain 유지가 조건부로 성립한다.

현재 운용 설정은 대신 다음 step을 예측하는 discrete command guard를 사용한다.
fresh fake run에서는 buffered margin 위반이 0이었지만, 이는 RMPflow closure
정리에 의한 formal invariance가 아니라 구현된 guard의 경험적 검증이다.

## 9. 이번 검증에서 바뀐 코드

제어 gain, leaf 식, root resolve 또는 command 출력은 변경하지 않았다.

변경 내용은 검증과 관측성에 한정된다.

- 전체 RMP property test 추가
- risk-damped handoff integration test 강화
- handoff generation을 debug slot과 CSV에 기록
- closed-loop checker가 실제 generation 변화와 logger가 놓친 cycle을 구분

## 10. 종합 결론

현재 전체 제어기는 fake 폐루프에서 수치적으로 정상 동작하며, Escape leaf가
base RMP 조합에 유한하고 tangent한 영향을 주고 zero-effect handoff를 수행한다.

논문 수준에서 현재 주장 가능한 범위는 다음이다.

> 증명용 고정 mode 내부에서 각 leaf와 전체 pullback/resolve graph는
> structured-GDS 조건을 만족하며, 환경이 고정된 구간에서 energy-tank를 포함한
> 조건부 Lyapunov 비증가가 수치적으로 검증되었다.

현재 주장할 수 없는 범위는 다음이다.

> 운용 profile 전체의 structured-GDS stability, memory/switching을 포함한
> global hybrid stability, 장애물 거리 안전집합의 formal forward invariance.

