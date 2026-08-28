# TG-RMP 안전성 구조 검증 산출물

기준일은 2026-08-21이고, 검증 대상은 base commit `0bc03aa`에 검사 전부터 존재한 작업 트리 변경을 포함한 현재 소스입니다. 기존 수정 파일은 변경하지 않았습니다.

- [PDF 보고서](./TG_RMP_Safety_Verification_Report_2026-08-21.pdf): 23쪽 A4 고정 레이아웃. 한글과 수식이 고정되어 있어 열람·인쇄용으로 권장합니다.
- [HTML 원본](./TG_RMP_Safety_Verification_Report_2026-08-21.html): 편집 가능한 원본. 수식은 브라우저 native MathML입니다.

## 종합 판정

**안전 승인 보류 · 사람 근접 운전 NO-GO**

TG-RMP의 접선 기하와 rank-one PSD metric 구조는 상당 부분 타당합니다. 그러나 현재 구성만으로 충돌 안전집합의 forward invariance를 보장할 수 없고, sensor freshness/coverage, solver failure 시 정지, 독립 watchdog, hard safety filter에 P0 gap이 있습니다.

신선한 빌드의 CTest 12개 target은 소켓 환경 오탐을 별도 해소한 뒤 실질적으로 11개 통과, 1개 실패입니다. 남은 실패는 CasADi TG가 active로 보고되지만 root metric 기여가 0인 회귀입니다. Production stability analyzer도 미선언 parameter 조회로 국소 안정성 수치 판정 결과 생성에 실패했습니다.

## 보고서의 핵심 제안

1. 원본 timestamp/sequence 기반 sensor·state·TF freshness supervisor와 latched stop
2. 20채널 topic–frame–control-point 1:1 정합 및 blind-zone 검증
3. 실측 관절속도 경로의 강제·검증
4. solver/deadline/QP failure 시 bounded deceleration 또는 protective stop
5. TG-RMP 뒤의 whole-body robust HOCBF-QP 또는 동등한 제동 기반 safety shield
6. TG/ROS 2와 독립적인 watchdog·E-stop·보호정지
7. SiL(Software/Simulation-in-the-Loop) → HIL → 사람 없는 저속 더미 → 제한 실기 순서의 단계별 검증

이 산출물은 연구·개발 단계의 verification report이며 ISO 적합성 평가나 안전 인증을 대신하지 않습니다.
