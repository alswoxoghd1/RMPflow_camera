# Interactive Diffusion Policy — Low-Level Execution Layer

**Section IV의 low-level controller 부분.** 초안 구조 대응: 4.1 intro → 4.2 framework
(RMP+근접센서 / DP / OOD→LPB).

> **Drafting note (논문 본문 아님).** 식과 수치는 "없으면 설명이 안 되는 것"만 남겼다.
> 나머지 값은 `config/params.yaml`이 원본. Tangent-escape(TG-RMP)는 본 논문 범위 제외.
> `→ 논문 반영` 메모는 최종 원고에서 삭제.

---

## A. Why avoidance is not put inside the policy

An imitation-learned policy is only as safe as its demonstrations. A person's hand entering
between the gripper and the workpiece is precisely the event that demonstrations do not
contain, and it is also the event that must be handled within milliseconds. Three common
remedies all have the same weakness:

- **collecting avoidance demonstrations** makes safety a function of data coverage;
- **conditioning the policy on an obstacle signal** makes safety a function of training quality;
- **adding an avoidance guidance term to the sampler** couples the safety response to the
  denoising schedule, so its latency and magnitude are inherited from the generative model
  rather than chosen.

In every case safety becomes probabilistic and is re-litigated each time the policy is
retrained. We therefore separate the two questions by **role**, in the same spirit as
compliant-execution systems that place the reactive term below the policy: the diffusion
policy decides **where to go**, and a geometric layer running at the control rate decides
**how to get there right now, given what is near the robot at this instant**.

The layer is an RMPflow tree. Its task input is the policy's target pose; its safety input is
the robot-mounted proximity sensing of Section III. Both enter the same solve.

→ 논문 반영: 이 절이 4.1. 3–4문장으로 압축 가능. "별도 guidance를 두지 않는다"의 근거가
핵심.

---

## B. Framework: one solve, two inputs

At every control cycle the layer receives

- \(p_g\): the end-effector target published by the diffusion policy, at policy rate, and
- \(\{d_k\}\): the calibrated proximity distances of the robot-mounted sensors, at sensor rate,

and solves, at **200 Hz**,

\[
A=\underbrace{A_{\mathrm{tar}}}_{\text{go to } p_g}+\underbrace{A_{\mathrm{col}}}_{\text{stay off } \{d_k\}}+A_{\mathrm{cons}},
\qquad
f=f_{\mathrm{tar}}+f_{\mathrm{col}}+f_{\mathrm{cons}},
\qquad
\ddot q^{*}=(A+\lambda I)^{-1}f ,
\]

where each term is the pullback \(A_i=J_i^{\top}M_iJ_i\),
\(f_i=J_i^{\top}M_i(a_i-\dot J_i\dot q)\) of a task-space policy \((a_i,M_i)\) through its own
task map. The resulting joint acceleration is integrated and guarded into a joint-velocity
command.

**The two behaviours are not sequenced — they are superposed.** This is the property the
whole framework rests on, and it deserves to be stated explicitly in the paper:

1. **The target term is never switched off.** While a person is being avoided, the pull
   toward \(p_g\) is still in the solve; what changes is the relative weight of the metrics.
   The arm is therefore *deflected* around the obstacle while still making whatever progress
   toward the target the obstacle permits.
2. **Recovery needs no separate mechanism.** The collision metric has compact support, so it
   vanishes as the person withdraws, and the same solve returns to pure target tracking on
   the next cycle. There is no "resume" command, no re-planning, and no state machine.
3. **The two run at different rates by design.** The policy publishes targets at policy rate;
   avoidance runs at 200 Hz. Between two policy outputs the robot is still fully reactive,
   which is what makes the layer's safety independent of the policy's inference latency.

→ 논문 반영: 이 절이 4.2의 뼈대. 위 식 한 줄 + 세 성질 + Fig.(framework). 회피와 추종이
동시에 일어난다는 점을 그림에서 화살표 두 개가 한 solve로 합쳐지는 형태로 보여줄 것.

---

## C. Diffusion target → task RMP

The policy's pose target becomes a position attractor on the TCP and orientation attractors
on its axes. The position term is **direction-normalized**,

\[
a_{\mathrm{tar}}=k_p\frac{p_g-p}{\lVert p_g-p\rVert+\epsilon}-k_d\dot p ,
\]

so the commanded acceleration saturates at \(k_p\) no matter how far the target is. This
matters specifically because the target comes from a generative model: an occasional
inconsistent or distant sample cannot inject an unbounded force into the solve, it can only
change the direction the arm is pulled.

The term's metric is shaped by distance to the goal, and this shaping is what makes
simultaneous avoidance possible:

- **Far from the goal the metric is anisotropic** — strong along the goal direction, weak
  across it. Lateral deflection therefore costs the task term almost nothing, so the
  collision term can route the arm around a person without fighting the attractor.
- **Near the goal it becomes isotropic and stiff**, so the compliance that was useful during
  transit is not paid for in final placement accuracy.

In other words, the policy's authority is concentrated on *making progress*, and the
directions orthogonal to progress are left available to the safety term. Avoidance is a
reshaping of the same motion, not an override of it.

→ 논문 반영: 식 1개 + metric anisotropy 문단. 이게 "why simultaneous"의 절반.

---

## D. Proximity sensing → collision RMP

**Why this modality.** The dangerous case in a shared workspace is a hand or forearm entering
the volume immediately around the robot's own surface — behind a link, beside the gripper, in
the camera's shadow. That is exactly where an external vision system is least reliable and
where a surface-mounted sensor is most reliable. The sensing itself is described in Section
III; what matters here is that the layer's safety input is **local to the robot's surface and
independent of external line of sight**, so avoidance does not degrade when the scene
occludes the camera.

**From range to obstacle.** Each valid range is placed along its sensor's own axis in the base
frame, and a sphere is seated so that its near surface coincides with the measured surface
point:

\[
p_{o,k}=p_{s,k}+n_k(d_k+r_o).
\]

Obstacles are created and destroyed each cycle rather than accumulated into a map, which is
the right model for a person who is moving. Signed clearance for a control point of radius
\(r_c\) is \(s=\lVert p_c-p_o\rVert-r_c-r_o\), and because each control point sits at its own
sensor origin this reduces to \(s=d_k-r_c\): the control-point radius is a **keep-out shell
around every sensor**, and all reported clearances can be read directly against measured
range.

**The collision term.** On the positive clearance,

\[
a_{\mathrm{col}}=k_re^{-s/\sigma}-\gamma(\dot s)\frac{k_b\dot s}{s/\sigma_b+\epsilon},
\qquad
m_{\mathrm{col}}=\gamma(\dot s)\,g(s)\,\frac{\mu}{s/\sigma_m+\epsilon_m},
\]

with two properties that carry the safety argument:

- **Velocity-directional gating.** \(\gamma(\dot s)\) closes toward zero when clearance is
  *growing*. A control point receding from a person is barely constrained; one closing on
  them is heavily constrained. The arm can therefore leave a near-contact configuration
  promptly instead of being held by its own barrier.
- **Compact support.** \(g(s)\) drives the metric to exactly zero beyond an influence radius.
  This is what makes property B.2 true: when the person leaves, the safety term does not fade
  slowly or need resetting — it is simply absent, and the target term is again alone.

**Sensing the human's own approach.** A distance-only sensor cannot by itself tell an
approaching robot from an approaching person. We decompose the clearance rate: the part
explained by the robot's own motion is computed in closed form from the control-point
velocity, and the residual — low-passed and dead-banded — is attributed to the obstacle. For a
static obstacle the residual is noise; for a person walking in, it is their approach velocity.
**This is the only channel through which the controller can react to a human approaching a
stationary robot**, and it is consumed by the damping term above.

**Self-detection.** A link-mounted sensor also sees the robot itself. Following prior work on
joint-configuration-based compensation, readings dominated by self-detection in known joint
ranges are suppressed for those configurations, so only externally caused distance changes
reach the safety input.

→ 논문 반영: occlusion 강인성 주장(contribution 2)의 근거가 이 절 전체. sphere 식 +
\(s=d-r_c\) + gating/compact support + closing-rate residual.

---

## E. Constraint terms

Joint limits, a joint-velocity cap, a weak posture bias resolving the null space, and joint
damping complete the tree. The joint-limit and velocity-cap metrics diverge as their bounds
are approached, so those limits are enforced by the geometry of the solve rather than by
clipping the output. One sentence each in the paper.

---

## F. Command generation and guards

The solved acceleration is limited in magnitude while preserving its direction — so
saturation never distorts *where* the arm is being sent — integrated into a joint-velocity
command, and passed through a last-resort envelope (position clamping, velocity reduction
near a limit, velocity saturation, a predictive joint-limit guard, a minimum-height check)
before transmission to the manipulator over its direct velocity interface.

The guards sit outside the RMP tree by design: the tree is responsible for behaviour, the
guards for the envelope.

→ 논문 반영: 2문장.

---

## G. What this layer gives the policy — and what it costs

**Gives.** The policy can be trained purely on task demonstrations and still run beside a
person: physical safety comes from sensing and geometry, not from data coverage. Avoidance
never terminates an episode, and when the person withdraws the arm resumes tracking the
policy's target on the very next cycle with no explicit recovery step.

**Costs.** The trajectory taken *during* avoidance is not one the policy has seen. Its
observations while being deflected — end-effector pose, and whatever visual state the task
uses — lie outside the demonstration distribution, so after the obstacle clears, the policy
may be conditioned on a state it cannot act on reliably even though the robot is physically
fine. **The reactive layer solves the safety problem and, in doing so, creates a distribution
problem.**

This is exactly the gap the next component closes: the learned progress/behaviour module
(LPB) re-anchors the policy after a deflection so the task continues from wherever avoidance
left the arm, and the recovery-aware data collection of Section IV-C supplies the data it
needs. The two are complementary by construction — the RMP layer guarantees the robot stays
safe and keeps the target alive, and LPB guarantees the task survives the detour.

→ 논문 반영: 4.2.3으로 이어지는 다리. 이 절을 짧게 쓰되 반드시 넣을 것 — RMP 챕터의 한계가
LPB의 존재 이유가 되므로 논문 흐름이 여기서 완결된다.

---

## H. Implementation and scope of the claims

Runtime control uses an explicit Pinocchio/Eigen accumulation of the root solve at 200 Hz
with the tree pre-compiled in topological order; a CasADi implementation of the same tree is
retained as a reference and compared against it on identical states.

**We claim:** every term is verified finite with positive-semidefinite metric over the
operating envelope; the root solve is well-conditioned under the scaled regularizer; and the
joint, velocity, and workspace envelope is maintained empirically across the reported runs.

**We do not claim:** a formal stability certificate for the operating configuration — it
deliberately includes state-dependent metric shaping, the velocity-cap term, the root
regularizer, and post-solve guards, all outside the assumptions of the RMPflow theorem — nor
forward invariance of the obstacle safe set. Physical safety rests on the collision term plus
the guards, demonstrated experimentally. A restricted, strictly structured-GDS configuration
of the collision and joint-limit terms exists in the implementation and passes automated
closure and Lyapunov checks; it is a verification artifact and is not merged with the
operating configuration into a single guarantee.

We also note the layer's own limitation: a purely reactive tree can stall where attraction
and repulsion cancel. In the reported configuration this is handled at the framework level
rather than inside the controller.

→ 논문 반영: limitation 절로. 마지막 문단은 정직성 확보용이자 LPB 연결 보강.

---

## I. Key settings for the experimental section

| Quantity | Value |
|---|---|
| Control rate | 200 Hz |
| Keep-out shell per sensor \(r_c\) | 0.05 m |
| Collision influence radius | 0.3 m clearance |
| Sensor trigger range (wrist / forearm ring) | 0.5 m / 0.3 m |
| Joint velocity / acceleration limit | 1.0 rad/s, 10 rad/s² |
| Enabled sensing stations | wrist group + distal forearm ring |

**Metrics.** Task success rate, safety success rate, minimum clearance
\(\min_t\min_k d_k\), time to completion, command smoothness — reported over the
vision vs. vision+proximity × intervention vs. no-intervention grid of Section V.

---

## J. Figure caption drafts

**Fig. (framework).** The diffusion policy publishes an end-effector target; robot-mounted
proximity sensors publish distances; both enter a single RMPflow root solve at 200 Hz that
produces one joint-velocity command. Target tracking and obstacle avoidance are superposed in
this solve rather than arbitrated: the target term stays active while a person is avoided,
and the collision term vanishes as the person withdraws.

**Fig. (sensing).** Sensor placement and obstacle construction. Each enabled sensor
contributes a control point and, when its range falls below the trigger distance, one
obstacle sphere seated so its near surface coincides with the measured surface point; the
signed clearance is the measured range minus the keep-out shell.

**Fig. (behaviour).** A person reaching into the workspace during a pick-and-place episode:
TCP path, minimum clearance, and distance-to-target over time, showing the arm deflecting
around the hand while target distance continues to decrease, and resuming direct tracking as
the hand withdraws — with no mode switch and no replanning.

---

## References to reuse

- Capacitive–ToF proximity sensing and self-curve-fitted calibration: `\cite{yim2024emtof, kang2025adaptive}`
- Manipulator-mounted proximity sensing and self-detection compensation: `\cite{moon2021dual}`
- RMPflow / RMP2: `\cite{cheng2018rmpflow, li2021rmp2}`
