# Tangent-Plane Tangent Escape RMP

## Runtime selection

```yaml
enable_tangent_escape_rmp: true
tangent_escape_rmp_acceleration_model: "tangent_plane"
```

`canonical_velocity` and `risk_damped` remain compiled and selectable. Each model
owns a separate hybrid state, so switching models never carries a latched pair,
tangent, or failure memory across the boundary.

This model keeps the score-based direction selection, pair ownership, blockage
gating, failure memory, handoff phases, and hard timeout of `risk_damped`. It
changes exactly two things: the drive law and the leaf metric.

## Role separation

\[
d\ \longrightarrow\ \text{how strongly Escape acts},
\qquad
S\ \longrightarrow\ \text{which direction it acts in}.
\]

## Distance gates

With clearance \(d\) measured exactly as Collision measures it,

\[
\alpha(d;d_{\rm near},d_{\rm on})
=H\!\left(\operatorname{clip}\frac{d_{\rm on}-d}{d_{\rm on}-d_{\rm near}},0,1\right),
\qquad H(r)=10r^3-15r^4+6r^5 .
\]

Two gates are derived from it:

\[
\alpha_a=\alpha(d;d^{a}_{\rm near},d^{a}_{\rm on}),
\qquad
\alpha_M=\min\!\left(\alpha(d;d^{M}_{\rm near},d^{M}_{\rm on}),\ \alpha_a\right).
\]

The clamp \(\alpha_M\le\alpha_a\) is enforced in code and is not optional. A metric
that switches on before the push would ask the leaf for **zero tangent-plane
acceleration at full weight**, which brakes every other leaf inside the plane
instead of escaping. The rank-two metric below makes that failure cover a
two-dimensional subspace rather than a single axis, so the clamp matters more here
than it would for a rank-one leaf.

Non-positive distance overrides inherit one level up: the push gate inherits
`safe_distance` / `influence_distance`, and the metric gate inherits the push gate.
With the inherited defaults \(\alpha_M=\alpha_a=\alpha_d\), which is numerically the
same gate `risk_damped` uses.

## Drive law

Let \(t\) be the applied tangent (below), \(v_t=t^\top J_i\dot q\), and \(b_t\) the
tangent damping gain. Then

\[
a_{\rm push}=\min\!\left(a_{\max}\,\alpha_a(d),\ b_tv_{t,\max}\right),
\qquad
a_y=a_{\max}\tanh\!\left(\frac{a_{\rm push}-b_tv_t}{a_{\max}}\right).
\]

The \(b_tv_{t,\max}\) cap is the same one `risk_damped` applies. Without it the
steady state below would sit at \(a_{\max}/b_t\) and silently ignore `max_speed`
whenever \(b_t\neq a_{\max}/v_{t,\max}\) — which the shipped `config/params.yaml`
does (`max_accel=4.0`, `b_t=10.0`, `max_speed=0.20` gives 0.4 m/s uncapped).

There is no closing-rate term, no stuck-acceleration floor, and no clearance-rate
low-pass filter. `tanh` bounds the result symmetrically in \([-a_{\max},a_{\max}]\),
so \(-b_tv_t\) stays a real damper rather than a one-sided cutoff: an overspeeding
tangent is decelerated instead of merely left alone.

Holding \(d\) fixed, \(a_y=0\) at

\[
v_t=\frac{a_{\rm push}(d)}{b_t},
\]

so \(a_{\max}\) and \(b_t\) are not independent. Setting

\[
b_t=\frac{a_{\max}}{v_{t,\max}}
\]

makes the cap inactive and the two knobs consistent. The cap protects the
configured speed either way, but matching them keeps `max_accel` meaningful.

## Applied tangent

The score-selected tangent \(t^\ast\) is still latched in the world frame, but it is
re-projected onto the current tangent plane every cycle:

\[
P_T=I-\hat o\hat o^\top,
\qquad
t=\frac{P_Tt^\ast}{\lVert P_Tt^\ast\rVert}.
\]

Consequences:

- A latched world tangent can never leak a normal component into the metric as
  \(\hat o\) rotates. The `normal_tolerance` check becomes a *reselect trigger*
  rather than a correctness requirement.
- If \(\lVert P_Tt^\ast\rVert\) collapses, the leaf contributes exactly zero metric
  and zero force for that cycle; the reselect path still runs.
- Diagnostics publish the applied tangent \(t\), not the latched \(t^\ast\).

## Leaf metric

\[
M_{\rm leaf}=m\left[(1-\gamma)\,tt^\top+\gamma\,P_T\right],
\qquad
m=m_{\max}\,\lambda\,\alpha_b\,\alpha_M .
\]

Both terms are PSD and both coefficients are nonnegative for \(\gamma\in[0,1]\), so
\(M_{\rm leaf}\) is PSD for every admissible input. Because \(t\) is re-projected into
the range of \(P_T\), the rank is at most two — it never acquires a normal-direction
component that would fight Collision.

The pullback uses the full three-dimensional task map, so the curvature term is the
control-point curvature vector rather than its scalar projection:

\[
\Delta M=J_i^\top M_{\rm leaf}J_i,
\qquad
\Delta f=J_i^\top M_{\rm leaf}\left(a_yt-c_i\right),
\qquad c_i=\dot J_i\dot q .
\]

`alpha_b` is retained in \(m\) so that entry still requires a blocked nominal intent;
\(\alpha_M\) replaces only the distance factor of blockage, which is what makes the
metric gate independently tunable.

### The \(\gamma\) knob

\(\gamma\) is the in-plane isotropy and the intended sweep axis:

| \(\gamma\) | metric | direction change perturbs the metric by |
|---|---|---|
| 0 | \(m\,tt^\top\), rank 1 | the full rank-one axis swap |
| \(\gamma\) | blend | \((1-\gamma)m\) |
| 1 | \(m\,P_T\), rank 2 | nothing |

At \(\gamma=1\) the selected direction has left the metric entirely: a score-driven
direction change is a rotation of a bounded force, not a discontinuity of \(\Delta M\).
It is still not a fully continuous metric — \(\hat o\) itself jumps when the owning
control-point/obstacle pair changes, which is why the \(\lambda\) handoff ramp is
retained for pair replacement.

At \(\gamma=0\) the metric is rank one along the *re-projected* tangent. That is the
rank-one structure of the other two models up to the re-projection, which is itself a
correctness fix rather than a behavioral knob.

`metric_scalar` needs re-tuning across the sweep: at \(\gamma=1\) the same \(m\)
weights a two-dimensional subspace instead of one axis.

## Injected power

Splitting \(a_y\) into the push and the damper gives a leaf that is an ordinary GDS
in everything except one term:

\[
\underbrace{(G=mP_T,\ B=b_tm\,tt^\top,\ \Phi=0)}_{\text{GDS}}
\quad+\quad
\underbrace{F_{\rm ext}=m\,a_{\rm push}\,t}_{\text{not a potential gradient}} .
\]

Since \(M_{\rm leaf}t=mt\), the power the non-GDS term injects into the base storage is

\[
\boxed{P_{\rm inj}=m\,a_{\rm push}\,v_t}
\]

which needs no solve result, unlike the exact rank-one interconnection power
\(P_e=m v_t(a_y-\ddot y)\). It is published in the diagnostics and is the quantity to
bound if the energy tank is applied to this model.

This does **not** make the leaf a GDS. \(a_{\rm push}(d)\,t\) has

\[
\nabla\times\left(a_{\rm push}(d)\,t\right)
=a_{\rm push}'(d)\,(\nabla d\times t)\neq0,
\]

so no potential exists for it. Replacing \(a_{\rm push}(d)\) with
\(a_{\rm push}(d)\,\sigma'(y)\) for a tangent progress coordinate \(y\) would close that
gap; it is not implemented here.

## What still stops an episode

`alpha_a(d) -> 0` is the only distance-driven release. Sliding along a wall holds
\(d\) roughly constant, so the hard episode timeout
(`tangent_escape_rmp_maximum_active_duration`) is retained and is what bounds a
persistent-obstacle episode. Goal convergence is not claimed while the obstacle
persists.

## Parameters

| Parameter | Symbol | Notes |
|---|---|---|
| `tangent_escape_rmp_metric_isotropy` | \(\gamma\) | default `1.0`, clamped to \([0,1]\) |
| `tangent_escape_rmp_push_safe_distance` | \(d^{a}_{\rm near}\) | `<= 0` inherits `safe_distance` |
| `tangent_escape_rmp_push_influence_distance` | \(d^{a}_{\rm on}\) | `<= 0` inherits `influence_distance` |
| `tangent_escape_rmp_metric_safe_distance` | \(d^{M}_{\rm near}\) | `<= 0` inherits the push gate |
| `tangent_escape_rmp_metric_influence_distance` | \(d^{M}_{\rm on}\) | `<= 0` inherits the push gate |
| `tangent_escape_rmp_max_accel` | \(a_{\max}\) | push magnitude at \(\alpha_a=1\) |
| `tangent_escape_rmp_risk_tangent_damping_gain` | \(b_t\) | set to \(a_{\max}/v_{t,\max}\) |
| `tangent_escape_rmp_metric_scalar` | \(m_{\max}\) | re-tune when sweeping \(\gamma\) |

`risk_distance_*`, `risk_approach_*`, `risk_velocity_gate_scale`,
`risk_clearance_rate_filter_time_constant`, and `risk_stuck_*` are **not read** by
this model.

## Diagnostics

`schema_id = 8`, fixed prefix of 76 values, then the same 17-value candidate records.
Slots 0–67 keep their `risk_damped` meaning; slot 61 now carries the analytic
clearance rate.

| Slot | Field |
|---:|---|
| 68 | \(\alpha_a\) |
| 69 | \(\alpha_M\) |
| 70 | \(a_{\rm push}\) |
| 71 | \(b_tv_t\) |
| 72 | analytic \(\dot d=-\hat o^\top J_i\dot q\) |
| 73 | \(P_{\rm inj}\), in RMP metric units (the metric is a weight, not a mass in kg) |
| 74 | \(\gamma\) |
| 75 | \(\lvert t^{\ast\top}\hat o\rvert\) before re-projection |

Slot 72 is the robot's own contribution to the clearance rate, taken from the
Jacobian rather than by differencing a filtered distance. It carries no filter state
and no phase lag, and obstacle self-motion is not included. It is published for
diagnosis only; the drive law does not read it.

`scripts/rmpflow_trace_logger.py` decodes schema 8 and emits the eight columns above.

## Running the fake-proximity sweep

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

python3 src/RMP_Proximity-Sensor/scripts/run_tangent_plane_gamma_sweep.py \
  --gammas 0.0 0.5 1.0 --with-risk-damped-baseline --duration 45
```

Each configuration is one headless `ros2 launch` run, SIGINT'd after `--duration`
so the trace logger closes its CSV, then summarised into one table plus
`sweep_summary.json`.

The runner passes `start_fake_proximity:=true`. The launch file defaults that to
`false` because the fake publisher is normally started from a second terminal; an
unattended sweep has none, and without it every run records a scenario with no
obstacle and Escape never activates. Pass `--no-start-fake-proximity` if you are
driving the publisher yourself.

`controller_goal_*` is only populated when something publishes `/goal_pose`. This
launch takes the goal as a node parameter, so the runner falls back to its own
`--goal-x/y/z` and says so in the notes. `goal_err_start` therefore depends on where
in the motion the logger happened to start and is **not** comparable across runs;
`goal_err_end` is.

To re-summarise CSVs without re-running:

```bash
python3 src/RMP_Proximity-Sensor/scripts/run_tangent_plane_gamma_sweep.py \
  --summarize-only a=path/to/a.csv b=path/to/b.csv
```

## Experiment factorization

The two changes are separable through the model selection:

| Comparison | Isolates |
|---|---|
| `risk_damped` → `tangent_plane`, \(\gamma=0\) | drive law only |
| `tangent_plane` \(\gamma=0\) → \(\gamma=1\) | leaf metric only |
| \(\gamma\) swept over \([0,1]\) | metric change as a continuous curve |

`risk_damped` does not read `metric_isotropy`, so a metric-only change on top of the
old drive law is not currently reachable.

## Test coverage

`test/test_tangent_escape_canonical.cpp :: run_tangent_plane_test` asserts, over the
blocking scenario and both \(\gamma\) endpoints:

- schema id 8 and the 76-value prefix;
- \(\Delta M\) symmetric, PSD, rank \(\le1\) at \(\gamma=0\) and reaching 2 at \(\gamma=1\);
- \(\Delta M\) annihilates a joint velocity whose control-point motion is purely along
  \(\hat o\);
- \(\alpha_M\le\alpha_a\) with a deliberately wider metric distance gate;
- \(a_{\rm push}=a_{\max}\alpha_a\) and \(\lvert a_y\rvert\le a_{\max}\);
- \(P_{\rm inj}=m\,a_{\rm push}\,v_t\);
- the applied tangent is unit and orthogonal to \(\hat o\);
- the published clearance rate equals \(-\hat o^\top J_i\dot q\) under nonzero \(\dot q\);
- an unknown `acceleration_model` throws instead of falling back.

## Not claimed

- Structured-GDS closure for this leaf.
- Collision-safe-set forward invariance.
- Goal convergence while a blocking obstacle persists.
