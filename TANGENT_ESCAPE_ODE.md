# Escape RMP v2 — `tangent_ode`

## Runtime selection

```yaml
enable_tangent_escape_rmp: true
tangent_escape_rmp_acceleration_model: "tangent_ode"
```

`canonical_velocity`, `risk_damped`, and `tangent_plane` remain compiled and
selectable. v2 owns a separate state object, so switching models never carries
anything across.

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

Since \(\hat{\mathbf t}\perp\hat{\mathbf o}\), the two leaves own orthogonal
one-dimensional subspaces and do not compete.

## Leaf

The leaf is designed in **canonical** `(a, M)` form. `f = M a` is only the
representation the pullback transports.

```
J_e   = t^T Jp                                    (1 x 6)
ydot  = J_e qdot

a_e   = a_max e^{-d/sigma_a} e^{-s/s0} alpha_free  -  b_e ydot     [m/s^2]
m     = m_max e^{-d/sigma_m} alpha_free

f_e   = m a_e                                     (representation change)
f_q   = J_e^T (f_e - m Jdot_e qdot)
M_q   = m J_e^T J_e                                (PSD, rank <= 1)
```

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
      + w_r ( eps * sum_axial w_j  -  sum_inplane w_j u_j cos(phi_j) )
b = w_g (e2 . dg) + w_v (e2 . vhat)
      + w_r (                      -  sum_inplane w_j u_j sin(phi_j) )
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
