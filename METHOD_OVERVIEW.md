# Strike-EA — Method Overview

Author: auto-generated from codebase (March 2026)

---

## Table of Contents

1. [Agent-Based Modeling Architecture](#1-agent-based-modeling-architecture)
2. [MAPPO Algorithm and Formulas](#2-mappo-algorithm-and-formulas)
3. [Actor Network](#3-actor-network)
4. [Critic Network](#4-critic-network)
5. [Actor–Critic Interaction Flow](#5-actorcritic-interaction-flow)
6. [Reward Design](#6-reward-design)
7. [Critical Feedback and Failure Analysis](#7-critical-feedback-and-failure-analysis)

---

## 1. Agent-Based Modeling Architecture

### 1.1 World Representation

The environment is a 2D continuous space with normalized coordinates in `[0.0, 1.0]`.

| Physical quantity | Normalized unit |
|---|---|
| Distance | 1 unit = 1 000 km |
| Time | 1 step = 60 s (1 min) |
| Speed | 0.02 units/step ≈ Mach 1 |
| Radar range | 0.20 units = 200 km |

All tensors are vectorised over a batch dimension `B = num_envs` (default 256) so that all environments run in parallel on the same device.

### 1.2 Entity Types

| Type | Count | Controllable? | Role |
|---|---|---|---|
| **Striker** | `n_strikers` (default 1) | Yes (actor 1) | Destroy targets kinetically |
| **Jammer** | `n_jammers` (default 1) | Yes (actor 2) | Suppress radar electronically |
| **Target** | `n_targets` (default 1) | No (static) | Objective to destroy |
| **Radar** | `n_radars` (default 1) | No (static) | Threat that kills agents |

Total controllable agents: `A = n_strikers + n_jammers` (default 2).

### 1.3 Kinematic Model (Unicycle)

Each agent follows a **unicycle kinematic model** with separate linear and angular dynamics:

**State per agent:** `(x, y, ψ, v, ω)` — position, heading, speed, heading rate.

**Linear velocity update:**
```
v ← clamp( v + a_v × accel_magnitude,  v_min,  v_max )
```

**Heading rate update (turning acceleration):**
```
ω ← clamp( ω + a_ω × h_accel_magnitude,  −ψ̇_max,  +ψ̇_max )
```

**Minimum turn radius constraint:** At low speed, heading rate is further limited:
```
ω_max_actual = min( ψ̇_max,   v / R_min )
```

**Heading and position update:**
```
ψ ← (ψ + ω × dt) mod 2π
x ← x + v × cos(ψ) × dt
y ← y + v × sin(ψ) × dt
```

Positions are clamped to `[low, high]` = `[0, 1]`.

**Kinematic Parameters (from `config.py`)**

| Parameter | Value | Physical meaning |
|---|---|---|
| `v_max` | 0.02 | 20 km/min ≈ Mach 1 |
| `accel_magnitude` | 0.01 | 10 km/min per step |
| `dpsi_max` | 12°/step | Max sustained turn rate |
| `h_accel_magnitude_fraction` | 0.1 | Angular accel = 1.2°/step |
| `min_turn_radius` | 0.01 | 10 km minimum turn radius |
| `v_min` (striker & jammer) | 0.01 | Cannot hover; always moving |

### 1.4 Spawn Logic

- **Agents** spawn at the bottom-centre of the map (`y ≈ 0.05`), spaced 20 km apart, all facing north (`ψ = π/2`), at `v_min`.
- **Radars** spawn randomly in the top half (`y > 0.5`) with a 100 km border margin.
- **Targets** spawn at 90 % of radar range from their assigned radar (`d_spawn ≈ 0.18`) in a random direction.

### 1.5 Observation Space (Ego-Centric)

Each agent receives a **local, ego-centric observation vector** of dimension:

```
obs_dim = 2 + 2×(A−1) + 2×R + 3×T
```

For default config (A=2, R=1, T=1): `obs_dim = 2 + 2 + 2 + 3 = 9`.

**Layout per agent:**

| Section | Dimensions | Content |
|---|---|---|
| Own state | 2 | `[v / v_max,  ω / ψ̇_max]` (normalised speed & heading rate) |
| Other agents | `2 × (A−1)` | `[dist/d_max, Δψ/π]` per other agent (zeroed if outside `R_obs` = 300 km or dead) |
| Radars | `2 × R` | `[dist/d_max, Δψ/π]` per radar |
| Targets | `3 × T` | `[dist/d_max, Δψ/π, alive]` per target |

All angles are **relative to the agent's current heading** and normalised to `[−1, 1]`.  
Distances are normalised by the map diagonal `d_max = √2 ≈ 1.41`.

**Key assumptions of this representation:**
- Agents always see all radars and targets regardless of range (full observability for threats/objectives).
- Allied agents are only visible within `R_obs = 300 km` (limited ally awareness).
- Observations are purely **relative** and **ego-centric** — the agent has no absolute position awareness.

### 1.6 Action Space

Each agent has **2 discrete action dimensions**, each a `Categorical(7)`:

| Index | Value | Effect |
|---|---|---|
| 0 | −1.0 | Maximum decelerate / turn-left hard |
| 1 | −0.5 | Medium decelerate / turn-left medium |
| 2 | −0.1 | Soft decelerate / soft left |
| 3 | 0.0 | Hold current state |
| 4 | +0.1 | Soft accelerate / soft right |
| 5 | +0.5 | Medium accelerate / turn-right medium |
| 6 | +1.0 | Maximum accelerate / turn-right hard |

- **Dim 0** → linear acceleration: `a_v × accel_magnitude (0.01 units/step)`
- **Dim 1** → angular acceleration: `a_ω × h_accel_magnitude (1.2°/step)`

Total action space per agent: `7 × 7 = 49` joint actions.

### 1.7 Scenario Mechanics

**Jamming (automatic, passive):**  
Every step, for each jammer that is alive and within `jam_radius = 0.25` of a radar, all radars covered by at least one jammer have their effective range reduced:
```
radar_eff_range = max(0,  radar_range − delta_range) = max(0, 0.20 − 0.10) = 0.10
```
Jamming is *automatic* — the agent does not need to "press a button". It only needs to navigate to the right position.

**Radar kills (automatic, per step):**  
Any agent (striker or jammer) within `radar_eff_range` of a radar is killed with probability `kill_probability = 1.0` (instant kill).

**Striker engagement (automatic, passive):**  
Any striker within `engage_range = 0.10` of a target *and* with the target inside its `engage_fov = ±30°` forward cone destroys that target immediately.

**Safe jamming zone:**  
`engage_range < dist < jam_radius` → `0.20 < dist < 0.25` (only 50 km wide corridor).

---

## 2. MAPPO Algorithm and Formulas

### 2.1 Paradigm: CTDE

The method uses **Centralised Training, Decentralised Execution (CTDE)**:

- **Execution (deployed):** Each agent acts using only its own local observation → no communication required at inference time.
- **Training:** A centralised critic has access to the full global state → better value estimates than any single agent could compute alone.

### 2.2 Advantage Estimation — GAE

Generalised Advantage Estimation (Schulman et al., 2018) is used to reduce variance in gradient estimates while controlling bias via λ.

**TD residuals:**
```
δ_t = r_t + γ V(s_{t+1}) − V(s_t)
```

**GAE:**
```
Â_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
```

Or equivalently, recursively:
```
Â_t = δ_t + γλ Â_{t+1}
```

**Parameters (from `config.py`):**

| Parameter | Value | Effect |
|---|---|---|
| `gamma` (γ) | 0.99 | Long-horizon discounting |
| `lmbda` (λ) | 0.95 | GAE bias–variance tradeoff |

> ⚠️ **Current setting:** `normalize_advantage = False` (hardcoded in `utils.py`). This means advantages enter the loss with their raw magnitudes, which can be very large, causing unstable gradient steps.

### 2.3 Clipped PPO Objective

The policy is updated to maximise the **clipped surrogate objective** (Schulman et al., 2017):

**Probability ratio:**
```
r_t(θ) = π_θ(a_t | o_t) / π_θ_old(a_t | o_t)
```

**Clipped objective (policy loss):**
```
L^{CLIP}(θ) = E_t [ min( r_t(θ) Â_t,  clip(r_t(θ), 1−ε, 1+ε) Â_t ) ]
```

With `clip_eps = ε = 0.2`, the ratio is clamped to `[0.8, 1.2]`.

**Value function loss (MSE):**
```
L^{VF}(φ) = E_t [ ( V_φ(s_t) − V^{targ}_t )² ]
```

where `V^{targ}_t = Â_t + V(s_t)` (the GAE-corrected target).

**Entropy bonus:**
```
L^{ENT}(θ) = E_t [ H( π_θ(· | o_t) ) ]
```

For `MultiCategorical`, entropy is the sum of per-dimension Categorical entropies:
```
H = H(dim_0) + H(dim_1) = −Σ_a p_a log p_a  (for each dimension)
```

**Total combined loss (minimised):**
```
L^{total} = −L^{CLIP} + c_v L^{VF} − c_e L^{ENT}
```

with `entropy_coef = c_e = 0.01`. Note: `c_v` is internally set by `ClipPPOLoss` from TorchRL.

### 2.4 Data Collection

Per training iteration:
- `frames_per_batch = 51 200` transitions are collected from `num_envs = 256` parallel environments (= exactly 200 steps per environment = 1 full episode per env if no early termination).
- Data is flattened to a single buffer and shuffled.
- `num_epochs = 5` passes are made over the buffer, each with minibatches of size 1 024.
- Total gradient steps per iteration: `⌊51 200 / 1 024⌋ × 5 = 250 updates`.

---

## 3. Actor Network

### 3.1 Architecture: `DualPolicyNet`

```
Input: obs [B, A, obs_dim]
         │
         ├── obs[:, :n_strikers, :] ──► StrikerMLP ──► [B, ns, act_dim × n_choices]
         │                                                     │
         └── obs[:, n_strikers:, :] ──► JammerMLP  ──► [B, nj, act_dim × n_choices]
                                                               │
                                        cat(dim=1) ──► [B, A, act_dim × n_choices]
                                                               │
                                        reshape    ──► [B, A, act_dim, n_choices]
                                             = [B, A, 2, 7]   (logits)
```

- **`StrikerMLP`** and **`JammerMLP`** are separate `MultiAgentMLP` networks (TorchRL).
- Within each role, **all agents of that role share network parameters** (`share_params=True`).
- Depth: 3 hidden layers, width: 256 neurons, activation: ReLU.
- Strikers and jammers have **different learned behaviours** because they are separate networks.

### 3.2 Distribution: `MultiCategorical`

The logits `[B, A, 2, 7]` are passed to a custom `MultiCategorical` distribution:
- Two independent `Categorical(7)` distributions, one per action dimension.
- **Sample:** argmax (deterministic eval) or categorical sample (stochastic training).
- **Log probability:** `log π(a | o) = log π(a_0 | o) + log π(a_1 | o)` (sum over dims).
- **Entropy:** `H = H_0 + H_1` (sum over dims).

### 3.3 TorchRL Integration

The actor is a `ProbabilisticActor` wrapped around a `TensorDictModule`:

```
TensorDict in:   ("agents", "observation")   → shape [B, A, obs_dim]
TensorDict out:  ("agents", "action")         → shape [B, A, 2]       ← sampled
                 ("agents", "sample_log_prob") → shape [B, A]          ← log π(a|o)
                 ("agents", "logits")          → shape [B, A, 2, 7]    ← raw logits
```

---

## 4. Critic Network

### 4.1 Architecture: `DualCentralisedCritic`

```
Input: global_state [B, state_dim]
         │
         ├── StrikerHead (3-layer MLP) ──► [B, n_strikers]
         │
         └── JammerHead  (3-layer MLP) ──► [B, n_jammers]
                                                    │
                     cat + unsqueeze ──► [B, n_agents, 1]
```

Each head: `state_dim → 256 → 256 → n_agents_in_role`.

### 4.2 Global State Composition

The critic reads **absolute global state** (not ego-centric):

```
state_dim = 2A + 2A + A + 2T + T + 2R
```

For default (A=2, T=1, R=1): `state_dim = 4 + 4 + 2 + 2 + 1 + 2 = 15`.

| Component | Shape | Content |
|---|---|---|
| Agent positions | `[B, 2A]` | Absolute `(x, y)` per agent |
| Agent headings | `[B, 2A]` | `(sin ψ, cos ψ)` per agent |
| Agent alive | `[B, A]` | Binary alive flag |
| Target positions | `[B, 2T]` | Absolute `(x, y)` per target |
| Target alive | `[B, T]` | Binary alive flag |
| Radar positions | `[B, 2R]` | Absolute `(x, y)` per radar |

> ⚠️ **Missing from critic state:** Current `radar_eff_range` (whether any radar is currently being jammed), agent speeds, and step count. The critic therefore cannot distinguish a situation where the jammer is actively suppressing the radar from a situation where it is not.

### 4.3 TorchRL Integration

```
TensorDict in:   "state"                         → shape [B, state_dim]
TensorDict out:  ("agents", "state_value")        → shape [B, A, 1]
```

A **separate head per role** means the striker value function and jammer value function can learn independently — a sensible design given their fundamentally different roles.

---

## 5. Actor–Critic Interaction Flow

```
──────────────────────────────────────────────────────────────────────────────
                          TRAINING ITERATION
──────────────────────────────────────────────────────────────────────────────

 Phase 1: DATA COLLECTION  (on-policy rollout)
 ┌────────────────────────────────────────────────────────────────────────┐
 │                                                                        │
 │   env.reset()                                                          │
 │       │                                                                │
 │       ▼                                                                │
 │   [observations, state]   ─────────────►   Actor(obs)                 │
 │                                                │                       │
 │                                         sample action                  │
 │                                         compute log π(a|o)             │
 │                                                │                       │
 │       ◄──────────────────────────────────  env.step(action)           │
 │                                                                        │
 │   [next obs, reward, done]  ×  max_steps  per env                     │
 │                                                                        │
 │   Stored: (o_t, a_t, log π_old, r_t, done_t, state_t)                 │
 │                                                                        │
 └────────────────────────────────────────────────────────────────────────┘

 Phase 2: VALUE ESTIMATION (no gradient)
 ┌────────────────────────────────────────────────────────────────────────┐
 │                                                                        │
 │   Critic(state_t)       → V(s_t)   for all t                          │
 │   Critic(state_{t+1})   → V(s_{t+1}) for all t                        │
 │                                                                        │
 │   GAE:  δ_t = r_t + γ V(s_{t+1}) − V(s_t)                            │
 │         Â_t = Σ (γλ)^l δ_{t+l}                                        │
 │         V^targ_t = Â_t + V(s_t)                                       │
 │                                                                        │
 └────────────────────────────────────────────────────────────────────────┘

 Phase 3: OPTIMISATION  (5 epochs × ~50 minibatches = 250 gradient steps)
 ┌────────────────────────────────────────────────────────────────────────┐
 │                                                                        │
 │   For each minibatch (size 1 024):                                     │
 │                                                                        │
 │     Actor:   r_t = π_new(a_t|o_t) / π_old(a_t|o_t)                    │
 │              L_CLIP = E[ min( r_t Â_t, clip(r_t, 0.8, 1.2) Â_t ) ]    │
 │                                                                        │
 │     Critic:  L_VF = E[ (V(s_t) − V^targ_t)² ]                        │
 │                                                                        │
 │     Entropy: L_ENT = E[ H(π) ]                                        │
 │                                                                        │
 │     Total:   L = −L_CLIP + L_VF − 0.01 × L_ENT                       │
 │                                                                        │
 │     Adam.step()   (single optimizer for actor + critic)               │
 │     clip_grad_norm(max=1.0)                                            │
 │                                                                        │
 └────────────────────────────────────────────────────────────────────────┘

 Phase 4: POLICY SYNC
 ┌────────────────────────────────────────────────────────────────────────┐
 │   collector.update_policy_weights_()                                   │
 │   (push updated actor weights into the collector for next rollout)     │
 └────────────────────────────────────────────────────────────────────────┘
```

**Key architectural property:** During collection, the **critic is not called**. The actor samples actions fully independently. The critic is only called in Phase 2 (value estimation) and Phase 3 (value loss). This is the CTDE principle in action.

---

## 6. Reward Design

### 6.1 Reward Components

All rewards are per-agent per-step, shape `[B, A, 1]`.

#### Component 1 — Team kill reward (shared)
```
r_kill = (n_targets_killed × target_destroyed) / n_alive_agents
```
Applied equally to all alive agents regardless of role.
- `target_destroyed = 10`
- Cooperative signal: everyone benefits when the striker kills a target.

#### Component 2 — Border penalty (per agent)
```
r_border = max(0, (border_thresh − dist_to_edge) / border_thresh) × border_penalty
```
- `border_thresh = 0.05`, `border_penalty = −1`
- Proportional proximity penalty (0 when safe, −1 at the exact edge).

#### Component 3 — Timestep penalty (per alive agent)
```
r_time = timestep_penalty = −0.1   per step
```
Over 200 steps: cumulative = −20 per agent (larger than the task reward of +10).

#### Component 4 — Jammer proximity reward (jammers only)
```
dist_jr_min = min_r( ||jammer − radar_r|| )
safe_mask   = dist_jr_min > radar_eff_range[nearest radar]
prox_rew_j  = (1 − dist_jr_min / d_max) × jammer_jamming × safe_mask
```
- `jammer_jamming = 0.5`
- Rewards the jammer for being **close to a radar** but **outside its lethal range**.
- Once inside `radar_eff_range` (0.20, or 0.10 if another jammer is jamming), the reward is zeroed.
- **Note:** The jammer only actually suppresses the radar when `dist < jam_radius = 0.25`. The safe zone for reward is `radar_eff_range < dist < jam_radius` = a corridor of only **50 km** (`0.20 < dist < 0.25`).

#### Component 5 — Striker proximity reward (strikers only)
```
dist_target_min = min_t( ||striker − target_t|| )   (over alive targets only)
prox_rew_s      = (1 − dist_target_min / d_max) × striker_proximity
```
- `striker_proximity = 0.5`
- Encourages the striker to close on the nearest alive target, **regardless of radar threat**.

#### Component 6 — Agent destroyed penalty (per killed agent)
```
r_dead = agent_destroyed = −10   (once, on the step of death)
```

### 6.2 Reward Summary Table

| Agent | Signal | Magnitude | When triggered |
|---|---|---|---|
| Both | Team kill bonus | +10 / n_alive | Target destroyed |
| Both | Timestep penalty | −0.1 / step | Every step |
| Both | Border penalty | 0 to −1 | Near map edge |
| Both | Death penalty | −10 | On death step |
| Jammer | Proximity to radar | 0 to +0.5 | Alive, outside radar range |
| Striker | Proximity to target | 0 to +0.5 | Alive, target alive |

---

## 7. Critical Feedback and Failure Analysis

### 7.1 Desired Behaviour vs. What the Reward Teaches

**Intended sequence of events:**
```
1. Jammer approaches radar → enters jam_radius → radar range suppressed from 0.20 to 0.10
2. Radar suppressed → safe corridor for striker opens up
3. Striker approaches target (which is 0.18 away from radar) without being killed
4. Striker enters engage_range (0.10) of target → target destroyed → team reward
```

This is a **sequential cooperative task**. The reward does not explicitly encode this sequence.

---

### Issue 1 — No Coordination Incentive

The jammer reward and striker reward are **completely independent**:
- The jammer is rewarded for being near a radar, regardless of what the striker is doing.
- The striker is rewarded for being near a target, regardless of whether the radar is jammed.

The only signal that links the two roles is the shared `target_destroyed` team reward. To discover the coordination, both agents must by chance execute the full correct sequence simultaneously — a very long credit assignment chain for sparse reward.

**Fix suggestions:**
- Add a shaped reward to the striker that is proportional to `(radar_eff_range_baseline − current_radar_eff_range)` — reward the striker more when a radar is actually being jammed.
- Or add a reward to the jammer that scales with **how close the striker is to the target** times whether the jammer is within jammer range — explicitly coupling the two roles.

---

### Issue 2 — Geometric Incompatibility: Target Spawn vs. Radar Range

Targets spawn at distance `0.9 × radar_range = 0.18` from a radar.

The striker must get within `engage_range = 0.10` of the target to destroy it.

When the striker is at the target position (`dist_to_target = 0.10`), its distance to the radar is approximately:
```
dist_to_radar ≈ 0.18 − 0.10 = 0.08   (if approaching from the far side)
            or ≈ 0.18 + 0.10 = 0.28   (if approaching from the near side)
```

Even with jamming active (`radar_eff_range = 0.10`), the striker will be killed if it approaches from the **near side** (the side between radar and target). The agent must learn to approach the target from the **far side** (away from the radar). This requires:
1. The striker to learn **geometric reasoning about the relative positions** of target and radar.
2. The jammer to be in position **simultaneously**.

Neither of these is directly encoded in the current reward.

---

### Issue 3 — Timestep Penalty Dominates Task Reward

Accumulated timestep penalty over a full 200-step episode: `200 × (−0.1) = −20`.  
Task success reward: `+10 / n_alive` ≈ `+5` per agent (shared with the other agent).  
Death penalty: `−10`.

This means that **the timestep penalty is 4× larger than the task success reward**. Agents may find it locally rational to terminate episodes quickly (e.g., by dying) rather than solving the task.

**Fix:** Reduce `timestep_penalty` to `−0.01` (giving a max cumulative of `−2`), keeping it below the task success signal.

---

### Issue 4 — Jammer Safe Zone is Too Narrow

The target jamming corridor is `0.20 < dist < 0.25` (only 50 km = 0.05 units).

At `v_min = 0.01 units/step`, the jammer crosses this corridor in ~5 steps. With any heading error, the jammer either:
- Is too far (not jamming, no reward), or
- Enters radar range and is killed instantly (`kill_probability = 1.0`).

The corridor width is comparable to the agent's turning radius (`min_turn_radius = 0.01`, but heading changes are slow).

**Fix:** Increase the gap between `radar_range` and `jammer_jam_radius`, e.g. set `jammer_jam_radius = 0.35` (keeping `radar_range = 0.20`). This creates a 150 km safe jamming corridor instead of 50 km.

---

### Issue 5 — Critic Missing Jamming State

The global state read by the critic does **not include `radar_eff_range`**. This means the critic cannot distinguish:
- A situation where the jammer is actively jamming (radar range = 0.10), and
- A situation where it is not (radar range = 0.20).

These two states have very different values for the striker, but the critic assigns them the same value estimate. This degrades advantage estimation quality and makes training unstable.

**Fix:** Add `radar_eff_range` (`[B, R]`) to the global state tensor in `_build_global_state()`.

---

### Issue 6 — `normalize_advantage = False`

In `training/utils.py`, advantage normalisation is disabled. Without normalisation, advantages vary in raw magnitude across batches. A batch containing many terminal events (deaths, kills) will have advantages in the range `[−20, +10]`, while a quiet batch will have advantages near `[−2, 0]`. This causes the effective learning rate to vary dramatically between batches, producing the observed high variance in training curves.

**Fix:** Set `normalize_advantage = True` in `make_ppo_loss()`.

---

### Issue 7 — Single Optimizer for Actor and Critic

Both the actor and critic are updated with the same `Adam(lr=1e-4)`. The critic generally benefits from a higher learning rate (faster value fitting) while the actor should update more conservatively. With a shared optimizer, one must compromise.

**Fix:** Create two separate optimizers:
```python
actor_optimizer  = Adam(actor.parameters(),  lr=3e-4)
critic_optimizer = Adam(critic.parameters(), lr=1e-3)
```

---

### Issue 8 — n_env_layouts = 1 (Single Fixed Scenario)

With `n_env_layouts = 1`, all 256 parallel environments use the same radar position every episode. The policy memorises the specific layout rather than generalising. At test time with a slightly different scenario, performance degrades.

**Fix:** Use `n_env_layouts = 0` (fully random) or `n_env_layouts ≥ 10` for scenario diversity.

---

### Issue 9 — Very Short Training Budget

With `n_iters = 40` and `frames_per_batch = 51 200`:
- Total frames: `40 × 51 200 = 2 048 000`
- Estimated episodes: `2 048 000 / 200 = 10 240 episodes`

For a sequential multi-agent coordination task with sparse reward, this is extremely short. State-of-the-art MARL benchmarks typically require 1M–100M episodes for comparable difficulty.

**Fix:** Increase to `n_iters = 200–500` as a minimum.

---

### Issue 10 — Striker Proximity Reward Ignores Radar Threat

The striker receives a proximity reward for closing on the target regardless of radar state. This incentivises the striker to always move toward the target, even when doing so means entering radar range and being killed. The agent learns "move toward target" but cannot learn "wait for jammer to suppress radar first."

**Fix:** Weight the striker proximity reward by the degree of radar suppression:
```python
jam_ratio = 1.0 - (radar_eff_range.min() / radar_range)  # 0 if not jammed, 1 if fully jammed
prox_rew_s_shaped = prox_rew_s * (0.3 + 0.7 * jam_ratio)  # always some shaping, more when jammed
```

---

### 7.2 Summary of Critical Issues

| # | Issue | Severity | Root cause |
|---|---|---|---|
| 1 | No coordination incentive between roles | **Critical** | Reward design |
| 2 | Geometric incompatibility: striker must approach from correct side | **High** | Spawn logic + engage_range |
| 3 | Timestep penalty dominates task reward (−20 vs +5) | **High** | Reward scale |
| 4 | Jammer safe zone only 50 km wide | **High** | jam_radius vs radar_range |
| 5 | Critic missing jamming state | **Medium** | Global state composition |
| 6 | `normalize_advantage = False` → erratic gradient magnitudes | **Medium** | Training config |
| 7 | Single optimizer for actor+critic | **Medium** | Training setup |
| 8 | Single fixed scenario (overfit to one layout) | **Medium** | n_env_layouts = 1 |
| 9 | Training too short for coordination task | **Medium** | n_iters = 40 |
| 10 | Striker reward ignores radar threat | **Medium** | Reward design |

---

*End of document — generated from codebase review, March 2026.*
