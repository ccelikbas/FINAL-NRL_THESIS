# Training Mechanics & High Variance Analysis

## 1. Episode Reward (`ep_rew`) — What It Means & How It's Plotted

### What is `ep_rew`?
`ep_rew` (**episode reward**) is the **cumulative reward summed over an entire episode** (from reset to terminal state or timeout). 

**Location in code:** [trainer.py line 164-169](strike_ea/training/trainer.py#L164-L169)
```python
# Extract episode rewards for episodes that completed this iteration
done_mask = td.get(("next", base_env.group, "done"))  # [B, 1] boolean mask
ep_rew = td.get(("next", base_env.group, "episode_reward"))[done_mask]  # [B, n_agents, 1] → filtered

# Average across completed episodes
ep_rew_mean = float(torch.nanmean(ep_rew).item())  # Ignore NaN values
```

**Key insight:** 
- Each iteration collects `frames_per_batch` (51,200) transitions across `num_envs` (256) parallel environments
- Not all environments complete an episode in every iteration
- Only environments with `done=True` contribute to `ep_rew_mean`
- If **no episodes complete in an iteration**, `ep_rew_mean = NaN`

### How is it plotted?
[visualize.py line 22-26](strike_ea/evaluation/visualize.py#L22-L26):
```python
ax1.plot(logs["episode_reward_mean"])  # One point per iteration
ax1.set_title("Training: Episode reward mean")
ax1.set_ylabel("Mean episode reward")
```

**Plot characteristics:**
- **X-axis:** Iteration number (0 to n_iters = 100)
- **Y-axis:** Mean episode reward across agents that finished that iteration
- **Interpretation:** 
  - Upward trend = agents learning (receiving more cumulative reward)
  - High variance = **highly unstable policy** (some episodes very good, some very bad)
  - NaN values = **no episodes completed** in that iteration

---

## 2. Actor Parameter Updates — When & How

### The Training Loop

[trainer.py line 98-153](strike_ea/training/trainer.py#L98-L153)

```
FOR each iteration (1 to n_iters):
    ├─ COLLECT ROLLOUT (single forward pass)
    │  ├─ Run actor on observations from all 256 envs
    │  ├─ Sample actions from MultiCategorical(7 choices) per action dim
    │  └─ Collect 51,200 transitions: (obs, action, reward, next_obs, done)
    │
    ├─ COMPUTE ADVANTAGES (no gradient)
    │  ├─ Run critic on current_state → V(s_t)
    │  └─ Run critic on next_state → V(s_{t+1})
    │  └─ Compute TD residuals: δ = r + γV(s_{t+1}) - V(s_t)
    │  └─ Apply GAE: Advantage = Σ(λγ)^l δ_{t+l}
    │
    ├─ RESHAPE & MINIBATCH (prepare for GPU)
    │  └─ Flatten [B, n_agents, ...] → [B*n_agents, ...]
    │
    └─ UPDATE PARAMETERS (gradient steps)
       FOR each of num_epochs (10) passes:
           ├─ Shuffle data into random minibatches (size 1,024)
           │
           FOR each minibatch:
            ├─ Forward: compute log π(a|s), V(s)
            ├─ Compute PPO loss:
            │    loss_actor = -E[ min(r_t * Âₜ, clip(r_t, 1±ε) * Âₜ) ]
            │    loss_critic = MSE(V̂(s) - G_t)
            │    loss_entropy = -H[π]  (bonus)
            │    loss_total = loss_actor + loss_critic - entropy_coef * H
            ├─ Backward pass (compute gradients)
            ├─ Clip gradients: ||∇|| > max_grad_norm → rescale
            └─ Adam optimizer.step()  ← **ACTOR PARAMS UPDATED HERE**
```

### When are actor parameters updated?

- **1 gradient step per minibatch**: With 51,200 transitions, ~51 minibatches per epoch, 10 epochs → **~510 gradient updates per iteration**
- **Timing:** After each minibatch, immediately (not accumulated)
- **Frequency:** Once per ~100ms per step (GPU-dependent)

### What gets updated?

Actor network only:
```python
# Both striker and jammer policy MLPs parameters updated jointly
actor = DualPolicyNet(
    striker_net=MultiAgentMLP(...),  # ← Updated
    jammer_net=MultiAgentMLP(...),   # ← Updated
)
```

Critic is also updated but separately:
```python
critic = DualCentralisedCritic(
    striker_head=Sequential(...),  # ← Also updated
    jammer_head=Sequential(...),   # ← Also updated
)
```

---

## 3. Sequential Steps in Learning & Parallel Environments

### Schematic: How 256 Parallel Environments Work

```
STEP 1: Reset all 256 envs
├─ Each env in parallel initializes random radar position, agent spawns
├─ Returns obs shape [B=256, n_agents=4, obs_dim=18]
└─ Returns state shape [B=256, state_dim]

STEP 2: Policy rollout (vectorized, no loop)
├─ Actor.forward(obs [256, 4, 18])
│  ├─ Splits by agent type internally
│  ├─ Runs striker_net([256, 2, 18]) → [256, 2, 2, 7] logits
│  └─ Runs jammer_net([256, 2, 18]) → [256, 2, 2, 7] logits
│  └─ Samples actions [256, 4, 2] ∈ {0..6}
├─ All 256 envs take actions simultaneously (vectorized)
└─ Returns: next_obs [256, 4, 18], reward [256, 4, 1], done [256, 1]

STEP 3: Repeat steps 1-2
├─ For each of 256 envs independently, run 32 steps (51,200 / 256)
├─ Some envs finish episodes (done=True), others continue
└─ Total: 51,200 transitions collected

        ┌─ Env 1: Episode finishes at step 15
        │  (returns to STEP 1, starts new episode)
        │
        ├─ Env 2: Still running (step 32 of 200)
        │
        └─ Env 256: Episode finishes at step 28
           (returns to STEP 1, starts new episode)

STEP 4: Compute advantages (all data touchable after collection)
├─ Flatten [256, 4, ...] → [1024, ...]
├─ Run critic on all 1024 state vectors
└─ Compute advantages via GAE

STEP 5: Mini-batch gradient updates
├─ Shuffle 1024 data points
├─ Process in batches of 32-128 per GPU
└─ Update actor & critic 500+ times
```

### Key insight on variance

**High variance likely comes from:**
1. **Epsiodic termination variance:** 
   - If some episodes are completing after just 10 steps (agents dying), and others after 150 steps (success), the distribution of episode rewards is **bimodal**
   - Early episode rewards: mostly penalties → very negative
   - Late episodes: some successes → mixed signals

2. **Episode lengths vary:** 
   - With 256 parallel envs, some episodes may finish in 15 steps (all agents killed by radar)
   - Others go 200 steps without destroying a target
   - The `ep_rew` is the **cumulative reward** — short episodes have different distributions than long ones

3. **Sparse rewards:**
   ```python
   target_destroyed: 200  # Huge bonus, rare events
   agent_destroyed: -200   # Huge penalty
   timestep_penalty: -1    # Small per-step cost
   ```
   If targets are hard to destroy, most episodes end with only timestep penalties, creating **bimodal distribution**: episodes where 1+ target is destroyed (high variance success) vs episodes that timeout (consistent negative).

---

## 4. Do You Use a Buffer for GPU Parallelization?

### Short answer: NO explicit replay buffer, but YES implicit buffering

**TorchRL's Collector maintains an internal buffer:**

[trainer.py line 79-81](strike_ea/training/trainer.py#L79-L81):
```python
collector = make_collector(env, actor, frames_per_batch, n_iters, device)
```

[utils.py line 53-74](strike_ea/training/utils.py#L53-L74):
```python
def make_collector(..., frames_per_batch, total_frames, ...) -> Collector:
    # TorchRL's Collector internally:
    # 1. Runs rollouts asynchronously (or synchronously in newer versions)
    # 2. Stores transitions in a replay buffer (default: on CPU or GPU depending on storing_device)
    # 3. Returns batched TD after collecting frames_per_batch transitions
    # 4. **Discards buffer** before next rollout (on-policy)
```

**Flow:**
```
Collector maintains [size=frames_per_batch] buffer
├─ Iteration 1: Collect 51,200 frames → Buffer fills → Return to trainer
├─ Trainer: Updates policy 500 times on these 51,200 transitions
├─ After updates: Collector.shutdown() (buffer discarded)
└─ Collector re-initializes buffer
    └─ Iteration 2: Collect NEW 51,200 frames → ...
```

**Why this matters for variance:**
- **On-policy learning:** Data becomes "stale" after policy updates
- If you run policy for 32 steps, then update policy 10 times (500 gradient steps), policy has **changed significantly**
- When you collect the next 32 steps with the new policy, transitions using old policy no longer match current policy
- **→ This is acceptable** (on-policy is MAPPO's design), but **can cause instability** if policy updates are aggressive

---

## 5. Loss Function Plots — What's Minimized vs Maximized?

### The 4 Loss Components

[trainer.py line 133-139](strike_ea/training/trainer.py#L133-L139):
```python
loss_policy  = E[ -min(r_t * Âₜ, clip(r_t, 1±ε) * Âₜ) ]   # Minimize (negative)
loss_value   = MSE(V̂(s) - Gₜ)                               # Minimize
loss_entropy = -H[π]                                        # Minimize (negative entropy)
loss_total   = loss_policy + loss_value + loss_entropy
```

### What Each Loss Means

| Loss Component | Formula | Direction | Meaning |
|---|---|---|---|
| **loss_policy** | `min(ratio * advantage, clip(ratio) * advantage)` | **↓ minimize** | Aligned actions (good trajectories) get higher probability |
| **loss_value** | `(V_predicted - return)²` | **↓ minimize** | Critic learns to predict cumulative rewards accurately |
| **loss_entropy** | `-sum(π * log π)` shown as **negative** | **↓ minimize** (= ↑ entropy) | Encourages exploration (stochastic policy) |
| **loss_total** | `policy + value + entropy` | **↓ minimize** | All balanced |

### Interpretation of the plot:

**Typical healthy MAPPO training:**
```
Iteration →

loss_policy:   ╱╱╱ drops → plateaus → wiggle (policy converging)
loss_value:    ╱╱╱ drops → plateaus (critic converging)
loss_entropy:  ░░░ stays low (entropy penalty active)
loss_total:    ╱╱╱ drops monotonously (net loss decreasing)
```

### Why HIGH VARIANCE in `episode_reward_mean`?

The losses can be **low** (good) while `episode_reward_mean` is **noisy** because:

1. **Sparse reward + stochastic environment**
   - Policy may be improving (loss ↓), but episode outcomes still highly variable
   - Target destruction is rare → episode rewards depend on which rare events occurred
   
2. **Environment has chaos**
   - With relative observations and new 7-choice action space, agent policy may not have stabilized yet
   - Early training: exploration → high variance
   - Expected: variance should diminish as policy becomes deterministic

3. **Critic can't reduce variance it doesn't see**
   - Advantage estimation helps, but if episodes are fundamentally variable (some short deaths, some long successes), value estimates can't eliminate that randomness
   - **Advantage** = reward - baseline_value: if reward is ±200 (sparse), advantage is ±200 regardless of value fit

---

## Diagnosis: Why Is YOUR Training Unstable?

### Likely culprits:

1. **New relative observation space**
   - Agents haven't learned to interpret ego-centric (distance, angle) observations yet
   - Early training = high exploration randomness
   - **Fix:** Train longer (increase `n_iters`)

2. **New 7-choice action space**
   - Instead of {-1, 0, +1}, now {-1, -0.5, -0.1, 0, +0.1, +0.5, +1}
   - More action variance → noisier early learning
   - **Fix:** Tune `entropy_coef` (higher = more exploration penalty = lower variance)

3. **Reward signal still sparse**
   - `target_destroyed: 200` happens rarely (depends on successful navigation)
   - `agent_destroyed: -200` happens when agents blunder
   - **Fix:** Increase shaping rewards (`striker_proximity`, `jammer_jamming`)

4. **Batch size too small or learning rate too high**
   - With `minibatch_size: 1024` and `lr: 1e-4`, gradients may be noisy
   - **Fix:** Reduce `lr` to `5e-5`, or increase `minibatch_size` to `2048`

5. **Clipping too loose or too tight**
   - `clip_eps: 0.1` is moderate; if set too high (0.2+), allows wild policy swings
   - **Fix:** Try `clip_eps: 0.05` (more conservative)

---

### Quick tuning checklist to reduce variance:

| Problem | Adjustment | Effect |
|---|---|---|
| **Noisy gradients** | ↓ learning rate: `1e-4 → 5e-5` | Smoother updates |
| **Exploring too much** | ↓ entropy_coef: `0.01 → 0.001` | More deterministic policy |
| **Policy changing too fast** | ↓ clip_eps: `0.1 → 0.05` | Conservative policy updates |
| **Sparse rewards dominating** | ↑ shaping rewards (proximity, jamming) | More continuous signal |
| **Too few episodes completing** | Monitor first 5 iterations; if all NaN, agents dying instantly → ↑ radar_kill_probability = 0.5 | Easier early learning |
| **High variance expected** | ↑ n_iters: `100 → 500` | Longer training → convergence |

---

### Monitoring script (add to trainer.py):

```python
# After each iteration, log:
if it > 0 and ep_rew.numel() > 0:
    ep_rew_std = float(torch.nanstd(ep_rew).item())
    ep_rew_min = float(torch.nanmin(ep_rew).item())
    ep_rew_max = float(torch.nanmax(ep_rew).item())
    n_complete = ep_rew.numel()
    
    print(
        f"Iter {it:3d}: mean={ep_rew_mean:7.1f}, "
        f"std={ep_rew_std:7.1f}, "
        f"range=[{ep_rew_min:7.1f}, {ep_rew_max:7.1f}], "
        f"complete={n_complete}"
    )
```

This will show:
- **If std is large:** episodes highly variable → instability
- **If range is wide:** some huge successes, some huge failures → bimodal
- **If complete < 10:** few episodes finishing → timeout-dominated → adjust task difficulty
