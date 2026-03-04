# Visual Diagrams: Training Mechanics

## 1. The MAPPO Training Loop (Detailed Timing)

```
INITIALIZATION (one-time)
│
├─ base_env = StrikeEA2DEnv(num_envs=256, max_steps=200)
├─ actor = DualPolicyNet(...) + ProbabilisticActor
├─ critic = DualCentralisedCritic(...)
├─ collector = Collector(env, actor, frames_per_batch=51_200)
└─ optimizer = Adam(loss_module.parameters(), lr=1e-4)

FOR it in range(n_iters=100):
│
├─ PHASE 1: Data Collection (≈100-200ms)
│  │
│  ├─ collector.collect() → iterates over 256 envs
│  │  │
│  │  └─ FOR 32 steps per environment (51_200 / 256):
│  │     │
│  │     ├─ obs [256, 4, 18] from all parallel envs
│  │     │
│  │     ├─ actor(obs) [256, 4, 2, 7] logits
│  │     │  ├─ striker_net(obs[:,:2,:])
│  │     │  └─ jammer_net(obs[:,2:,:])
│  │     │
│  │     ├─ π.sample() → action [256, 4, 2] ∈ {0..6}
│  │     │
│  │     ├─ env.step(action) [256, 4, 2]
│  │     │  ├─ Step physics for all 256 envs (vectorized)
│  │     │  └─ Returns: next_obs [256, 4, 18],
│  │     │            reward [256, 4, 1],
│  │     │            done [256, 1]
│  │     │
│  │     └─ Accumulate in buffer:
│  │        ├─ (obs, action, log_π, next_obs, reward, done)
│  │        └─ Some envs reset internally (done=True)
│  │
│  └─ Total buffer size: 51,200 transitions
│
├─ PHASE 2: Advantage Estimation (≈50ms, no gradients)
│  │
│  ├─ with torch.no_grad():
│  │  ├─ critic(td) → V(s_t) [1024]
│  │  └─ critic(td.next) → V(s_{t+1}) [1024]
│  │
│  ├─ Compute TD residuals: δ_t = r_t + γV(s_{t+1}) - V(s_t)
│  └─ GAE: Â_t = Σ(λγ)^l δ_{t+l}  (smooth advantage)
│
├─ PHASE 3: Policy Update (≈2-5 seconds)
│  │
│  ├─ data.reshape(-1) → [1024, ...] (flatten batch & agent dims)
│  │
│  └─ FOR epoch in range(num_epochs=10):
│     │
│     ├─ perm = torch.randperm(1024)  # Shuffle for SGD
│     │
│     └─ FOR minibatch_start in range(0, 1024, minibatch_size=1024):
│        │
│        ├─ idx = perm[minibatch_start : minibatch_start+1024]
│        │
│        ├─ sub = data[idx]  # [1024, ...]
│        │
│        ├─ loss_vals = loss_module(sub)
│        │  ├─ Forward actor: π(a|s) [1024, 2] logits → log_π
│        │  ├─ Forward critic: V(s) [1024, 1]
│        │  ├─ loss_actor = -mean(min(ratio*Â, clip(ratio)*Â))
│        │  ├─ loss_critic = mean((V-G)²)
│        │  └─ loss_entropy = -mean(entropy[π])
│        │
│        ├─ total_loss = loss_actor + loss_critic + loss_entropy
│        │              [1024] → scalar
│        │
│        ├─ optimizer.zero_grad()
│        ├─ total_loss.backward()  ← Compute ∇loss w.r.t. all parameters
│        ├─ torch.nn.utils.clip_grad_norm_(model, max_norm=1.0)
│        └─ optimizer.step()  ← **PARAMETERS UPDATED**
│           (Adam updates: θ ← θ - α * ∇loss)
│
├─ PHASE 4: Logging & Cleanup
│  │
│  ├─ Extract completed episodes: done_mask
│  ├─ Compute mean episode reward
│  ├─ Log: episode_reward_mean, loss_total, loss_policy, loss_value, loss_entropy
│  │
│  └─ collector.update_policy_weights_()  (copy updated weights to inference threads)
│
└─ END ITERATION
   (Buffer discarded, Policy retained)

FINAL: Return (base_env, actor, critic, logs)
       Plot logs["episode_reward_mean"], logs["loss_*"]
```

---

## 2. How 256 Parallel Environments Work (Simplified)

```
ENVIRONMENT STATE (persistent across entire training)

Env 1:   [agents] [targets] [radars]
Env 2:   [agents] [targets] [radars]
Env 3:   [agents] [targets] [radars]
...
Env 256: [agents] [targets] [radars]

┌─ All 256 envs step() in parallel (vectorized) ─────────────────┐
│                                                                  │
│  obs_batch [256, 4, 18]  ──actor(obs_batch)──→  action_batch    │
│                                                   [256, 4, 2]    │
│                                                        │         │
│  env_batch.step(action_batch) runs ALL 256 simultaneously       │
│                                                        │         │
│  Returns: next_obs [256, 4, 18]                      │         │
│           reward [256, 4, 1]                         │         │
│           done [256, 1]  ← {True, False}            │         │
│                                                        │         │
│  If Env[i].done[i]==True:                            │         │
│    ├─ Episode finished                               │         │
│    ├─ "episode_reward" = sum of all rewards          │         │
│    └─ Env[i] auto-resets → new episode starts        │         │
│                                                        │         │
│  If Env[i].done[i]==False:                           │         │
│    └─ Episode continues (same agents/targets/radars) │         │
│                                                                  │
│  REPEAT 32 times (51_200 / 256 = 200)                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

EPISODE COMPLETION PATTERN

Iteration 1:  

Env 1   Step: 1  2  3  4  5  6  7  8  9  10|DONE (reward=-150)
Env 2   Step: 1  2  3  4  5  6  7  8  9  10|DONE (reward=+50)
Env 3   Step: 1  2  3  4  5  6  7  8  9  10|DONE (reward=-200)
Env 4   Step: 1  2  3  4  5  6  7  8  9  10|DONE (reward=-100)
...
Env 256 Step: 1  2  3  4  5  6  7  8  9  10|DONE (reward=-50)

After iteration 1:
├─ ~256 episodes completed (~51 episodes × 5 steps average)
│  (Not exact; depends on max_steps timeout)
├─ All episodes reset
├─ Collect episode_reward values: [-150, +50, -200, -100, ..., -50]
└─ Compute mean: sum / count ≈ -120 (with HIGH VARIANCE if spread wide)

Iteration 2: (same process with updated policy)
```

---

## 3. Loss Landscape During Training

```
ITERATION 0 (random initialization)
│
│   loss_total = 50 ┃ loss_policy = 30
│                   ┃ loss_value = 15
│                   ┃ loss_entropy = 5
│
├─ Episode reward mean: mostly NEGATIVE (agents blundering)
│
ITERATION 10 (early learning)
│
│   loss_total = 20 ┃ loss_policy = 10
│                   ┃ loss_value = 8
│                   ┃ loss_entropy = 2
│
├─ Episode reward mean: improving, but NOISY
│  └─ Some episodes: -200 (all agents killed fast)
│  └─ Some episodes: -50 (agents navigate, die at border)
│  └─ Few episodes: +10 (rare successes)
│
ITERATION 50 (mid-training)
│
│   loss_total = 5 ┃ loss_policy = 2
│                  ┃ loss_value = 2.5
│                  ┃ loss_entropy = 0.5
│
├─ Episode reward mean: STILL NOISY (high variance!)
│  └─ Possible causes:
│     1. Sparse rewards (target_destroyed is rare)
│     2. Observation/action space still confusing agents
│     3. Environment chaos (radar kill probability = 1.0)
│
ITERATION 100 (convergence)
│
│   loss_total = 1 ┃ loss_policy = 0.3
│                  ┃ loss_value = 0.5
│                  ┃ loss_entropy = 0.2
│
├─ Episode reward mean: CONVERGED (low variance)
│  └─ Policy deterministic; agents follow learned strategy
```

---

## 4. What the 4 Loss Plots Mean

```
PLOT 1: Loss Components over time

loss_policy         loss_value          loss_entropy        loss_total
(PPO actor loss)    (Value regression)  (Exploration bonus) (sum of all)

    50 ╱╲                50 ╱╲                10 ╱╲              50 ╱╲
       ╱  ╲                  ╱  ╲                 ╱  ╲                ╱  ╲
   25 ┤   ╲            25 ┤   ╲              5  ┤    ╲          25 ┤    ╲
       ╱     ╲___            ╱     ╲___           ╱      ╲__            ╱      ╲___
    0 └─────────────      0 └─────────────      0 └──────────       0 └─────────────
      0  25  50  75 100    0  25  50  75 100    0  25  50  75 100   0  25  50  75 100
      ─── MIN (good)      ─── MIN (good)       ─── MIN (search)    ─── MIN (good)
      Why: As π improves, Why: Critic learns  Why: Entropy naturally
          policy already    value predictions  fades as policy
          good → less room  → loss decreases   converges to
          for improvement                      deterministic

INTERPRETATION by phase:

Phase 1: All losses high (random policy)
├─ loss_policy: 30+ (all actions equally bad)
├─ loss_value: 15+ (critic predicts wild vals)
└─ loss_entropy: high (maximum randomness)

Phase 2: All dropping (optimization active)
├─ loss_policy: ↓ (better actions selected)
├─ loss_value: ↓ (critic learning)
└─ loss_entropy: ↓ (gradual determinism)

Phase 3: Plateau (convergence or overfitting)
├─ loss_policy: stable (policy saturated)
├─ loss_value: may wiggle (chasing moving targets)
└─ loss_entropy: ~0 (policy deterministic)

HEALTHY SIGN: All curves smooth downward (no spikes)
UNHEALTHY SIGN: All curves oscillate wildly (training unstable)
```

---

## 5. Episode Reward Distribution (Root Cause of High Variance)

```
HISTOGRAM of episode rewards from a single iteration

If HIGH VARIANCE (current problem):

  Frequency
      │
   10 │           ┌─────┐
      │           │     │
    5 │     ┌─────┤     ├─────┐
      │     │     │     │     │
    0 └─────┴─────┴─────┴─────┴────── Reward value
      -300  -200  -100    0   +100   +200

  ← Bimodal: either agents die fast (-300) or succeed (+50)
  ← High variance = spread is wide
  ← Agents not learning middle ground


If LOW VARIANCE (target):

  Frequency
      │
   30 │          ┌────────┐
      │          │        │
   15 │     ┌────┤        ├────┐
      │     │    │        │    │
    0 └─────┴────┴────────┴────┴────── Reward value
      -50   -25    0    +25   +50

  ← Gaussian: episodes cluster around mean
  ← Low variance = all similar quality (learned policy)
  ← Stable training
```

---

## 6. Decoder: Reading Your Current Training Output

```
When you see:
│
├─ Iter   1 | ep_rew  -100.5 | loss   4.2531
├─ Iter   2 | ep_rew  -250.3 | loss   3.8042
├─ Iter   3 | ep_rew   +30.1 | loss   2.1045  ← Big swing!
├─ Iter   4 | ep_rew  -180.2 | loss   1.5032
├─ Iter   5 | ep_rew     NaN | loss   1.2045  ← No episodes finished!
├─ Iter   6 | ep_rew  -50.4  | loss   0.9032

DIAGNOSIS:

ep_rew swings -500 to +100? 
  └─ HIGH VARIANCE ← You're here
     ├─ Cause: bimodal environment (agents either die instantly or survive)
     ├─ Fix: ↑ radar_kill_probability or ↓ encounter difficulty
     └─ Or: Increase n_iters to let learning stabilize

NaN for some iterations?
  └─ No episodes completed that iteration
     ├─ Cause: max_steps=200 but episodes running full 200 without reset
     ├─ Fix: Decrease max_steps or set n_env_layouts to encourage variety
     └─ Or: This is OK; just means timeout-dominated episodes

Loss dropping steadily?
  └─ Good! Network is optimizing
     ├─ Even if ep_rew is noisy, loss ↓ means gradients working
     └─ Wait for convergence (loss plateaus) + more iterations

Loss stable but ep_rew noisy?
  └─ Normal for early training
     ├─ Policy improving but environment is chaotic
     ├─ Fix: Add curriculum (start easier, increase difficulty)
     └─ Or: Increase entropy_coef for more exploration time
```

---

## Key Numbers for Your Setup

```
COLLECTION PHASE (per iteration):
  ├─ 256 parallel envs
  ├─ 32 steps per env (51_200 / 256)
  ├─ 10-50 episodes complete (if max_steps=200 average → ~40 steps per episode)
  ├─ Total transitions: 51_200
  └─ Time: 100-300ms (GPU-dependent)

GRADIENT UPDATE PHASE (per iteration):
  ├─ 10 epochs
  ├─ ~50 minibatches per epoch (1_024 minibatch / ~50-100 size)
  ├─ ~500 gradient steps total
  ├─ ~50,000 forward/backward passes
  └─ Time: 2-8 seconds (GPU-dependent)

PARAMETER UPDATE RATE:
  ├─ 500 updates per iteration
  ├─ 100 iterations total
  ├─ 50,000 total parameter updates during entire run
  └─ Actor learns from ~5 million transitions total (51_200 × 100)
```
