"""Smoke test: env reset/step, actor/critic forward, no NaN."""
import torch
from strike_ea.env import StrikeEA2DEnv
from strike_ea.models import make_actor, make_critic

env = StrikeEA2DEnv(num_envs=4, max_steps=10, device="cpu",
                    n_strikers=2, n_jammers=2, n_targets=2, n_radars=2)

print(f"obs_dim   = {env.obs_dim}")
print(f"state_dim = {env.state_dim}")

A, T, R = 4, 2, 2
expected = A * (2*T + 2*(A-1) + 2*R) + A + T + R
print(f"expected  = {expected}")
assert env.state_dim == expected, f"state_dim mismatch: {env.state_dim} != {expected}"

td = env.reset()
obs   = td.get(("agents", "observation"))
state = td.get("state")
print(f"obs shape   = {obs.shape}")
print(f"state shape = {state.shape}")
assert obs.shape == (4, 4, env.obs_dim)
assert state.shape == (4, expected)
assert not torch.isnan(obs).any(), "NaN in obs"
assert not torch.isnan(state).any(), "NaN in state"
print("Reset OK")

act = torch.randint(0, 7, (4, 4, 2))
td.set(("agents", "action"), act)
next_td = env._step(td)
obs2   = next_td.get(("agents", "observation"))
state2 = next_td.get("state")
rew    = next_td.get(("agents", "reward"))
print(f"step obs   = {obs2.shape}")
print(f"step state = {state2.shape}")
print(f"step rew   = {rew.shape}")
assert not torch.isnan(obs2).any(), "NaN in step obs"
assert not torch.isnan(state2).any(), "NaN in step state"
assert not torch.isnan(rew).any(), "NaN in step rew"
print("Step OK")

actor  = make_actor(env, hidden=64)
critic = make_critic(env, hidden=64)
print(f"actor  params = {sum(p.numel() for p in actor.parameters()):,}")
print(f"critic params = {sum(p.numel() for p in critic.parameters()):,}")

with torch.no_grad():
    actor_out = actor(td)
    critic_out = critic(td)
    sv = td.get(("agents", "state_value"))
    print(f"state_value shape = {sv.shape}")

print("ALL SMOKE TESTS PASSED")
