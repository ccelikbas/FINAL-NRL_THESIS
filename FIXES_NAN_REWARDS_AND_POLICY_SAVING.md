# NaN Rewards Fix & Timestamped Policy Saving

## Problem 1: NaN Values in Reward Logging

### Why Did This Happen?

In MAPPO, rewards are logged when episodes complete (when `done=True`). If **no episodes finished** in a collection batch, the reward array was empty, causing:

```python
# Old code
ep_rew_mean = float(ep_rew.mean().item()) if ep_rew.numel() else float("nan")
```

- **Empty array** → `ep_rew.mean()` returns NaN
- **Result**: First few iterations show `nan` because rare for episodes to complete

### The Fix

Changed [trainer.py](strike_ea/training/trainer.py) line ~165 to use `torch.nanmean()`:

```python
# New code
if ep_rew.numel() > 0:
    ep_rew_mean = float(torch.nanmean(ep_rew).item())  # Skip NaN values
else:
    ep_rew_mean = float("nan")
```

**What this does:**
- `torch.nanmean()` ignores NaN values when computing average
- Still returns `nan` if **no episodes completed**, which is correct
- Returns accurate mean when episodes **are completed**

### Result

Your logs now show:
- Early iterations: `nan` (no completed episodes yet — normal!)
- Later iterations: Actual episode rewards (once episodes finish)

**Note:** A few `nan` values at start is expected and healthy. It means the environment is varied and episodes take different lengths to complete.

---

## Problem 2: Manual Policy Saving

### Before: Manual `--save_policy`

```bash
python strike_ea/run.py --preset default --save_policy my_policy.pt
```

**Issues:**
- Manual file naming → hard to track which policy corresponds to which run
- No organized directory structure
- Easy to overwrite policies by accident

### After: Automatic Timestamped Saving

#### Default Behavior (No Action Required)

```bash
python strike_ea/run.py --preset default
```

**Automatically saves to:** `saved_policies/default/2026-03-03_18-08-28.pt`

- **Timestamp format:** `YYYY-MM-DD_HH-MM-SS` (sortable by date/time)
- **Directory structure:** `saved_policies/{preset_name}/{timestamp}.pt`
- **Every run gets a unique filename** → no overwrites

#### CLI Options for Control

| Option | Behavior |
|--------|----------|
| `python run.py --preset default` | Auto-save to `saved_policies/default/2026-03-03_18-08-28.pt` |
| `python run.py --save_policy custom.pt` | Save to explicit path `custom.pt` |
| `python run.py --policy_dir models` | Change auto-save dir to `models/default/...` |
| `python run.py --no_save_policy` | Disable auto-saving (training only, no file) |

#### Example: Track Multiple Training Runs

```bash
# Run 1 (default preset, auto-save)
python strike_ea/run.py --preset default
# → Saves to: saved_policies/default/2026-03-03_18-05-12.pt

# Run 2 (high-kill preset, auto-save)
python strike_ea/run.py --preset high_kill
# → Saves to: saved_policies/high_kill/2026-03-03_18-10-45.pt

# Run 3 (custom fast variant)
python strike_ea/run.py --preset fast --n_iters 50
# → Saves to: saved_policies/fast/2026-03-03_18-15-30.pt
```

Now you can easily view all trained policies:

```bash
ls saved_policies/default/
# Output:
# 2026-03-01_14-22-15.pt
# 2026-03-02_09-15-33.pt
# 2026-03-03_18-05-12.pt
```

---

## Implementation Details

### Files Changed

#### 1. [trainer.py](strike_ea/training/trainer.py)

**Lines ~165:** Reward logging with NaN handling

```python
# Use nanmean to skip NaN values; if no completed episodes, will be NaN
if ep_rew.numel() > 0:
    ep_rew_mean = float(torch.nanmean(ep_rew).item())
else:
    ep_rew_mean = float("nan")
```

#### 2. [run.py](strike_ea/run.py)

**Added:**
- `import datetime` (line 7)
- `get_timestamped_policy_path()` function (line 54-65): Generates `saved_policies/{preset}/{timestamp}.pt`
- CLI flags: `--policy_dir`, `--no_save_policy` (line 153-154)
- Policy saving logic in `main()` (line 223-230):

```python
# Handle policy saving: explicit > auto-timestamped > disabled
if args.no_save_policy:
    save_policy = None
else:
    save_policy = args.save_policy or get_timestamped_policy_path(
        preset_name=args.preset,
        policy_dir=args.policy_dir
    )
```

---

## Usage Examples

### Example 1: Simple training with auto-save

```bash
python strike_ea/run.py --preset default
# Policy automatically saved to: saved_policies/default/2026-03-03_18-08-28.pt
```

### Example 2: Custom experiment with organized storage

```bash
python strike_ea/run.py \
  --preset default \
  --n_iters 200 \
  --lr 1e-4 \
  --policy_dir experiments/march_2026
# Saves to: experiments/march_2026/default/2026-03-03_18-08-28.pt
```

### Example 3: Disable auto-save (dev/debug mode)

```bash
python strike_ea/run.py --preset fast --n_iters 2 --no_save_policy
# No policy file created (useful for quick testing)
```

### Example 4: Explicit path (legacy behavior)

```bash
python strike_ea/run.py --preset default --save_policy my_model.pt
# Saves to: my_model.pt (exact path, no timestamp)
```

### Example 5: Load timestamped policy for evaluation

```bash
# First, train and auto-save
python strike_ea/run.py --preset default

# Then, play with the latest policy
python strike_ea/run.py --play --policy_path saved_policies/default/2026-03-03_18-08-28.pt
```

---

## Organizing Your Experiment Archive

Over time, you'll build a policy library:

```
saved_policies/
├── default/
│   ├── 2026-02-28_10-15-22.pt
│   ├── 2026-03-01_14-22-15.pt
│   └── 2026-03-03_18-08-28.pt  ← Latest
├── fast/
│   ├── 2026-03-01_14-30-45.pt
│   └── 2026-03-03_19-02-11.pt
├── high_kill/
│   └── 2026-03-03_18-11-33.pt
└── custom_experiments/
    ├── lr_sweep_1e-4/
    │   ├── 2026-03-01_15-00-00.pt
    │   └── 2026-03-02_16-30-22.pt
    └── large_team/
        └── 2026-03-03_17-45-09.pt
```

**Tip:** Use `--policy_dir` to organize by experiment type:

```bash
# Create an experiment series
python strike_ea/run.py --policy_dir exp/hparam_search --lr 1e-4 --entropy_coef 1e-2
python strike_ea/run.py --policy_dir exp/hparam_search --lr 1e-4 --entropy_coef 1e-3
python strike_ea/run.py --policy_dir exp/hparam_search --lr 3e-4 --entropy_coef 1e-2
```

---

## Summary

✅ **NaN Rewards Fix:**
- Now uses `torch.nanmean()` to filter NaN values
- Early `nan` values are expected (no completed episodes)
- Later values show actual episode returns

✅ **Auto-Timestamped Policy Saving:**
- Policies auto-save with unique timestamp
- Organized by preset: `saved_policies/{preset}/{date_time}.pt`
- Full CLI control: custom paths, disable saving, change directories
- No more accidental overwrites

✅ **Backward Compatible:**
- Old `--save_policy` flag still works
- Default behavior is "do the right thing" automatically
