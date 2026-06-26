# JAMMER ESCORT fix — /loop experiment log

Goal: two striker+jammer formations, each committing to a different target.
Metric: eval coalition fragmentation (frag). N=4 → one blob F=0; two pairs F=2/3≈0.667 (instantaneous), episode-mean lower.
Success: clear improvement over baseline (frag≈0.00) AND rollouts visibly show two pairs splitting to two targets.

Fixed setup (per user): legacy MLP (NO --use_fofe), HF radar ON, n_strikers=2, n_jammers=2, num_envs=2048, n_iters=50.
Launch: `python "0_TA_HF_FOFE-MAPPO/train_mappo.py" --n_strikers 2 --n_jammers 2 --num_envs 2048 --n_iters 50 --no_animate --no_plot --save_name <name>.pt`
Each iter saves to runs/<name>.pt and logs to runs/<name>_{out,err}.log. Never overwrite runs/fofe_mappo.pt.

## Baseline (user-provided)
escort ℓ=0.05, κ=1, w_s=0.1, w_j=0.1; striker_sep/jammer_sep OFF.
Result: iter50 frag **0.00** — single blob, group moves to one target then the other. NOT desired.

## CONSTRAINT (user correction, do not violate)
Fix must live ENTIRELY in the JAMMER ESCORT reward. Do NOT enable striker/jammer
same-role separation or any other term. Reason: a separation penalty hard-codes
"strikers must be apart", which breaks generalization to complex scenarios (e.g.
excess strikers, where 2 strikers sharing one formation/escort must be allowed).
Shared/additive coverage must be PRESERVED so 1 jammer can still escort a tight
striker cluster. The escort fix must be emergent/geometry-driven, scenario-independent.

## Iteration 1 — striker separation — REVERTED (violated constraint above)
Was: striker_sep enabled. Rejected by user before training. Reverted to 0.

## Open design tension (needs user direction)
Key insight: with shared coverage, when both strikers sit on the SAME target each
still has a jammer → escort term is already maxed (J=2κ) → no internal gradient to
separate them. So escort-only cannot, by itself, pull co-located strikers onto
different targets; striker target-separation must come from target-approach.
Question to user: should the escort fix only make jammers COMMIT (accompany one
striker, killing the straddle), with striker split emerging from targets — or do
they expect escort-only to break the blob? + which escort-internal mechanism.

USER ANSWERS: (1) Escort fix = commit jammers; striker target-split comes from the
existing target-approach reward. (2) Mechanism = soft-commit jammer credit (softmax τ).

## Iteration 2 — soft-commit jammer credit (softmax τ)  [escort-only]
Structural change INSIDE the escort reward (environment.py + HF_environment.py):
each jammer = one unit of escort, softly assigned over strikers by a_sj=softmax_s(−d_sj/τ)
(alive strikers only); committed contribution kc_sj=a_sj·k_sj; c_s=Σ_j kc_sj; marginal
uses kc_sj. Kills the straddle at any ℓ (midpoint delivers ≤1 total → can't saturate
both strikers), preserves shared coverage for tight clusters (within ~τ). Recovers
original additive coverage as τ→∞ and original single-striker behaviour for ns=1.
New param: escort_commit_temp (τ).
Params this run: ℓ=0.15 (raised from 0.05 — softmax lets ℓ be long-range for a real
commit-from-range gradient), τ=0.10, κ=1, w_s=0.1, w_j=0.1. No other rewards touched.
Run env fix: console needs UTF-8 → prefix `$env:PYTHONIOENCODING="utf-8"` (cp1252 chokes
on ≥ in a triton-missing warning; triton absent on Windows → compile falls back to eager).
Checkpoint: runs/fofe_mappo_softcommit.pt; logs runs/softcommit_{out,err}.log
Smoke test (64 envs, 1 iter): OK, no shape/NaN errors.
Result: TRAINED 50 iters @ 2048 envs (runs/fofe_mappo_softcommit.pt).
  Training frag: iter10 0.01, iter30 0.01, iter40 0.03, iter50 0.01 (eval frag).
  Task perf improved: comp 0.93, tgt 0.95, surv 0.91, eval_ret +8.9.
  Diagnostic (_diag_softcommit.py, 6 seeds → runs/diag_softcommit.png):
    episode-mean frag = 0.017 (target ~0.667 for two pairs)
    mean striker-striker distance = 0.058 (coalition_radius=0.2) → strikers GLUED
    frac of time strikers nearest DIFFERENT targets = 0.000 → always same target
  VERDICT: soft-commit kills the straddle as designed (jammers escort, don't sit
  between strikers), BUT the two strikers never separate → still one blob (frag≈0).

## CONCLUSION — escort-only cannot break the blob (structural)
Evidence + analysis agree: when both strikers sit on the SAME target each still has
a jammer, so the escort term is already maxed (J=2κ) and has ZERO gradient w.r.t.
inter-pair separation — by design (shared coverage, for generalization). No escort
parameter (τ, ℓ, κ, w_*) changes this; the strikers are glued by TARGET choice, not
by escort. Two formations therefore require a striker→target distribution mechanism.
Recommended next (NEEDS USER OK — outside escort): a striker→target COVERAGE FIELD,
the exact mirror of the escort (targets demand κ_t strikers; strikers earn the
saturating difference reward). Emergent + scenario-independent (NOT a separation
penalty): generalizes to any #targets/#strikers, and lets strikers cluster when
targets are few — same property the escort gives jammers. Soft-commit escort change
is a genuine improvement (straddle fix) and is kept regardless.
STOPPED autonomous escort iteration here per "stop if it's really not working".

USER APPROVED (after report): (1) add striker→target coverage field; (2) keep soft-commit.

## Iteration 3 — striker→target coverage field (mirror of escort) + soft-commit kept
New reward term in BOTH env files (sections "8c"), the exact twin of the escort:
each striker = one unit of attack mass, soft-committed to nearest target
b_ts=softmax_t(−d_ts/τ_t); C_t=Σ_s b_ts·exp(−d_ts/ℓ_t); G=Σ_t min(C_t,κ_t);
striker reward r_s=+w_st·(G−G_{−s}). κ_t cap → strikers distribute one-per-target
(emergent, scenario-independent; runs alongside striker_approach which gives the
long-range pull). New params (rewards.py): target_cover_kernel_length=0.15 (ℓ_t),
target_cover_capacity=1 (κ_t), target_cover_scale=0.1 (w_st), target_cover_commit_temp=0.1 (τ_t).
Plumbing: added "target_cover" to env _episode_component_reward init (environment.py),
both last_reward_components dicts, and trainer EVAL_REWARD_COMPONENT_KEYS.
Escort unchanged from iter2 (soft-commit, ℓ=0.15, τ=0.1, κ=1, w_s=w_j=0.1).
Checkpoint: runs/fofe_mappo_targetcover.pt; logs runs/targetcover_{out,err}.log
Smoke test (64 envs,1 iter): OK.
Result: TRAINED (runs/fofe_mappo_targetcover.pt). frag iter10-50: 0.04,0.05,0.04,0.03,0.02.
  Task perf strong (comp 0.99, tgt 0.99, surv 0.92, dur 69).
  Diagnostic (runs/diag_targetcover.png, 6 seeds): episode-mean frag=0.015,
  striker-striker dist=0.043 (GLUED), frac diff-targets=0.000. NO separation.
  Why: ℓ_t=0.15 too local → distribution signal only bites point-blank, after both
  strikers already committed to same nearest target; plain striker_approach pulls both
  to same target everywhere. w_st=0.1 can't compete at range.

## Iteration 4 — long-range + stronger target cover field (param-only)
Change (rewards.py): ℓ_t 0.15→0.4 (long-range gradient toward uncovered targets),
κ_t 1→0.5 (target claimed from range → frees 2nd striker earlier), w_st 0.1→0.25
(compete with approach), τ_t 0.1→0.2. Escort + soft-commit untouched. Code unchanged
(param-only) → no smoke test needed.
Checkpoint: runs/fofe_mappo_targetcover2.pt; logs runs/targetcover2_{out,err}.log
Result: REWARD-HACKED. comp 0.99→0.01, dur→199 (never finishes), eval_ret↑44.6 (farmed),
  frag 0.00, striker-striker dist 0.018. Diagnostic (runs/diag_targetcover2.png): all 4
  agents orbit a point ABOVE/BETWEEN the two targets, farming positive coverage reward,
  never engaging. κ_t=0.5 + ℓ_t=0.4 made a single striker at the target-midpoint saturate
  BOTH half-demands → straddle-farm (the escort straddle, mirrored onto targets).

## KEY USER GUIDANCE (general principle for all shaping here)
Shaping rewards should be NEGATIVE, = 0 (least negative) at the desired behaviour. The
codebase's existing terms (approach/border/radar) are all negative penalties = 0 at ideal,
so the best an agent can do is 0 — nothing to FARM. My escort jammer reward and the
target-cover reward were POSITIVE (coverage provided) → farmable → the iter4 hack.

## Iteration 5 — NEGATIVE unmet-demand shaping for BOTH coverage fields
Reformulate (both env files), keep soft-commit + κ caps:
  • Escort jammer: was +w_j·(J−J_{-j}); now r_j = −w_j·Σ_s (κ−c_s)+  (team unmet striker
    escort; 0 when all strikers escorted). c_s still soft-committed.
  • Target cover striker: was +w_st·(G−G_{-s}); now r_s = −w_st·Σ_t (κ_t−C_t)+ (team unmet
    target coverage over ALIVE targets; 0 when all targets covered). C_t still soft-committed.
Both are negative, farm-proof (best=0 = fully distributed/covered), and the κ cap + soft-commit
make U=0 require DISTRIBUTION (one per target/striker). The blob becomes actively penalised
(unmet>0) instead of neutral — should finally push the split. Loitering at midpoint leaves
targets uncovered → penalised → strikers driven ONTO targets (→ engage/destroy preserved).
Params: target cover κ_t 0.5→1, ℓ_t 0.4→0.3, w_st 0.25→0.15, τ_t 0.2. Escort unchanged
(ℓ=0.15,κ=1,w_s=w_j=0.1,τ=0.1). Checkpoint: runs/fofe_mappo_negshape.pt
Result: BIG IMPROVEMENT, no hack. comp 1.00, tgt 1.00, surv 0.95, dur 37 (efficient).
  frag iter10-50: 0.10,0.15,0.16,0.16,0.26 (rising). Diagnostic (runs/diag_negshape.png):
  episode-mean frag=0.212, striker-striker dist=0.239 (>0.2!), frac diff-targets=0.265.
  → STRIKERS NOW SPLIT to two different targets (target-cover field works!). frag plateaus
  at 0.5 = "3 together + 1 alone": both jammers escort striker1, striker0 goes ALONE.
  So jammers don't distribute — same RANGE problem as targets had: escort ℓ=0.15 too short
  for a jammer to feel a striker that peeled off ~0.5 away to a far target (k(0.5)=0.04).

## Iteration 6 — lengthen escort range so jammers follow separated strikers (param-only)
Change (rewards.py): escort_kernel_length 0.15→0.3 (k(0.5)=0.19, real gradient to a far
striker). Negative escort already makes leaving striker0 unescorted a penalty (U=κ); it
just needs range to act on. Everything else unchanged (negative shaping, soft-commit τ=0.1,
κ=1, w_s=w_j=0.1; target cover ℓ_t=0.3,κ_t=1,w_st=0.15,τ_t=0.2). Code unchanged → no smoke.
Checkpoint: runs/fofe_mappo_negshape2.pt; logs runs/negshape2_{out,err}.log
Result: WORSE for frag. comp 0.98 (fine), frag iter50 0.17; diagnostic episode-mean frag=0.123
  (down from 0.21), striker-striker dist 0.214 (strikers still split same as iter5), frac
  diff-targets 0.257. Figure: strikers split (blue→left tgt, cyan→right tgt) but BOTH jammers
  follow the RIGHT striker (still "3+1"); longer range let jammer0 trail toward middle → bridges
  pairs → lower frag. CONCLUSION: longer escort range = jammers hover between → worse. Shorter
  (iter5 ℓ=0.15, frag 0.26) is better. Core issue unchanged: both jammers commit to ONE striker.
  NOTE: jammer_approach is DISABLED → escort is the jammers' dominant positioning signal, so
  strengthening escort distribution should fix the jammer split.

## Iteration 7 — revert escort range, strengthen escort distribution (param-only)
Change (rewards.py): escort_kernel_length 0.3→0.15 (back to best), escort_jammer_scale 0.1→0.25
(stronger pull so ONE jammer commits to the lone unescorted striker). Single-variable from iter5.
Everything else unchanged. Checkpoint: runs/fofe_mappo_negshape3.pt; logs runs/negshape3_{out,err}.log
Result: comp 0.99, frag iter50 0.24. Diagnostic episode-mean frag=0.238, striker-striker
  dist 0.244, frac diff-targets 0.311. Figure: SOME seeds now reach frag 0.83-1.0 (jammers
  DO split into pairs) but others still plateau 0.5 (3+1). So stronger w_j made the jammer
  split happen SOMETIMES but not reliably. Marginal mean improvement over iter5 (0.21→0.24).

## DIAGNOSIS of residual (jammer 1-1 distribution)
The escort's shared negative-U penalty gives BOTH (param-shared) jammers the same signal —
it doesn't break the symmetry of WHICH jammer escorts WHICH striker. Strikers split reliably
because they have DISTINCT spatial anchors (the two fixed targets) + the negative target-cover
field; the jammers have no equivalent distinct anchors (escort alone treats both strikers
symmetrically). Result: jammers either clump (3+1, frag 0.5) or over-disperse (frag→1.0), not
clean tight pairs (0.667). jammer_approach is DISABLED so the escort is doing all jammer
positioning.

## RECOMMENDATION (paused for user decision)
Add a JAMMER→RADAR coverage field = mirror of the striker→target field (negative shaping,
soft-commit, κ_r): jammers distribute one-per-radar. Each target has a radar; each striker
goes to a target; so a jammer assigned to that target's radar pairs with that striker →
consistent two formations. It's the proper jammer→radar distribution mechanism (replacing the
disabled jammer_approach) and the scenario-independent analog of the approved target-cover.
Alternatives: keep tuning escort (diminishing returns — it's a symmetry/coordination issue),
or stop (strikers split = main win, jammers partially escort).

## frag progression (episode-mean, diagnostic): 
orig 0.015 → soft-commit 0.015 → target-cover w0.1 0.015 → long-range(hacked) 0.000 →
NEG-SHAPE 0.212 (strikers split!) → +escort range 0.123 → +escort w_j 0.238.

## USER REJECTED radar-coverage idea. Must SCALE to variable/non-symmetric (4-10) configs.
"keep tuning escort OR adapt the reward STRUCTURE of this specific reward type. Go back to
basics, make a plan." Radar field wrong (breaks with variable radar count). Anchor must be
STRIKERS, not radars. This 2s2j case is just the playground for a scalable escort.

## ROOT CAUSE (back to basics)
Why do STRIKERS distribute reliably but JAMMERS don't? Strikers have TWO terms:
  (1) striker_approach = LONG-RANGE soft-nearest pull to targets (d_max=1, full map) — lets a
      striker REACH a far target; (2) target_cover (neg-U, κ cap) — decides WHICH target.
Together: approach gets them to targets, cover distributes (2nd striker → the UNCOVERED far
target, which it can reach because approach is long-range).
JAMMERS have only the escort (ONE exp kernel doing both jobs) and NO long-range attraction
(jammer_approach is disabled). A single kernel can't be both long-range (reach a far striker)
AND short-range-credit (no hovering): short ℓ → can't feel a striker that split far → both
clump on the near one; long ℓ → cover both from between → bridge/hover. That tension is the
whole problem.

## PLAN (restructure the escort to mirror the proven striker→target structure)
Split the escort's jammer side into the SAME two parts that make strikers scale:
  (A) jammer→striker ATTRACTION: negative, LONG-RANGE, soft-nearest over strikers, piecewise
      lin-exp, 0 at d=0 (exact analog of striker_approach→targets). Gives a jammer the pull to
      REACH a striker anywhere on the map. Pure jammer↔striker (scales with counts; no radars).
  (B) escort COVERAGE distribution: keep the negative-U with κ cap + soft-commit (r_j =
      −w_cov·Σ_s(κ−c_s)₊). Decides WHICH striker (push the redundant jammer to an UNDER-served
      one). Tune w_cov ≥ w_attract (like target_cover ≥ striker_approach) so the 2nd jammer
      takes the uncovered striker, while attraction lets it travel there.
Scales to non-symmetric (ns,nj): every jammer is attracted to strikers and the cap prevents
redundancy — identical structure to strikers→targets, which already generalises. κ keeps the
"jammers per striker" knob. STATUS: plan proposed, awaiting user go-ahead.

## Iteration 8 — IMPLEMENTED the plan (escort = attraction + coverage), during /loop
Proceeded autonomously (loop firing + user asked for a plan "to make this work"; reversible,
within constraints). Changes:
  • rewards.py: NEW param jammer_escort_approach_scale (w_a)=0.1; escort_kernel_length 0.15→0.3.
    (Coverage needs ℓ=0.3 so a jammer FEELS a far under-served striker — same reason target_cover
    works at ℓ_t=0.3; the new attraction stops the long-range hovering that ℓ=0.3 alone caused.)
  • environment.py + HF_environment.py escort block: added jammer→striker ATTRACTION =
    −w_a·(soft-nearest distance to an alive striker)·alive, folded into jammer_escort (and
    escort_full). Guard now includes w_a. Coverage term (−w_j·Σ_s(κ−c_s)₊) unchanged; w_j=0.25 ≥ w_a=0.1.
Mechanism: attraction = REACH + commit ONTO a striker (no hovering); coverage (ℓ=0.3, κ cap) =
WHICH striker (feels far under-served one). Mirrors striker_approach + target_cover exactly.
Checkpoint: runs/fofe_mappo_escortv2.pt; logs runs/escortv2_{out,err}.log
Result: frag 0.18 (worse), comp 0.98. Diagnostic frag=0.140, striker dist 0.261 (strikers
  split fine), frac diff-tgt 0.276. Figure: BOTH jammers still go right with striker1 (3+1).
  ROOT CAUSE found: hard soft-commit τ=0.1 makes a jammer committed to striker1 have a_0≈0.007
  → contributes ~nothing to striker0 → the coverage gradient that should pull it to the far
  under-served striker is SUPPRESSED. Attraction made it worse (reinforced go-to-nearest).

## Iteration 9 — REMOVE soft-commit → additive coverage (keep attraction + cap)
Change (both env files): c_s = Σ_j k_sj (additive) instead of soft-commit kc_sj. Now a jammer
at striker1 contributes k(0.5)=0.19 to striker0 (ℓ=0.3) → un-suppressed gradient to go cover it;
the ATTRACTION (w_a=0.1) prevents the straddle/hover that additive used to allow. Bonus: additive
restores the "1 jammer protects a tight striker cluster" generalization. (tau/escort_commit_temp
now unused.) Everything else unchanged: ℓ=0.3, κ=1, w_s=0.1, w_j=0.25, w_a=0.1, negative shaping.
Checkpoint: runs/fofe_mappo_escortv3.pt; logs runs/escortv3_{out,err}.log
Result: frag 0.17, comp 0.99. Diagnostic frag=0.164, striker dist 0.242, frac diff-tgt 0.381
  (strikers split BEST yet). Figure: both jammers sit in the MIDDLE between the two split
  strikers (straddle), covering both additively. Some seeds reach 0.83 (split happens).

## CORE DIAGNOSIS — jammer split is an EQUILIBRIUM-SELECTION / symmetry problem (not reward shape)
Reward CHECK (additive, ℓ=0.3, 2j, strikers 0.5 apart):
  • both jammers in middle: U=0.28 → escort ≈ −w_j·0.28 − w_a·0.25 = −0.095 each.
  • one jammer per striker:  U=0    → escort = 0 each.
So the escort ALREADY makes two-pairs the reward optimum (0 > −0.095). Yet the parameter-shared
policy converges to the SYMMETRIC middle (a worse but symmetric equilibrium). A permutation-
symmetric (=scalable) reward inherently has a symmetric critical point at the middle, so reward
shaping alone can't *force* the break. STRIKERS escape it because TARGETS are distinct FIXED
anchors (each striker is nearer a different one from the start); JAMMERS' anchors (strikers)
start co-located and only separate later, so the jammers settle symmetric before they can pair.
Status of escort: RESTRUCTURED & correct (attraction + additive coverage + negative shaping,
jammer↔striker only, scalable) — two-pairs is reward-optimal; the gap is policy/optimization.

## frag (episode-mean): best ~0.24 (iter5/7). Structural changes (attraction iter8, additive
iter9) made two-pairs OPTIMAL & occur in some seeds (→0.83) but didn't raise the MEAN (still
symmetric-trap). comp stayed ~0.99 throughout (negative shaping = no hack).

## OPTIONS for user (paused, check-in):
(1) Stronger attraction w_a (jammers TRACK nearest striker through the split; may pair reliably
    if initial jammer positions are generically asymmetric) — one more reward tweak, scalable.
(2) Policy/architecture symmetry-break: agent-ID embeddings, higher entropy_coef, or enable FOFE
    (permutation-invariant) — addresses the root (symmetric policy), OUTSIDE the reward.
(3) Accept: strikers split reliably + escort structurally correct; move on to scaling tests.

## Iteration 10 — USER chose: stronger attraction (param-only)
Change (rewards.py): jammer_escort_approach_scale 0.1→0.4 (now > w_j=0.25). Strongly pulls each
jammer ONTO its nearest striker → breaks the symmetric MIDDLE straddle, and tracks the nearest
striker through the split (relies on distinct jammer spawn positions to pair). Everything else
unchanged: additive coverage, ℓ=0.3, κ=1, w_s=0.1, w_j=0.25, negative shaping.
Checkpoint: runs/fofe_mappo_escortv4.pt; logs runs/escortv4_{out,err}.log
Result: frag 0.17→diag 0.115 (WORSE), comp 0.99. Both jammers STILL commit to the same striker
(consistently the right one); seed 1000 one jammer just loops in the middle. Stronger attraction
did NOT break the symmetry.

## CONCLUSION (reward tuning exhausted)
Confirmed: the jammer 1-1 assignment is an EQUILIBRIUM-SELECTION / symmetry-breaking problem that
reward tuning cannot fix. The escort is structurally correct & scalable (attraction + additive
negative coverage, jammer↔striker only; two-pairs is the reward OPTIMUM, comp stays ~0.99). The
parameter-shared policy collapses both jammers onto one striker because their anchors (strikers)
start co-located — unlike strikers, which split via the distinct FIXED target anchors. This needs
a POLICY/ARCHITECTURE symmetry-break, not more reward shaping:
  • agent-ID / heterogeneous jammer policy (let the 2 jammers specialise), or
  • higher entropy_coef (explore the asymmetric equilibrium), or
  • enable FOFE permutation-invariant encoder.
OR accept (escort correct & scalable; strikers split) and move to the larger non-symmetric tests.
Best frag achieved: episode-mean ~0.24 (iter5/7); two-pairs reached in some seeds (panels →0.83-1.0).
PAUSED for user decision.

RESOLUTION (Phase 1): user increased radar kill rate + iterations (n_iters default→200) → the
2s2j two-formation case now WORKS. Higher radar lethality makes an unescorted striker actually
die, supplying the symmetry-breaking the symmetric reward couldn't. Escort reward kept as-is.

# ============================================================================
# PHASE 2 — composition-agnostic: SAME reward must give 2 formations (2s2j) AND
# one clump (2s1j). One policy, trained over mixed n_jammers, adapts on observation.
# ============================================================================
Failure: trained directly on 2s1j, strikers still SPLIT (one ends up unescorted) instead of
clumping around the single jammer. Confirmed by user.
DIAGNOSIS: reward rated an uncovered target (w_st=0.15) worse than an unescorted striker
(w_s=0.1) → strikers split to cover both targets even when one can't be escorted.
FIX (user-approved "escort-first priority"): set w_s > w_st so an UNESCORTED striker is always
worse than an UNCOVERED target → strikers spread only as far as jammers can escort.
  • rewards.py: escort_striker_scale 0.1 → 0.35 (now > w_st=0.15; note 2 strikers sharing 1
    jammer still carries residual unmet, so margin matters — threshold est. ~0.29).
  • Reinforced by radar lethality: an unescorted split striker dies → big mission penalty.
TRAINING (user-approved): go straight to MIXED domain-randomized training, ONE policy.
  • run.py: added --dr_n_jammers LO HI (DomainRandomization over ACTIVE jammers; allocates HI
    slots) and --radar_kill_probability CLI flag. build_env already plumbs env_cfg.dr; DR-absent
    jammers are correctly not-alive (env.py:1176).
  • Plan: train with --n_strikers 2 --dr_n_jammers 1 2 (mix 1j/2j), S1, 2 targets/radars.
EVAL: legacy MLP actor obs is fixed-slot (n_other_agent_obs_slots), so the SAME policy can be
rolled out at fixed n_jammers=1 and n_jammers=2. Check: 2s1j → frag≈0 (clump of 3), 2s2j →
frag≈0.667 (two pairs). (May need to add --n_jammers override to _diag_softcommit.py.)
USER VALUES: radar_kill=0.05 (was 0.01 before; 0.05 is the working value), n_iters=300.

## PHASE 2 RESULT — SUCCESS, composition-agnostic ✓
Mixed-composition training: ONE policy, escort-first reward (w_s=0.35 > w_st=0.15), DR
n_jammers∈{1,2}, radar_kill=0.05, 300 iters → runs/fofe_mappo_mixcomp.pt.
Training (mixed episodes): comp 0.96, surv 0.93, tgt 0.98 (climbing the whole run).
Per-composition eval (SAME policy, _diag_softcommit.py --n_jammers k):
  • 2s1j (runs/diag_mix_1j.png): episode-mean frag=0.061, striker-striker dist=0.065,
    frac-diff-tgt=0.10. 5/6 seeds frag stays 0 the whole episode → ALL 3 AGENTS CLUMP
    (both strikers + the lone jammer fly as one tight group to a target). ✓ DESIRED.
  • 2s2j (runs/diag_mix_2j.png): episode-mean frag=0.350, frac-diff-tgt=0.41. Every seed:
    strikers split to the 2 targets, EACH with its own jammer → frag-vs-step reaches exactly
    0.667 (two pairs) in steady state and holds. Mean diluted only by the shared approach phase.
    ✓ DESIRED — and cleaner/more consistent than Phase 1 (1j training reinforced escort commit).
CONCLUSION: escort-first priority (w_s>w_st) + mixed DR training gives ONE policy that, from
observation alone, forms TWO formations with 2 jammers and ONE clump with 1 jammer. The reward
is now composition-agnostic for the 2s/{1,2}j case. DONE.

## MULTI-CONFIG VALIDATION (same policy → runs/composition_agnostic.pt = copy of mixcomp)
Added _diag_softcommit.py --n_strikers override; built _diag_compare.py (side-by-side 4-config).
Evaluated ONE policy (6 seeds each) — figures runs/diag_cfg_{2s2j,2s1j,1s1j,1s2j}.png and the
side-by-side runs/diag_compare_4cfg.png:
  • 2s2j: mean frag 0.350 (steady-state 0.667) — strikers split, EACH with a jammer → 2 formations ✓
  • 2s1j: mean frag 0.061 — all 3 clump (share lone jammer) → 1 formation ✓
  • 1s1j: mean frag 0.000 — striker+jammer pair travel together → 1 formation ✓ (NEVER trained on 1 striker)
  • 1s2j: mean frag 0.034 — striker + both jammers travel together (surplus jammer stays) → 1 formation ✓ (untrained)
ALL MATCH EXPECTATION. Policy even GENERALISES to 1-striker compositions it never saw in training
(only n_jammers was domain-randomised; n_strikers fixed at 2). Legacy per-role actor obs is
fixed-slot so the same weights roll out at any (ns,nj). Composition-agnostic CONFIRMED.
