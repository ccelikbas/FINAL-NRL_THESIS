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

# ============================================================================
# PHASE 3 — strikers prefer a 2-JAMMER escort (κ=2). Same composition-agnostic logic.
# ============================================================================
Change: rewards.py escort_capacity 1 → 2 (κ). Everything else unchanged (escort-first w_s=0.35,
additive coverage, attraction w_a=0.4, target_cover κ_t=1/w_st=0.15, negative shaping). No code
change — the escort penalties already use κ generically.
Expected emergent behaviour (escort-first: striker wants 2 jammers > cover a 2nd target):
  • 2s4j → TWO formations of (1 striker + 2 jammers). N=6, two groups of 3 → frag≈0.60.
  • 2s2j → ONE clump of 4 (2 strikers share the 2 jammers; each ~2-covered when tight) → frag≈0.
  • 1s2j → ONE formation (1 striker + 2 jammers) → frag≈0.
  • 1s1j → ONE formation (1 striker + 1 jammer, under-escorted) → frag≈0.
TRAINING: ONE policy, --n_strikers 2 --dr_n_jammers 1 4 (sees scarcity→abundance), radar_kill=0.05,
n_iters 400 (harder: wider jammer range). Checkpoint runs/fofe_mappo_kappa2.pt.
EVAL: _diag_compare.py CONFIGS updated to [(2,4),(2,2),(1,2),(1,1)]; per-config via
_diag_softcommit.py --n_strikers/--n_jammers. NOTE frag target for two 3-agent formations is 0.60
(not 0.667).
OOM at 2048 envs: κ=2 → n_jammers=4 slots → 6 agents → centralized critic OOMs 8GB GPU at iter 4
(critic(td) wrapped in try/except → values None → transpose crash). expandable_segments unsupported
on Windows. FIX: num_envs 2048→1024 (mem ~halves), n_iters 400→600 (keep sample budget). Relaunched.
Checkpoint runs/fofe_mappo_kappa2.pt. Status: training (1024 envs, 600 iters).

## Phase 3 result @ 600 iters — STRUCTURE CORRECT but UNDER-TRAINED
Final: comp 0.80, surv 0.73, tgt 0.89 — and STILL CLIMBING at iter 600 (comp 0.66→0.80 over last
110 iters). κ=2 is harder (6 agents, 2-jammer coordination); same sample budget as κ=1 (which hit
comp 0.96) is not enough. Per-config eval (runs/diag_k2_*.png):
  • 2s4j: frag 0.258 (steady ~0.54), frac-diff-tgt 0.50 — strikers split to 2 targets, 4 jammers
    follow ~2-per-striker → TWO formations of (1s+2j) FORMING (structure correct, not fully clean).
  • 2s2j: frag 0.179 — mostly clump, some split (under-trained).
  • 1s2j: frag 0.038 — one formation (1s+2j) ✓.
  • 1s1j: frag 0.120 — mostly one pair, some split.
Structure is RIGHT (escort-first κ=2 → 2s4j makes two 1s+2j formations); just needs more training.
FIX: resume from checkpoint (run.py --load_checkpoint loads policy+critic+normalizer; fresh optim)
+600 more iters → runs/fofe_mappo_kappa2_ext.pt. Status: extending.

## Phase 3 result @ 1200 iters (extended) — κ=2 WORKS, composition-agnostic, not perfectly crisp
Final: comp 0.91 (up from 0.80; plateaued ~0.89-0.93 iters 530-600), surv 0.83, tgt 0.95.
Saved: runs/composition_agnostic_k2.pt. Per-config eval (runs/diag_k2e_*.png, side-by-side
runs/diag_compare_k2.png):
  • 2s4j: frag 0.263 (steady-state ~0.54), frac-diff-tgt 0.50 — TWO formations of (1s+2j): strikers
    split to 2 targets, 4 jammers split 2-per-striker. Correct structure; ~0.54 vs 0.60 ideal (jammers
    not perfectly tight 2-groups every seed).
  • 2s2j: frag 0.191 — one clump of 4 (2 strikers share 2 jammers), with some transient splitting.
  • 1s2j: frag 0.202 — one formation (1s+2j); a couple seeds a jammer wanders.
  • 1s1j: frag 0.039 — one pair (1s+1j) ✓.
VERDICT: escort_capacity κ=2 + escort-first + mixed DR(1,4) gives ONE policy that prefers 2-jammer
escorts and stays composition-agnostic (formations merge under jammer scarcity). Behaviours all
qualitatively correct. Looseness (jammer wander, surv 0.83) = κ=2 is a harder coordination task;
comp plateaued at 0.91 so more iters give marginal gains — crisper would need tuning (w_a/w_j/κ
balance or more iters) OR is the same jammer equilibrium-selection looseness as κ=1. Reported to user.

## USER CAUGHT a real bug: 2s4j does 1-3 jammer splits, NOT balanced 2-2.
frag KPI is BLIND to this: 2-2 (two groups of 3) → frag 0.600; 1-3 (groups of 2 and 4) → frag 0.533.
Only 0.067 apart — measured ~0.54 = the BAD 1-3 split. Added a proper metric to _diag_softcommit.py:
jammers/striker IMBALANCE (max-min nearest-assigned count; 0=even 2-2, 2=a 1-3) + balanced-frac.
Measured on κ=2 policy (2s4j): IMBALANCE 1.638, balanced only 17.8% → 1-3 splits dominate. Confirmed.

## Phase 3b — OVER-COVERAGE penalty to force balanced 2-2 (one more iteration)
ROOT CAUSE: escort only penalised UNDER-coverage (κ−c)₊; piling 3 jammers on one striker (c=3>κ=2)
cost nothing directly, and strong attraction (w_a=0.4) locked the unbalanced split.
FIX: add jammer OVER-coverage penalty −w_over·Σ_s(c_s−κ)₊ → balanced split (each striker exactly κ)
is now the unique optimum; a κ+1-th jammer on a striker is pushed to an under-served one.
  • rewards.py: NEW param escort_over_scale (w_over) = 0.3. Code: both env files, jammer escort now
    (−w_j·unmet − w_over·over)·j_alive. Doesn't affect 2s2j/1s2j (c≈κ, no over) or 1s1j (under).
TRAINING: resume from runs/fofe_mappo_kappa2_ext.pt (+over-penalty) +600 iters, 1024 envs, DR(1,4),
radar_kill 0.05 → runs/fofe_mappo_kappa2_bal.pt. Eval with the new IMBALANCE metric (want →0, balanced→1).
Status: training.

## Phase 3b RESULT — OVER-PENALTY FIXED THE BALANCE ✓ (note: 1st run hung overnight on a
## machine-sleep CUDA stall; killed + relaunched, ran clean)
Final: comp 0.93, surv 0.87 (BOTH up from 0.80-iter / 0.91-1200iter; balanced escorts protect
both strikers better). Saved runs/composition_agnostic_k2.pt (= fofe_mappo_kappa2_bal.pt).
Per-config (runs/diag_k2bal_*.png, side-by-side runs/diag_compare_k2bal.png):
  • 2s4j: IMBALANCE 0.000, BALANCED 1.000 (was 1.638 / 0.18!). frag-vs-step = EXACTLY 0.60 every
    seed → TWO formations of EXACTLY (1 striker + 2 jammers). 1-3 split ELIMINATED. ✓✓
  • 2s2j: IMBALANCE 0.000; mostly one clump (frag 0.17). ✓
  • 1s2j: one formation, slightly looser (frag 0.19, was 0.04) — minor over-penalty side effect.
  • 1s1j: one pair, slightly looser (frag 0.24). Minor.
VERDICT: adding the jammer over-coverage penalty −w_over·Σ(c−κ)₊ (escort_over_scale=0.3) makes the
balanced κ-per-striker split the unique optimum → 2s4j reliably 2-2. Composition-agnostic + balanced.
Minor: 1s configs a touch looser (over-penalty discourages over-tight clustering); reduce w_over or
add a small margin if tighter 1s formations wanted. KEY metric lesson: frag is blind to balance
(2-2=0.60 vs 1-3=0.533); the jammers/striker IMBALANCE metric in _diag_softcommit.py is the right gauge.

# ============================================================================
# PHASE 4 — does the SAME reward hold in scenario S2 (defensive line, radars BETWEEN
# agents and targets)? 6 radars, 2 targets. Want: 2s4j → two (1s+2j) formations;
# shortage → merge to one formation. Composition-agnostic + κ=2 balance, in S2.
# ============================================================================
Reward UNCHANGED from Phase 3b (κ=2, escort-first w_s=0.35, additive coverage, attraction w_a=0.4,
over-penalty w_over=0.3, target_cover κ_t=1/w_st=0.15, negative shaping). Only the SCENARIO changes.
Config via CLI (S1 defaults untouched): --scenario S2 --n_known_radars 6 --n_known_targets 2
--dr_n_jammers 1 4 --radar_kill_probability 0.05 --num_envs 1024.
Train FRESH (NOT resume): S2 critic global-state has 6 radars vs S1's 2 → critic dim differs, can't
load S1 weights. User: "might need double" iters (S2 harder). Staged: 600 first → runs/fofe_mappo_kappa2_s2.pt,
then resume +600 (=1200) if under-trained (protects against sleep-hang losing a long run).
Smoke (S2, 6 radars, 64 envs): OK, no spawn error, mem fine.
EVAL (unchanged scripts; they read S2 from the checkpoint env_cfg): per-config 2s4j/2s2j/1s2j/1s1j
with IMBALANCE metric + _diag_compare.py side-by-side. Status: training (600 fresh).

## Phase 4 result @ 600 iters — S2 TOO LETHAL TO LEARN FLAT (not a reward issue)
comp 0.10, surv 0.03, tgt 0.25 (S1 was 0.93/0.87). Trajectory: comp 0.01(it10)→0.10(it600), surv
~0.01-0.04 throughout — agents DIE almost immediately crossing the 6-radar defensive line, so the
escort/formation behaviour can't even manifest (frag ~0.03). It IS learning but VERY slowly.
ROOT CAUSE: S2 (6 radars BETWEEN agents and targets) at radar_kill=0.05 is far more lethal than S1
(2 radars guarding targets). Survival ~3% → almost no useful reward signal → slow/stuck learning.
This is scenario difficulty + exploration, NOT the escort reward (which is unchanged & worked in S1).
"double" (1200) is a big underestimate: the existing run_curriculum.py spends 1000+ iters on JUST the
first S2 stage — and with an EASIER 1-striker setup; ours is 2-striker + κ=2 (harder).
PAUSED — need user decision: (a) lower radar_kill for S2 (e.g. 0.02/0.01) to make it survivable so
the formations can be tested, then optionally ramp; (b) curriculum ramp (run_curriculum.py style);
(c) train MUCH longer at 0.05 (likely 3000+ iters, hours, may still struggle). Recommend (a) to first
verify the reward generalizes to S2 GEOMETRY, then add lethality.

USER chose (a): lower radar_kill. Plan: resume from runs/fofe_mappo_kappa2_s2.pt (has some S2
navigation) with radar_kill 0.05→0.02, S2/6 radars/2 targets, DR(1,4), 1024 envs, +800 iters →
runs/fofe_mappo_kappa2_s2lk.pt. Expect survival to jump → formations can manifest → test escort
reward in S2 geometry (2s4j two 1s+2j formations, shortage→merge, IMBALANCE→0). If survival still
low after, drop to 0.01. Status: training.

## Phase 4 result @ S2 radar_kill=0.02 (800 iters resumed) — TEAM CLUMPS, DOES NOT SPLIT
(NB: 2nd overnight hang on the 1st attempt; killed+relaunched, ran clean.)
comp 0.80, surv 0.61 (vs 0.10/0.03 at kill=0.05 — lower lethality FIXED survivability ✓), STILL RISING.
Per-config eval (runs/diag_k2s2lk_*.png, side-by-side runs/diag_compare_k2s2lk.png):
  • 2s4j: frac-diff-targets 0.000 (strikers NEVER split), frag 0.07 → ONE clumped group of 6.
    IMBALANCE 2.05 is an artifact (metric assumes split strikers; here they're together).
  • 2s2j: frac-diff 0.000, frag 0.02 → one clump.
  • 1s2j / 1s1j: one formation (as before).
So escort still works (jammers stay with strikers) but the STRIKER SPLIT (target_cover) does NOT emerge
in S2 — the whole team moves as ONE group up through the radar field. NOT the 2-formation goal.
WHY (3 candidates): (1) survival-driven — splitting to 2 targets through the 6-radar defensive line is
risky; clumping concentrates jamming → safer (clump may be genuinely optimal in S2); (2) under-trained
(comp/surv still rising at 800); (3) resume-bias — resumed from the kill=0.05 policy that learned to
clump-to-survive. PAUSED — report to user; options: train fresh/longer at 0.02 to see if split emerges;
boost target_cover (w_st) to force splitting despite risk; or accept S2 clumping as correct adaptation.
