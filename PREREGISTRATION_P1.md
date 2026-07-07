# PREREGISTRATION P1 — LLM Domain Study
## Aptadynamic LLM-LPM_PRAMA-Protokol · Observation interface and outcome declaration

**Discipline:** AS-1 §5, §8 / C5. This document is committed **before** any run on
outcome-labeled data at scale. Everything in it is frozen at commit time; amendments
after outcome exposure require a new study citing this one.

**Status of prior work:** the runs on synthetic data (2026-07) and any run predating
this commit are **exploratory** and are cited as design motivation, not evidence.

>  **AUTHOR DECISIONS D1–D6 REQUIRED BEFORE COMMITTING.** Proposed defaults are given.
> Fill, delete this block, commit.

---

## D1 — The system under evaluation
The system whose expected behavior Δ measures is the **deployed model** (population-level
causal expectation over previous sessions of the same model). Sessions are trajectories
*of* the system, not systems themselves.
[AUTHOR: confirm, or redefine as per-session with a within-session expectation — this
changes the entire O_D.]

## D2 — Observable stream and expectation (O_D)
- Event = token. ω = surprisal (−top1 logprob). Channels entropy/gap are ingested but
  NOT used in this study (reserved for a future declared extension).
- ω̂ = strictly causal per-position-bucket mean over previous sessions of the same model;
  buckets [0,8,16,32,64,128,256,512,1024,2048,4096]; warm-up min_sessions = 5.
- Empirical C3 record on synthetic data: PASS (r_Δω ≈ −0.03). C3 will be re-run and
  committed on the real stream before unblinding.
[AUTHOR: confirm bucket edges and min_sessions.]

## D3 — Kernel configuration and bin scale (the AS-1 gap, now with data)
AS-1 C5 fixes kernel parameters across domains *in bins* without defining bin scale.
This study declares: **bin = token**.

**Empirical finding (synthetic pipeline validation, 2026-07, pre-outcome):** the
validated grid configuration (tau_memory = 336, g_smooth = 24) is **mechanically
inapplicable** at K = 256 token windows — the accumulator's memory exceeds the
evaluation trajectory, so Ξ never reaches regime (latent AUC collapses to 0.22).
A tau × g_smooth sweep shows a **robust plateau at tau ∈ [16, 64]** (AUC 0.90–0.98
for every g_smooth ∈ {8, 16, 24}), collapsing for tau ≥ 128. The choice is a regime,
not a knife-edge — which is what distinguishes a scale declaration from retro-fitting.

**Declared configuration:** tau_memory = 64, g_smooth = 16 (on the plateau; all other
parameters at validated values: lambda_eq 1.0, lambda_recovery 0.005, lambda_min 0.1,
theta_scale 2.0, kappa 0.05). The validated 336/24 configuration will be run alongside
on the real stream as declared sensitivity, its mechanical failure documented.
[AUTHOR: confirm 64/16, or pick another plateau point — now, not after outcomes.]

**Flagged for AS-1 v1.1:** proposed "bin-scale declaration" clause — kernel parameters
are fixed in bins *given a declared bin scale satisfying tau_memory ≪ trajectory
length*; deviations from validated values must sit on a demonstrated robustness plateau
and be pre-registered.

## D4 — Evaluation window and inclusion
- Latent occupancy computed on the first K = 256 tokens of each session (valid rows only).
- Sessions with fewer than 32 valid tokens in the window are excluded (count reported).
[AUTHOR: confirm K and minimum.]

## D5 — Outcome schema (Y_o, Y_s)
- **Y_o (primary):** finish_reason = "length" (session fails to conclude) vs "stop".
  Aptadynamic reading: failure of the resolution phase (prescindence) — the session that
  cannot let go.
- **Confounder control (mandatory):** "length" conflates runaway degeneration with
  legitimately long tasks. Control: [AUTHOR: choose one — (a) stratify by declared prompt
  family; (b) generation protocol with a token cap far above the expected task length
  (e.g. 4× the P95 of stop-sessions per family); (c) manual audit of a random sample of
  N length-sessions to estimate the legitimate-long fraction].
- **Y_s (secondary, optional):** [AUTHOR: e.g. repetition rate in the final 20% of the
  session, as severity of the degeneration — or omit for this study.]
- **Boundary-proximity stratification (optional, H-IV hook):** [AUTHOR: include or defer.
  If included: declare the proximity proxy (e.g. refusal-probability score of the prompt)
  and the strata BEFORE unblinding.]

## D6 — Analysis, baselines, and success criteria
- Primary metric: AUC of latent-collapse occupancy for Y_o; ratio at top-quartile
  reported alongside.
- Mandatory baselines on the same K window, same causality: mean surprisal; mean entropy;
  mean top-1 gap; rolling Markovian surprisal intensity.
  [AUTHOR: confirm baseline list — additions allowed now, not later.]
- Null: label-permutation (1000 shuffles), threshold p < 0.01.
- **Success (confirmatory):** latent AUC exceeds the permutation null AND exceeds every
  baseline at matched false-positive rate.
- **Honest null:** signal present but not exceeding baselines → published as a negative
  result (ER-style).
- **Interface failure:** C3 fails on the real stream → the diagnosis is O_D, the study is
  invalid, a corrected interface opens a NEW study citing this one. The kernel is not
  touched.

## Commit order (all required)
1. This P1 (before large-scale outcome-labeled runs).
2. P2: thresholds/final baseline list if amended pre-unblinding.
3. P3: stream-only run + compliance record committed.
4. P4: unblinding marker.
5. P5: results + verdict, whatever they are.
