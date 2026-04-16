//! Prediction-market solve entry point with `B`-doubling + Moreau-Yosida
//! `μ` annealing + 4-step bisection for the mixed (split/merge enabled) mode.
//!
//! Julia parity target: `_solve_prediction_market_mixed`
//! (`src/prediction_market_api.jl:1302`). The direct-only path is a single
//! `BoundedDualProblem::solve_and_certify` call with no doubling.
//!
//! Warm-start across iterations: each `B` rebuilds the `BoundedDualProblem`
//! from scratch, but the converged ν from the most recent *certified* inner
//! solve is promoted as the seed for the next doubling (and for the first
//! bisection midpoint). Mirrors Julia `prediction_market_api.jl:1398`
//! (`ν_seed = ν_candidate`) and `:1383` (`ν_bisect = ν_mid`). The seed is
//! clamped into the new box by `BoundedDualProblem::clamp_seed` before it
//! reaches the C driver so a ν feasible at `B` stays feasible at `2·B`.

use forecast_flows_core::{
    BoundedDualProblem, CertifyTolerances, DualSolution, Objective, SolveCertificate, SolverOptions,
};
// CertifyTolerances stays in the import list because `derive_tols` still
// returns it; the explicit `opts.tolerances` field was removed so every
// entry point derives tolerances from `pgtol` the way Julia does.

/// Julia parity: `_solve_prediction_market_once` and
/// `_solve_prediction_market_mixed` both derive the certify tolerances from
/// `pgtol` at the top of the solve, not from a caller-provided knob. Mirror
/// that here so tightening `pgtol` in the test fixtures also tightens the
/// certificate — the fixed `CertifyTolerances::default()` floor was masking
/// the parity fixtures by rejecting sub-1e-6 residuals that Julia accepts.
#[inline]
fn derive_tols(solver: &SolverOptions) -> CertifyTolerances {
    CertifyTolerances::from_pgtol(solver.pgtol)
}

use crate::problem::{
    Mode, PredictionMarketProblem, ProblemError, analytical_split_bound, build_edges,
    build_objective, default_split_bound,
};

#[derive(Debug, thiserror::Error)]
pub enum SolveError {
    #[error(transparent)]
    Problem(#[from] ProblemError),
    #[error(transparent)]
    Core(#[from] forecast_flows_core::SolveError),
    #[error("mixed solve never certified across {0} split_bound levels")]
    NeverCertified(usize),
}

/// Knobs for the public solve entry point. Defaults mirror Julia's
/// `solve_prediction_market` keyword defaults.
#[derive(Debug, Clone, Copy)]
pub struct SolveOptions {
    pub mode: Mode,
    pub max_doublings: usize,
    pub solver: SolverOptions,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            mode: Mode::DirectOnly,
            max_doublings: 6,
            solver: SolverOptions::default(),
        }
    }
}

/// Outcome of a successful solve: the certified dual solution plus the
/// realized `split_bound` (matches the input bound for direct-only solves).
#[derive(Debug, Clone)]
pub struct SolveOutcome {
    pub solution: DualSolution,
    pub certificate: SolveCertificate,
    pub split_bound: f64,
}

/// Top-level solve entry point. Direct-only solves take a single
/// `BoundedDualProblem::solve_and_certify` pass. Mixed-enabled solves run
/// the doubling + bisection loop from `_solve_prediction_market_mixed`.
pub fn solve(
    problem: &PredictionMarketProblem,
    opts: SolveOptions,
) -> Result<SolveOutcome, SolveError> {
    solve_with_seed(problem, opts, None)
}

/// `solve` variant that threads a caller-provided dual seed into the
/// underlying `BoundedDualProblem`. For `Mode::MixedEnabled`, the seed only
/// seeds the *first* inner solve; subsequent doubling iterations (and the
/// bisection midpoints) warm-start from the ν of the most recent certified
/// inner solve. Mirrors Julia `_solve_prediction_market_mixed` at
/// `prediction_market_api.jl:1322` (`ν_seed = workspace.direct_seed`) →
/// `:1398` (`ν_seed = ν_candidate` after each certified outer solve) →
/// `:1383` (`ν_bisect = ν_mid` after each certified bisection midpoint).
pub fn solve_with_seed(
    problem: &PredictionMarketProblem,
    opts: SolveOptions,
    nu_seed: Option<&[f64]>,
) -> Result<SolveOutcome, SolveError> {
    match opts.mode {
        Mode::DirectOnly => solve_direct_only(problem, opts, nu_seed),
        Mode::MixedEnabled => solve_mixed_with_doubling(problem, opts, nu_seed),
    }
}

fn solve_direct_only(
    problem: &PredictionMarketProblem,
    opts: SolveOptions,
    nu_seed: Option<&[f64]>,
) -> Result<SolveOutcome, SolveError> {
    let obj = build_objective(problem)?;
    let edges = build_edges(problem, Mode::DirectOnly, None, 0.0)?;
    let bdp = BoundedDualProblem::new(Objective::EndowmentLinear(obj), edges, problem.n_nodes())?;
    let (solution, certificate) =
        bdp.solve_and_certify_with_seed(opts.solver, derive_tols(&opts.solver), nu_seed)?;
    // Direct-only mode has no split/merge edge, so the "realized bound" is
    // meaningless — return 0.0 as a sentinel that callers can ignore.
    Ok(SolveOutcome {
        solution,
        certificate,
        split_bound: 0.0,
    })
}

fn solve_mixed_with_doubling(
    problem: &PredictionMarketProblem,
    opts: SolveOptions,
    nu_seed: Option<&[f64]>,
) -> Result<SolveOutcome, SolveError> {
    let obj = build_objective(problem)?;
    let n_nodes = problem.n_nodes();
    let b_max = analytical_split_bound(problem);
    let mut split_bound = problem
        .split_bound()
        .unwrap_or_else(|| default_split_bound(problem))
        .min(b_max);

    // Julia v2.0.0 parity: the primary split/merge oracle is bang-bang
    // (`is_nonsmooth(::SplitMergeEdge) = true`,
    // `prediction_markets.jl:196`). L-BFGS-B converges to gap ≈ 0 at
    // complementary slackness — the bang-bang oracle returns zero flow on
    // the `|gap| ≤ √eps` plateau, so the gradient from the split/merge edge
    // is zero there and L-BFGS-B is free to push gap to numerical zero.
    // `recover_primal` then computes the correct interior mint via the
    // target-residual averaging path. The previous Moreau-Yosida smoothing
    // (`μ = gap_tol/(5·B²)`) left L-BFGS-B terminating with
    // `|gap| ≈ 5e-5 > 10·pgtol`, which `recover_splitmerge_flow` saturated
    // to `w = ±B` instead of computing an interior mint — the warmup
    // fixture hit exactly this failure mode.
    let primary_mu = 0.0;

    let mut best: Option<SolveOutcome> = None;
    // Fallback: the most-recent uncertified outcome. Julia parity:
    // `_solve_prediction_market_mixed` with `throw_on_fail=false` always
    // returns a `PredictionMarketSolveResult` — when no B certifies within
    // `max_doublings+1` attempts it returns the last uncertified attempt
    // (`passed=false`) instead of erroring. The Rust port used to emit
    // `SolveError::NeverCertified` here, which the worker then turned into a
    // request-level `ok: false`. This field is the fix: we carry the last
    // uncertified outcome forward and return it downgraded if the loop
    // exhausts, so callers still receive numeric certificate diagnostics.
    let mut last_uncertified: Option<SolveOutcome> = None;
    // Warm-start seed tracker: starts from the caller-provided seed, gets
    // promoted to the converged ν after each certified inner solve. Julia
    // parity: `prediction_market_api.jl:1398`.
    let mut seed: Option<Vec<f64>> = nu_seed.map(<[f64]>::to_vec);

    for doubling in 0..=opts.max_doublings {
        let outcome = solve_at_bound(
            problem,
            &obj,
            n_nodes,
            split_bound,
            primary_mu,
            opts,
            seed.as_deref(),
        )?;

        if !outcome.certificate.passed {
            if let Some(prior) = best.take() {
                // The last certified solve was at split_bound/2. Bisect
                // between that and the current (failed) split_bound to find
                // the highest certifiable bound. Julia line 1360 enters
                // bisection with `ν_bisect = ν_seed` — i.e. the most
                // recently certified ν, which is exactly our `seed`.
                return bisect(
                    problem,
                    &obj,
                    n_nodes,
                    split_bound / 2.0,
                    split_bound,
                    prior,
                    opts,
                    primary_mu,
                    seed.as_deref(),
                );
            }
            // No prior certified result yet — remember this attempt as the
            // fallback, then keep doubling. Leave `seed` alone: the ν from
            // an uncertified solve is not promoted (Julia line 1398 only
            // runs on the certified branch).
            let last_bound = split_bound;
            last_uncertified = Some(outcome);
            let next = (split_bound * 2.0).min(b_max);
            if next <= split_bound {
                // Already at b_max; a further doubling won't change B.
                // Return the last uncertified attempt as an uncertified
                // result — parity with Julia `throw_on_fail=false`, which
                // never errors here and returns `passed=false` instead.
                return Ok(downgrade_never_certified(
                    last_uncertified.expect("just stored"),
                    opts.max_doublings + 1,
                    last_bound,
                ));
            }
            split_bound = next;
            continue;
        }

        // Certified. Check the 0.8·B near-active criterion.
        let realized = realized_split_flow(&outcome.solution);
        if realized < 0.8 * split_bound {
            return Ok(outcome);
        }
        // Promote the converged dual as the seed for the next doubling
        // iteration (or for bisection, if the next iteration fails).
        // Mirrors Julia `ν_seed = ν_candidate` at line 1398.
        seed = Some(outcome.solution.nu.clone());
        best = Some(outcome);

        if doubling == opts.max_doublings {
            // Mirror Julia: with `throw_on_fail=false` (the default v1
            // behavior) we don't error on a near-active certified result;
            // we downgrade the certificate and return the best result we
            // have. The orchestration layer turns `passed=false` into
            // `status: "uncertified"` in the wire DTO.
            return Ok(downgrade_near_active(
                best.expect("just stored"),
                opts.max_doublings,
                split_bound,
            ));
        }
        let next = (split_bound * 2.0).min(b_max);
        if next <= split_bound {
            // At the analytical ceiling — return the best certified result
            // even if it is still near-active. Further doubling can't help.
            return Ok(best.expect("just stored"));
        }
        split_bound = next;
    }

    // Loop exhausted without any certified result. Return the last
    // uncertified outcome downgraded to `passed=false` rather than erroring:
    // Julia parity with `throw_on_fail=false`.
    if let Some(outcome) = last_uncertified {
        return Ok(downgrade_never_certified(
            outcome,
            opts.max_doublings + 1,
            split_bound,
        ));
    }
    best.ok_or(SolveError::NeverCertified(opts.max_doublings + 1))
}

/// 4-step bisection between a known-certifiable `lo` bound and a known-failing
/// `hi` bound. Mirrors the inner `for _ in 1:4` loop at
/// `prediction_market_api.jl:1363`.
fn bisect(
    problem: &PredictionMarketProblem,
    obj: &forecast_flows_core::EndowmentLinear,
    n_nodes: usize,
    mut lo: f64,
    mut hi: f64,
    initial_best: SolveOutcome,
    opts: SolveOptions,
    primary_mu: f64,
    nu_seed: Option<&[f64]>,
) -> Result<SolveOutcome, SolveError> {
    let mut best = initial_best;
    // Julia `prediction_market_api.jl:1360-1361` initializes
    // `ν_bisect = ν_seed` (= the most recently certified dual before
    // entering bisection) and updates `ν_bisect = ν_mid` on each certified
    // midpoint (line 1383). Uncertified midpoints do not promote.
    let mut seed: Option<Vec<f64>> = nu_seed.map(<[f64]>::to_vec);
    for _ in 0..4 {
        let mid = 0.5 * (lo + hi);
        let outcome = solve_at_bound(
            problem,
            obj,
            n_nodes,
            mid,
            primary_mu,
            opts,
            seed.as_deref(),
        )?;
        if outcome.certificate.passed {
            lo = mid;
            seed = Some(outcome.solution.nu.clone());
            best = outcome;
        } else {
            hi = mid;
        }
    }
    Ok(best)
}

/// Julia parity: `_SPLITMERGE_RESCUE_BAND` at `src/solver.jl:76`. The
/// regularized rescue uses a `band`-wide smoothing window on the split/merge
/// oracle. Julia's `splitmerge_regularized_oracle(gap, B, band)` returns
/// `w = clamp(B·gap/band, -B, B)`, which is algebraically identical to
/// Moreau-Yosida smoothing with `μ = band/B`. The Rust port therefore
/// realizes the rescue by rebuilding the SplitMerge edge with `μ = band/B`
/// — no new oracle needed.
const SPLITMERGE_RESCUE_BAND: f64 = 1e-2;

/// Single mixed-mode solve at a fixed `(B, μ)` — rebuilds the
/// `BoundedDualProblem` so the `Edge::SplitMerge` carries the requested
/// bound and smoothing. Primary path; no rescue.
fn solve_at_bound_once(
    problem: &PredictionMarketProblem,
    obj: &forecast_flows_core::EndowmentLinear,
    n_nodes: usize,
    split_bound: f64,
    mu: f64,
    opts: SolveOptions,
    nu_seed: Option<&[f64]>,
) -> Result<SolveOutcome, SolveError> {
    let edges = build_edges(problem, Mode::MixedEnabled, Some(split_bound), mu)?;
    let bdp = BoundedDualProblem::new(Objective::EndowmentLinear(obj.clone()), edges, n_nodes)?;
    let (solution, certificate) =
        bdp.solve_and_certify_with_seed(opts.solver, derive_tols(&opts.solver), nu_seed)?;
    Ok(SolveOutcome {
        solution,
        certificate,
        split_bound,
    })
}

/// Mixed-mode solve with an optional regularized rescue. Runs the primary
/// smoothed solve at `(B, μ)`; if certification fails, retries with a
/// gentler `μ_rescue = band/B` smoothing from a uniform cold seed and warm
/// starts the primary `μ` from the rescue's ν. Accepts the warm-start only
/// if its certificate strictly improves over the baseline.
///
/// Julia parity: `solve!` at `src/solver.jl:914` — specifically the
/// `_try_splitmerge_regularized_rescue!` branch that fires when the exact
/// split-merge oracle fails to certify. Julia's rescue uses a custom BFGS
/// over free coordinates only; the Rust port uses its existing L-BFGS-B
/// pipeline with a rebuilt SplitMerge edge, which is the Moreau-Yosida
/// analog of Julia's `splitmerge_regularized_oracle`.
fn solve_at_bound(
    problem: &PredictionMarketProblem,
    obj: &forecast_flows_core::EndowmentLinear,
    n_nodes: usize,
    split_bound: f64,
    mu: f64,
    opts: SolveOptions,
    nu_seed: Option<&[f64]>,
) -> Result<SolveOutcome, SolveError> {
    let baseline = solve_at_bound_once(problem, obj, n_nodes, split_bound, mu, opts, nu_seed)?;
    if baseline.certificate.passed {
        return Ok(baseline);
    }

    // μ_rescue = band / max(B, 1): Moreau-Yosida equivalent of Julia's
    // `splitmerge_regularized_oracle(gap, B, band)`. `max(B, 1)` mirrors
    // Julia's `B_safe = max(B, one(T))` guard in the OLDER smoothing helper
    // — extends naturally here so a B<1 doesn't invert the rescue/primary μ
    // ordering (the rescue must be *gentler* than the primary).
    let rescue_mu = SPLITMERGE_RESCUE_BAND / split_bound.max(1.0);
    if !(rescue_mu > mu) {
        // Primary μ is already at least as gentle as the rescue — nothing
        // more to try. Propagate the baseline (uncertified) outcome.
        return Ok(baseline);
    }

    let rescue = solve_at_bound_once(problem, obj, n_nodes, split_bound, rescue_mu, opts, None)?;
    if !rescue.solution.dual_value.is_finite() {
        return Ok(baseline);
    }
    for v in &rescue.solution.nu {
        if !v.is_finite() {
            return Ok(baseline);
        }
    }

    let warm = solve_at_bound_once(
        problem,
        obj,
        n_nodes,
        split_bound,
        mu,
        opts,
        Some(&rescue.solution.nu),
    )?;

    if should_accept_rescue(&baseline.certificate, &warm.certificate) {
        Ok(warm)
    } else {
        Ok(baseline)
    }
}

/// Julia parity: `_should_accept_splitmerge_rescue` at `src/solver.jl:742`.
/// Accept the rescue's warm-start result iff it certifies where the baseline
/// did not, or both certify and the candidate's primal strictly improves by
/// more than the tolerance.
fn should_accept_rescue(baseline: &SolveCertificate, candidate: &SolveCertificate) -> bool {
    if !candidate.primal_value.is_finite() || !candidate.dual_value.is_finite() {
        return false;
    }
    if candidate.passed && !baseline.passed {
        return true;
    }
    if baseline.passed && candidate.passed {
        // Julia `_splitmerge_rescue_accept_tol(T, primal_baseline)`:
        // `max(100·eps, 1e-10 · max(1, |primal_baseline|))`.
        let tol = (100.0 * f64::EPSILON).max(1e-10 * 1.0_f64.max(baseline.primal_value.abs()));
        return candidate.primal_value > baseline.primal_value + tol;
    }
    false
}

/// Mirror Julia `_mark_prediction_market_uncertified` +
/// `_append_prediction_market_message`: append the near-active diagnostic to
/// the existing certificate message, flip `passed` to false, and leave the
/// numeric fields untouched. The orchestration layer translates
/// `passed=false` into `status: "uncertified"`.
fn downgrade_near_active(
    mut outcome: SolveOutcome,
    max_doublings: usize,
    split_bound: f64,
) -> SolveOutcome {
    // `{split_bound:?}` formats a Float64 as "5.0" (Debug) rather than "5"
    // (Display), so the message matches Julia's `$(split_bound)` interpolation
    // exactly when `split_bound` is whole-valued.
    let extra = format!(
        "split/merge bound remained near-active after {max_doublings} doublings (bound={split_bound:?})"
    );
    let cert = &mut outcome.certificate;
    cert.passed = false;
    if cert.reason.is_empty() {
        cert.reason = extra;
    } else if !cert.reason.contains(&extra) {
        cert.reason = format!("{}; {}", cert.reason, extra);
    }
    outcome
}

/// Mirror Julia `_mark_prediction_market_uncertified` for the "mixed solve
/// never certified" branch. Preserves the numeric residuals (`duality_gap`,
/// `target_residual`, `bound_residual`) from the last inner solve — those
/// are load-bearing diagnostics for downstream callers — and stamps a
/// "never certified after N split_bound levels" message onto the existing
/// reason. `passed` is left as-is (already false from the inner solve's
/// verdict) but coerced to false defensively.
fn downgrade_never_certified(
    mut outcome: SolveOutcome,
    levels: usize,
    split_bound: f64,
) -> SolveOutcome {
    let extra = format!(
        "mixed solve never certified across {levels} split_bound levels (last bound={split_bound:?})"
    );
    let cert = &mut outcome.certificate;
    cert.passed = false;
    if cert.reason.is_empty() {
        cert.reason = extra;
    } else if !cert.reason.contains(&extra) {
        cert.reason = format!("{}; {}", cert.reason, extra);
    }
    outcome
}

/// `max(mint, merge) = |w| = |x[0]|` for the trailing `Edge::SplitMerge` in
/// the edge list. Matches Julia's
/// `max(result.split_merge.mint, result.split_merge.merge)` near-active
/// criterion at `prediction_market_api.jl:1400`.
fn realized_split_flow(solution: &DualSolution) -> f64 {
    let sm_idx = solution.edge_flows.len() - 1;
    solution.edge_flows[sm_idx][0].abs()
}

/// Helper that asserts the trailing edge is the split/merge hyperedge.
/// Used by tests to guard the contract `realized_split_flow` relies on.
#[cfg(test)]
fn assert_trailing_split_merge(edges: &[forecast_flows_core::Edge]) {
    match edges.last() {
        Some(forecast_flows_core::Edge::SplitMerge(_)) => {}
        _ => panic!("expected trailing Edge::SplitMerge"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{OutcomeSpec, UniV3Band, UniV3MarketSpec};

    fn problem_no_trade() -> PredictionMarketProblem {
        // Two outcomes with identical fair_value = 0.5, one UniV3 market at
        // p = 0.5, and an initial holding on "b" so the analytical split
        // bound is positive (every outcome must be reachable). At ν = c =
        // [1, 0.5, 0.5] the UniV3 edge sits in its no-trade cone (γ = 1,
        // p = 2 = current_price internal) and the split/merge gap is
        // 0.5 + 0.5 − 1 = 0 — so this is the equilibrium for both modes.
        let outcomes = vec![
            OutcomeSpec::new("a", 0.5, 0.0).unwrap(),
            OutcomeSpec::new("b", 0.5, 1.0).unwrap(),
        ];
        let market = UniV3MarketSpec::new(
            "m_a",
            "a",
            0.5,
            vec![
                UniV3Band::new(1.0, 100.0).unwrap(),
                UniV3Band::new(0.4, 100.0).unwrap(),
            ],
            1.0,
        )
        .unwrap();
        PredictionMarketProblem::new(outcomes, 1.0, vec![market], None).unwrap()
    }

    #[test]
    fn direct_only_certifies_no_trade_problem() {
        let problem = problem_no_trade();
        let opts = SolveOptions {
            mode: Mode::DirectOnly,
            ..Default::default()
        };
        let outcome = solve(&problem, opts).unwrap();
        assert!(
            outcome.certificate.passed,
            "certificate failed: {}",
            outcome.certificate.reason
        );
    }

    #[test]
    fn mixed_certifies_no_trade_problem_at_initial_bound() {
        // At equilibrium ν = c, gap = Σ c[1..] − c[0] = 1 − 1 = 0, so the
        // split/merge edge is idle. The initial B = default_split_bound
        // should pass on the first try, with realized flow ≈ 0 ≪ 0.8·B.
        let problem = problem_no_trade();
        let opts = SolveOptions {
            mode: Mode::MixedEnabled,
            ..Default::default()
        };
        let outcome = solve(&problem, opts).unwrap();
        assert!(
            outcome.certificate.passed,
            "certificate failed: {}",
            outcome.certificate.reason
        );
        // Don't assert exact split/merge flow here — at the no-trade
        // equilibrium the smoothed oracle's `w = clamp(gap/μ, -B, B)` can
        // saturate against tiny residual gaps even after `recover_primal`
        // snapping. Certificate-pass is the load-bearing assertion.
    }

    #[test]
    fn analytical_split_bound_takes_min_over_outcomes() {
        // caps["a"] = 0 + univ3_max_outcome_reserve(0.4..1.0 at L=100)
        //           ≈ 100·(1/√0.4 − 1/√1.0) ≈ 58.1
        // caps["b"] = 1 + 0 = 1
        // → b_max = min(58.1, 1) = 1.
        let problem = problem_no_trade();
        let b_max = analytical_split_bound(&problem);
        assert!((b_max - 1.0).abs() < 1e-12, "b_max = {b_max}");
    }

    #[test]
    fn build_edges_mixed_keeps_split_merge_trailing() {
        // Sanity: the doubling loop relies on the SplitMerge edge being the
        // last entry in the edge list so `realized_split_flow` reads the
        // correct edge_flows slot.
        let problem = problem_no_trade();
        let edges = build_edges(&problem, Mode::MixedEnabled, Some(1.0), 1e-4).unwrap();
        assert_trailing_split_merge(&edges);
    }

    /// P1 regression: warm-starting a mixed solve from a prior-converged ν
    /// should not increase L-BFGS-B iterations vs. a cold start. This is the
    /// load-bearing test for the seed-promotion fix in
    /// `solve_mixed_with_doubling`: without promotion, every doubling
    /// iteration restarted from the *initial* seed even after a certified
    /// inner solve, silently paying extra iterations on the mixed hot path.
    #[test]
    fn warm_seed_does_not_increase_iterations_over_cold() {
        let problem = problem_no_trade();
        let opts = SolveOptions {
            mode: Mode::MixedEnabled,
            ..Default::default()
        };
        let cold = solve_with_seed(&problem, opts, None).unwrap();
        assert!(cold.certificate.passed, "cold solve must certify");
        let warm = solve_with_seed(&problem, opts, Some(&cold.solution.nu)).unwrap();
        assert!(warm.certificate.passed, "warm solve must certify");
        assert!(
            warm.solution.iterations <= cold.solution.iterations,
            "warm iterations={} must be ≤ cold iterations={}",
            warm.solution.iterations,
            cold.solution.iterations,
        );
    }

    /// Regression for the deep_trading warmup fixture: 2 outcomes YES/NO
    /// with fair_values 0.55/0.45, collateral=1, two UniV3 markets at p=0.5
    /// with bands `[(1.0, L), (0.25, 0.0)]` and fee=0.9999. At `max_doublings=0`
    /// (LowLatency profile) the initial B=1 (analytical cap) must certify
    /// on the first try with an interior mint.
    ///
    /// Before the bang-bang primary fix, the primary solve used Moreau-Yosida
    /// smoothing (`μ = gap_tol/(5·B²) ≈ 1e-4`). L-BFGS-B terminated with
    /// `|gap| ≈ 5e-5 > recover_primal`'s `10·pgtol = 1e-5` gap tolerance, so
    /// `recover_splitmerge_flow` saturated `w = ±B = 1` instead of computing
    /// an interior mint — the dual objective had a non-zero split/merge
    /// gradient component at convergence that smoothing couldn't push to
    /// exactly zero at LowLatency `pgtol`. Switching to `μ=0` (Julia v2.0.0
    /// `is_nonsmooth(::SplitMergeEdge) = true` parity) puts L-BFGS-B on a
    /// `|gap| ≤ √eps` plateau where the split/merge edge contributes zero
    /// flow and zero gradient — L-BFGS-B can then minimize the remaining
    /// UniV3/endowment terms unobstructed, and `recover_primal` picks up
    /// the correct interior mint via target-residual averaging.
    ///
    /// Julia parity: `~/.julia/packages/ForecastFlows/zmQOD/src/solver.jl`
    /// with the same fixture produces `mint ≈ 0.687860`.
    #[test]
    fn warmup_fixture_certifies_at_max_doublings_zero() {
        let outcomes = vec![
            OutcomeSpec::new("YES", 0.55, 0.0).unwrap(),
            OutcomeSpec::new("NO", 0.45, 0.0).unwrap(),
        ];
        let market_yes = UniV3MarketSpec::new(
            "warm-u1",
            "YES",
            0.5,
            vec![
                UniV3Band::new(1.0, 10.0).unwrap(),
                UniV3Band::new(0.25, 0.0).unwrap(),
            ],
            0.9999,
        )
        .unwrap();
        let market_no = UniV3MarketSpec::new(
            "warm-u2",
            "NO",
            0.5,
            vec![
                UniV3Band::new(1.0, 9.0).unwrap(),
                UniV3Band::new(0.25, 0.0).unwrap(),
            ],
            0.9999,
        )
        .unwrap();
        let problem =
            PredictionMarketProblem::new(outcomes, 1.0, vec![market_yes, market_no], None).unwrap();
        let opts = SolveOptions {
            mode: Mode::MixedEnabled,
            max_doublings: 0,
            solver: SolverOptions {
                pgtol: 1e-6,
                ..SolverOptions::default()
            },
        };
        let outcome = solve(&problem, opts).unwrap();
        assert!(
            outcome.certificate.passed,
            "warmup fixture must certify on the first try: {}",
            outcome.certificate.reason
        );
        // Interior mint parity with Julia v2.0.0: `mint ≈ 0.687860`. The
        // split-merge flow magnitude is stored on the trailing edge; we
        // assert it's interior (not saturated at B=1) and within 1e-3 of
        // the Julia value.
        let sm_flow = outcome
            .solution
            .edge_flows
            .last()
            .expect("split/merge flow present")[0]
            .abs();
        assert!(
            (sm_flow - 0.687860).abs() < 1e-3,
            "interior mint must match Julia v2.0.0: got {sm_flow}"
        );
    }

    /// P1 regression: running the mixed doubling loop with
    /// `max_doublings > 0` must still certify and produce the same
    /// near-equilibrium result. Guards against a seed-promotion refactor
    /// accidentally passing a stale or miss-dimensioned ν into the inner
    /// `solve_at_bound`.
    #[test]
    fn mixed_doubling_loop_still_certifies_with_max_doublings_three() {
        let problem = problem_no_trade();
        let opts = SolveOptions {
            mode: Mode::MixedEnabled,
            max_doublings: 3,
            ..Default::default()
        };
        let outcome = solve(&problem, opts).unwrap();
        assert!(
            outcome.certificate.passed,
            "certificate failed after 3 doublings: {}",
            outcome.certificate.reason
        );
    }
}
