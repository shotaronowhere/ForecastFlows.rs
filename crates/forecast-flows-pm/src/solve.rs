//! Prediction-market solve entry point with `B`-doubling + Moreau-Yosida
//! `μ` annealing + 4-step bisection for the mixed (split/merge enabled) mode.
//!
//! Julia parity target: `_solve_prediction_market_mixed`
//! (`src/prediction_market_api.jl:1302`). The direct-only path is a single
//! `BoundedDualProblem::solve_and_certify` call with no doubling.
//!
//! No warm-start across iterations — each `B` rebuilds the
//! `BoundedDualProblem` from scratch and the solver re-seeds at
//! `clamp(c + 100·√eps, …)`. Julia warm-starts from the prior dual iterate
//! when a workspace is available; that optimization lands with the workspace
//! port in a later phase.

use forecast_flows_core::{
    BoundedDualProblem, CertifyTolerances, DualSolution, Objective, SolveCertificate,
    SolverOptions, moreau_yosida_mu,
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
    match opts.mode {
        Mode::DirectOnly => solve_direct_only(problem, opts),
        Mode::MixedEnabled => solve_mixed_with_doubling(problem, opts),
    }
}

fn solve_direct_only(
    problem: &PredictionMarketProblem,
    opts: SolveOptions,
) -> Result<SolveOutcome, SolveError> {
    let obj = build_objective(problem)?;
    let edges = build_edges(problem, Mode::DirectOnly, None, 0.0)?;
    let bdp = BoundedDualProblem::new(Objective::EndowmentLinear(obj), edges, problem.n_nodes())?;
    let (solution, certificate) = bdp.solve_and_certify(opts.solver, derive_tols(&opts.solver))?;
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
) -> Result<SolveOutcome, SolveError> {
    let obj = build_objective(problem)?;
    let n_nodes = problem.n_nodes();
    let b_max = analytical_split_bound(problem);
    let mut split_bound = problem
        .split_bound()
        .unwrap_or_else(|| default_split_bound(problem))
        .min(b_max);

    // gap_tol used to pick μ. Julia computes
    // `max(500·√eps, 500·pgtol)` independently of the certify tolerance, to
    // stay aligned with `_solve_certificate_tolerances`. We do the same so
    // the smoothing bias upper bound matches.
    let gap_tol_for_mu = (500.0 * f64::EPSILON.sqrt()).max(500.0 * opts.solver.pgtol);

    let mut best: Option<SolveOutcome> = None;

    for doubling in 0..=opts.max_doublings {
        let mu = moreau_yosida_mu(split_bound, gap_tol_for_mu);
        let outcome = solve_at_bound(problem, &obj, n_nodes, split_bound, mu, opts)?;

        if !outcome.certificate.passed {
            if let Some(prior) = best.take() {
                // The last certified solve was at split_bound/2. Bisect
                // between that and the current (failed) split_bound to find
                // the highest certifiable bound.
                return bisect(
                    problem,
                    &obj,
                    n_nodes,
                    split_bound / 2.0,
                    split_bound,
                    prior,
                    opts,
                    gap_tol_for_mu,
                );
            }
            // No prior certified result yet — keep doubling.
            let next = (split_bound * 2.0).min(b_max);
            if next <= split_bound {
                // Already at b_max; a further doubling won't change B and
                // won't change the verdict, so bail.
                return Err(SolveError::NeverCertified(opts.max_doublings + 1));
            }
            split_bound = next;
            continue;
        }

        // Certified. Check the 0.8·B near-active criterion.
        let realized = realized_split_flow(&outcome.solution);
        if realized < 0.8 * split_bound {
            return Ok(outcome);
        }
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
    gap_tol_for_mu: f64,
) -> Result<SolveOutcome, SolveError> {
    let mut best = initial_best;
    for _ in 0..4 {
        let mid = 0.5 * (lo + hi);
        let mu = moreau_yosida_mu(mid, gap_tol_for_mu);
        let outcome = solve_at_bound(problem, obj, n_nodes, mid, mu, opts)?;
        if outcome.certificate.passed {
            lo = mid;
            best = outcome;
        } else {
            hi = mid;
        }
    }
    Ok(best)
}

/// Single mixed-mode solve at a fixed `(B, μ)` — rebuilds the
/// `BoundedDualProblem` so the `Edge::SplitMerge` carries the requested
/// bound and smoothing.
fn solve_at_bound(
    problem: &PredictionMarketProblem,
    obj: &forecast_flows_core::EndowmentLinear,
    n_nodes: usize,
    split_bound: f64,
    mu: f64,
    opts: SolveOptions,
) -> Result<SolveOutcome, SolveError> {
    let edges = build_edges(problem, Mode::MixedEnabled, Some(split_bound), mu)?;
    let bdp = BoundedDualProblem::new(Objective::EndowmentLinear(obj.clone()), edges, n_nodes)?;
    let (solution, certificate) = bdp.solve_and_certify(opts.solver, derive_tols(&opts.solver))?;
    Ok(SolveOutcome {
        solution,
        certificate,
        split_bound,
    })
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
}
