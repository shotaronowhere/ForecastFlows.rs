//! Convert a solver `SolveOutcome` into the protocol-shaped `SolveResultDto`,
//! and orchestrate the two-mode `compare_prediction_market_families` call.
//!
//! Julia parity target: `_prediction_market_result`,
//! `_extract_prediction_market_trades`, `_extract_split_merge_plan`,
//! `compare_prediction_market_families` (`src/prediction_market_api.jl`).
//!
//! Narrowed v1 scope: no gas model — `estimated_execution_cost` and `net_ev`
//! are always `None` until the gas-model port lands.

use forecast_flows_core::{DualSolution, Edge, SolveCertificate};

use crate::problem::{Mode, PredictionMarketProblem};
use crate::protocol::{
    CertificateDto, CompareResponse, CompareResult, PROTOCOL_VERSION, SolveResultDto,
    SplitMergePlanDto, TradeDto, finite_or_null, finite_or_null_vec,
};
use crate::solve::{SolveError, SolveOptions, SolveOutcome, solve};

/// Tolerance for "edge is active" / non-zero trade reporting. Mirrors Julia's
/// `max(1e-12, √eps)` default (`prediction_market_api.jl:1096`).
fn default_atol() -> f64 {
    f64::EPSILON.sqrt().max(1e-12)
}

fn mode_label(mode: Mode) -> &'static str {
    match mode {
        Mode::DirectOnly => "direct_only",
        Mode::MixedEnabled => "mixed_enabled",
    }
}

fn certificate_dto(cert: &SolveCertificate) -> CertificateDto {
    CertificateDto {
        passed: cert.passed,
        message: cert.reason.clone(),
        primal_value: finite_or_null(cert.primal_value),
        dual_value: finite_or_null(cert.dual_value),
        duality_gap: finite_or_null(cert.duality_gap),
        target_residual: finite_or_null(cert.target_residual),
        bound_residual: finite_or_null(cert.bound_residual),
    }
}

/// Extract per-market trades (one entry per active UniV3 edge). The pm
/// edge-builder lays out one `Edge::UniV3` per `problem.markets()` in order
/// followed by an optional trailing `Edge::SplitMerge`, so we zip
/// `problem.markets()` with the leading slice of `edge_flows`.
fn extract_trades(problem: &PredictionMarketProblem, solution: &DualSolution) -> Vec<TradeDto> {
    let atol = default_atol();
    let n_markets = problem.markets().len();
    let mut trades = Vec::with_capacity(n_markets);
    for (spec, flow) in problem
        .markets()
        .iter()
        .zip(solution.edge_flows.iter().take(n_markets))
    {
        let dc = flow[0];
        let dout = flow[1];
        if dc.abs() <= atol && dout.abs() <= atol {
            continue;
        }
        trades.push(TradeDto {
            market_id: spec.market_id().to_string(),
            outcome_id: spec.outcome_id().to_string(),
            collateral_delta: finite_or_null(dc),
            outcome_delta: finite_or_null(dout),
        });
    }
    trades
}

/// Mirror Julia `_extract_split_merge_plan`. For an `(n+1)`-vector
/// `x = [-w, w, …, w]`, mint = w (when w > atol), merge = -w (when w < -atol),
/// otherwise both zero. We average over the trailing block so residual
/// asymmetry from the smoothed oracle averages out.
fn extract_split_merge(edges: &[Edge], solution: &DualSolution) -> SplitMergePlanDto {
    let zero = SplitMergePlanDto {
        mint: finite_or_null(0.0),
        merge: finite_or_null(0.0),
    };
    let Some(last_idx) = edges.len().checked_sub(1) else {
        return zero;
    };
    if !matches!(edges[last_idx], Edge::SplitMerge(_)) {
        return zero;
    }
    let x = &solution.edge_flows[last_idx];
    if x.len() <= 1 {
        return zero;
    }
    let tail: f64 = x[1..].iter().sum();
    #[allow(clippy::cast_precision_loss)]
    let w = tail / ((x.len() - 1) as f64);
    let atol = default_atol();
    if w > atol {
        SplitMergePlanDto {
            mint: finite_or_null(w),
            merge: finite_or_null(0.0),
        }
    } else if w < -atol {
        SplitMergePlanDto {
            mint: finite_or_null(0.0),
            merge: finite_or_null(-w),
        }
    } else {
        zero
    }
}

/// Convert a `SolveOutcome` into the wire-shaped `SolveResultDto`. Mirrors
/// Julia `_prediction_market_result` — initial/final EV use the dot product of
/// `fair_value` against initial/final holdings, with collateral as the
/// node-`0` aggregate netflow shifted by the spec's `collateral_balance`.
#[must_use]
pub fn extract_solve_result(
    problem: &PredictionMarketProblem,
    outcome: &SolveOutcome,
    mode: Mode,
    edges: &[Edge],
    solver_time_sec: Option<f64>,
) -> SolveResultDto {
    let outcome_ids: Vec<String> = problem
        .outcomes()
        .iter()
        .map(|o| o.outcome_id().to_string())
        .collect();
    let initial_holdings: Vec<f64> = problem
        .outcomes()
        .iter()
        .map(crate::problem::OutcomeSpec::initial_holding)
        .collect();
    let outcome_values: Vec<f64> = problem
        .outcomes()
        .iter()
        .map(crate::problem::OutcomeSpec::fair_value)
        .collect();
    let initial_collateral = problem.collateral_balance();
    let netflow = &outcome.solution.netflow;
    let final_collateral = initial_collateral + netflow[0];
    let final_holdings: Vec<f64> = initial_holdings
        .iter()
        .zip(netflow.iter().skip(1))
        .map(|(h, &dh)| h + dh)
        .collect();
    let initial_ev: f64 = initial_collateral
        + outcome_values
            .iter()
            .zip(initial_holdings.iter())
            .map(|(v, h)| v * h)
            .sum::<f64>();
    let final_ev: f64 = final_collateral
        + outcome_values
            .iter()
            .zip(final_holdings.iter())
            .map(|(v, h)| v * h)
            .sum::<f64>();
    let trades = extract_trades(problem, &outcome.solution);
    let split_merge = extract_split_merge(edges, &outcome.solution);
    let cert_dto = certificate_dto(&outcome.certificate);
    let status = if cert_dto.passed {
        "certified"
    } else {
        "uncertified"
    };

    SolveResultDto {
        status: status.to_string(),
        mode: mode_label(mode).to_string(),
        certificate: Some(cert_dto),
        solver_time_sec: solver_time_sec.and_then(finite_or_null),
        initial_ev: finite_or_null(initial_ev),
        final_ev: finite_or_null(final_ev),
        ev_gain: finite_or_null(final_ev - initial_ev),
        estimated_execution_cost: None,
        net_ev: None,
        outcome_ids,
        initial_collateral: finite_or_null(initial_collateral),
        final_collateral: finite_or_null(final_collateral),
        initial_holdings: finite_or_null_vec(&initial_holdings),
        final_holdings: finite_or_null_vec(&final_holdings),
        trades,
        split_merge,
    }
}

/// Run both `direct_only` and `mixed_enabled` solves on `problem` and bundle
/// the results into a `CompareResponse`. Mirrors Julia
/// `compare_prediction_market_families!` minus the workspace cache and gas
/// model. Per-mode `solve` failures bubble up as `SolveError` instead of being
/// swallowed into an "uncertified" result — the workspace-aware variant in a
/// later phase will catch and downgrade those.
pub fn compare_prediction_market_families(
    problem: &PredictionMarketProblem,
    request_id: Option<String>,
    direct_opts: SolveOptions,
    mixed_opts: SolveOptions,
) -> Result<CompareResponse, SolveError> {
    let direct_outcome = solve(problem, direct_opts)?;
    let direct_edges = crate::problem::build_edges(problem, Mode::DirectOnly, None, 0.0)
        .map_err(SolveError::Problem)?;
    let direct_result = extract_solve_result(
        problem,
        &direct_outcome,
        Mode::DirectOnly,
        &direct_edges,
        None,
    );

    let mixed_outcome = solve(problem, mixed_opts)?;
    let mixed_edges = crate::problem::build_edges(
        problem,
        Mode::MixedEnabled,
        Some(mixed_outcome.split_bound),
        0.0,
    )
    .map_err(SolveError::Problem)?;
    let mixed_result = extract_solve_result(
        problem,
        &mixed_outcome,
        Mode::MixedEnabled,
        &mixed_edges,
        None,
    );

    Ok(CompareResponse {
        protocol_version: PROTOCOL_VERSION,
        request_id,
        ok: true,
        command: "compare_prediction_market_families".to_string(),
        result: CompareResult {
            direct_only: direct_result,
            mixed_enabled: mixed_result,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{OutcomeSpec, UniV3Band, UniV3MarketSpec};
    fn problem_no_trade() -> PredictionMarketProblem {
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
    fn extract_solve_result_no_trade_certified_no_trades() {
        let problem = problem_no_trade();
        let direct_opts = SolveOptions {
            mode: Mode::DirectOnly,
            ..Default::default()
        };
        let outcome = solve(&problem, direct_opts).unwrap();
        let edges = crate::problem::build_edges(&problem, Mode::DirectOnly, None, 0.0).unwrap();
        let dto = extract_solve_result(&problem, &outcome, Mode::DirectOnly, &edges, None);
        assert_eq!(dto.status, "certified");
        assert_eq!(dto.mode, "direct_only");
        // initial_ev = 1.0 (collateral) + 0.5·0 + 0.5·1 = 1.5; same for final
        // (no trade), so ev_gain ~ 0.
        assert!((dto.initial_ev.unwrap() - 1.5).abs() < 1e-6);
        assert!(dto.ev_gain.unwrap().abs() < 1e-3);
        // No-trade equilibrium → any reported trades must be tiny solver
        // residuals (well below the slack pgtol convergence floor). We don't
        // assert empty because LBFGS-B's termination noise can leave residuals
        // above the default `√eps` atol on this seed.
        for t in &dto.trades {
            assert!(
                t.collateral_delta.unwrap().abs() < 1e-3,
                "unexpected nontrivial trade: {t:?}"
            );
        }
        // split_merge plan is None/zero in direct-only mode.
        assert_eq!(dto.split_merge.mint, Some(0.0));
        assert_eq!(dto.split_merge.merge, Some(0.0));
        // Holdings round-trip.
        assert_eq!(dto.initial_holdings, vec![Some(0.0), Some(1.0)]);
        assert_eq!(dto.outcome_ids, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn compare_runs_both_modes_no_trade() {
        let problem = problem_no_trade();
        let direct_opts = SolveOptions {
            mode: Mode::DirectOnly,
            ..Default::default()
        };
        let mixed_opts = SolveOptions {
            mode: Mode::MixedEnabled,
            ..Default::default()
        };
        let resp = compare_prediction_market_families(
            &problem,
            Some("req-1".to_string()),
            direct_opts,
            mixed_opts,
        )
        .unwrap();
        assert_eq!(resp.protocol_version, PROTOCOL_VERSION);
        assert_eq!(resp.command, "compare_prediction_market_families");
        assert_eq!(resp.request_id.as_deref(), Some("req-1"));
        assert_eq!(resp.result.direct_only.mode, "direct_only");
        assert_eq!(resp.result.mixed_enabled.mode, "mixed_enabled");
        assert_eq!(resp.result.direct_only.status, "certified");
        assert_eq!(resp.result.mixed_enabled.status, "certified");
    }
}
