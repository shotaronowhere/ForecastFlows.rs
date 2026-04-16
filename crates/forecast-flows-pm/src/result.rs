//! Convert a solver `SolveOutcome` into the protocol-shaped `SolveResultDto`,
//! and orchestrate the two-mode `compare_prediction_market_families` call.
//!
//! Julia parity target: `_prediction_market_result`,
//! `_extract_prediction_market_trades`, `_extract_split_merge_plan`,
//! `compare_prediction_market_families` (`src/prediction_market_api.jl`).
//!
//! Narrowed v1 scope: no gas model — `estimated_execution_cost` and `net_ev`
//! are always `None` until the gas-model port lands.

use std::time::Instant;

use forecast_flows_core::{DualSolution, Edge, SolveCertificate};

use crate::problem::{Mode, PredictionMarketProblem};
use crate::protocol::{
    CertificateDto, CompareResponse, CompareResult, PROTOCOL_VERSION, SolveResultDto,
    SplitMergePlanDto, TradeDto, finite_or_null, finite_or_null_vec,
};
use crate::solve::{SolveError, SolveOptions, SolveOutcome, solve_with_seed};
use crate::workspace::PredictionMarketWorkspace;

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

/// Julia parity: when a per-mode solve runs the doubling loop to exhaustion
/// without finding *any* certifiable `split_bound`, Julia's
/// `compare_prediction_market_families!` downgrades that mode to an
/// "uncertified" no-trade result rather than aborting the whole compare.
/// Mirrors `_mark_prediction_market_uncertified` applied to a null identity
/// outcome. Holdings round-trip (no trade) and the certificate carries the
/// "never certified" diagnostic so downstream telemetry can distinguish a
/// downgrade from a genuine certified-but-near-active result.
fn uncertified_no_trade_result(
    problem: &PredictionMarketProblem,
    mode: Mode,
    reason: String,
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
    // No-trade identity: final == initial for every balance.
    let final_collateral = initial_collateral;
    let final_holdings = initial_holdings.clone();
    let initial_ev: f64 = initial_collateral
        + outcome_values
            .iter()
            .zip(initial_holdings.iter())
            .map(|(v, h)| v * h)
            .sum::<f64>();
    let final_ev = initial_ev;
    SolveResultDto {
        status: "uncertified".to_string(),
        mode: mode_label(mode).to_string(),
        certificate: Some(CertificateDto {
            passed: false,
            message: reason,
            primal_value: None,
            dual_value: None,
            duality_gap: None,
            target_residual: None,
            bound_residual: None,
        }),
        solver_time_sec: solver_time_sec.and_then(finite_or_null),
        initial_ev: finite_or_null(initial_ev),
        final_ev: finite_or_null(final_ev),
        ev_gain: finite_or_null(0.0),
        estimated_execution_cost: None,
        net_ev: None,
        outcome_ids,
        initial_collateral: finite_or_null(initial_collateral),
        final_collateral: finite_or_null(final_collateral),
        initial_holdings: finite_or_null_vec(&initial_holdings),
        final_holdings: finite_or_null_vec(&final_holdings),
        trades: Vec::new(),
        split_merge: SplitMergePlanDto {
            mint: finite_or_null(0.0),
            merge: finite_or_null(0.0),
        },
    }
}

/// Julia parity: `SolveError::NeverCertified` at the per-mode call-site means
/// the doubling loop exhausted without any certificate, which Julia downgrades
/// to an uncertified no-trade `SolveResultDto` rather than aborting the
/// compare. Other `SolveError` variants (problem-validation, core L-BFGS-B
/// failure) still bubble as hard errors.
fn solve_mode_for_compare(
    problem: &PredictionMarketProblem,
    opts: SolveOptions,
    nu_seed: Option<&[f64]>,
) -> Result<Result<SolveOutcome, String>, SolveError> {
    match solve_with_seed(problem, opts, nu_seed) {
        Ok(outcome) => Ok(Ok(outcome)),
        Err(SolveError::NeverCertified(levels)) => Ok(Err(format!(
            "mixed solve never certified across {levels} split_bound levels"
        ))),
        Err(other) => Err(other),
    }
}

/// Run both `direct_only` and `mixed_enabled` solves on `problem` and bundle
/// the results into a `CompareResponse`. Mirrors Julia
/// `compare_prediction_market_families!` minus the workspace cache and gas
/// model. Per-mode `NeverCertified` failures are downgraded to an uncertified
/// no-trade result so the compare as a whole still succeeds.
pub fn compare_prediction_market_families(
    problem: &PredictionMarketProblem,
    request_id: Option<String>,
    direct_opts: SolveOptions,
    mixed_opts: SolveOptions,
) -> Result<CompareResponse, SolveError> {
    let direct_start = Instant::now();
    let direct_outcome = solve_mode_for_compare(problem, direct_opts, None)?;
    let direct_secs = direct_start.elapsed().as_secs_f64();
    let direct_result = match direct_outcome {
        Ok(outcome) => {
            let direct_edges = crate::problem::build_edges(problem, Mode::DirectOnly, None, 0.0)
                .map_err(SolveError::Problem)?;
            extract_solve_result(
                problem,
                &outcome,
                Mode::DirectOnly,
                &direct_edges,
                Some(direct_secs),
            )
        }
        Err(reason) => {
            uncertified_no_trade_result(problem, Mode::DirectOnly, reason, Some(direct_secs))
        }
    };

    let mixed_start = Instant::now();
    let mixed_outcome = solve_mode_for_compare(problem, mixed_opts, None)?;
    let mixed_secs = mixed_start.elapsed().as_secs_f64();
    let mixed_result = match mixed_outcome {
        Ok(outcome) => {
            let mixed_edges = crate::problem::build_edges(
                problem,
                Mode::MixedEnabled,
                Some(outcome.split_bound),
                0.0,
            )
            .map_err(SolveError::Problem)?;
            extract_solve_result(
                problem,
                &outcome,
                Mode::MixedEnabled,
                &mixed_edges,
                Some(mixed_secs),
            )
        }
        Err(reason) => {
            uncertified_no_trade_result(problem, Mode::MixedEnabled, reason, Some(mixed_secs))
        }
    };

    Ok(CompareResponse {
        protocol_version: PROTOCOL_VERSION,
        request_id,
        ok: true,
        command: "compare_prediction_market_families".to_string(),
        result: CompareResult {
            direct_only: direct_result,
            mixed_enabled: mixed_result,
            workspace_reused: false,
        },
    })
}

/// Workspace-aware compare. Mirrors Julia
/// `compare_prediction_market_families!(workspace, problem; …)`. The workspace
/// supplies the dual seed for the direct solve from `direct_seed` and the
/// mixed solve also reads `direct_seed` (Julia
/// `prediction_market_api.jl:1322`), then stores each solve's converged `ν`
/// back into the matching slot.
///
/// Caller is responsible for ensuring the workspace layout matches `problem` —
/// the worker uses `worker_compare_workspace` to enforce that. The caller also
/// supplies `workspace_reused`, which is forwarded verbatim onto the returned
/// `CompareResult` so downstream telemetry can distinguish cold vs. warm-start
/// solves.
pub fn compare_prediction_market_families_with_workspace(
    workspace: &mut PredictionMarketWorkspace,
    problem: &PredictionMarketProblem,
    request_id: Option<String>,
    direct_opts: SolveOptions,
    mixed_opts: SolveOptions,
    workspace_reused: bool,
) -> Result<CompareResponse, SolveError> {
    let direct_start = Instant::now();
    let direct_outcome = solve_mode_for_compare(problem, direct_opts, workspace.direct_seed())?;
    let direct_secs = direct_start.elapsed().as_secs_f64();
    let direct_result = match direct_outcome {
        Ok(outcome) => {
            workspace.store_direct_seed(&outcome.solution.nu);
            let direct_edges = crate::problem::build_edges(problem, Mode::DirectOnly, None, 0.0)
                .map_err(SolveError::Problem)?;
            extract_solve_result(
                problem,
                &outcome,
                Mode::DirectOnly,
                &direct_edges,
                Some(direct_secs),
            )
        }
        Err(reason) => {
            // No certified dual → do not promote a stale seed; Julia
            // `prediction_market_api.jl:1398` only promotes on certified
            // inner solves.
            uncertified_no_trade_result(problem, Mode::DirectOnly, reason, Some(direct_secs))
        }
    };

    // Julia line 1322: mixed reads `direct_seed`, not `mixed_seed`.
    let mixed_start = Instant::now();
    let mixed_outcome = solve_mode_for_compare(problem, mixed_opts, workspace.direct_seed())?;
    let mixed_secs = mixed_start.elapsed().as_secs_f64();
    let mixed_result = match mixed_outcome {
        Ok(outcome) => {
            workspace.store_mixed_seed(&outcome.solution.nu);
            let mixed_edges = crate::problem::build_edges(
                problem,
                Mode::MixedEnabled,
                Some(outcome.split_bound),
                0.0,
            )
            .map_err(SolveError::Problem)?;
            extract_solve_result(
                problem,
                &outcome,
                Mode::MixedEnabled,
                &mixed_edges,
                Some(mixed_secs),
            )
        }
        Err(reason) => {
            uncertified_no_trade_result(problem, Mode::MixedEnabled, reason, Some(mixed_secs))
        }
    };

    Ok(CompareResponse {
        protocol_version: PROTOCOL_VERSION,
        request_id,
        ok: true,
        command: "compare_prediction_market_families".to_string(),
        result: CompareResult {
            direct_only: direct_result,
            mixed_enabled: mixed_result,
            workspace_reused,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{OutcomeSpec, UniV3Band, UniV3MarketSpec};
    use crate::solve::solve;
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

    /// Downgrade-path regression: when the doubling loop exhausts without any
    /// certified result (`SolveError::NeverCertified`), Julia parity requires
    /// returning an uncertified no-trade result for that mode rather than
    /// failing the whole compare. Prove the helper's shape: holdings round-
    /// trip, no trades, certificate has `passed=false` with the supplied
    /// reason, and EV numbers match the no-trade identity.
    #[test]
    fn uncertified_no_trade_result_round_trips_identity() {
        let problem = problem_no_trade();
        let dto = uncertified_no_trade_result(
            &problem,
            Mode::MixedEnabled,
            "mixed solve never certified across 7 split_bound levels".to_string(),
            Some(0.0125),
        );
        assert_eq!(dto.status, "uncertified");
        assert_eq!(dto.mode, "mixed_enabled");
        let cert = dto.certificate.as_ref().expect("certificate populated");
        assert!(!cert.passed);
        assert!(cert.message.contains("never certified"));
        // No trades; holdings round-trip from initial to final.
        assert!(dto.trades.is_empty(), "no-trade fallback emits no trades");
        assert_eq!(dto.initial_holdings, vec![Some(0.0), Some(1.0)]);
        assert_eq!(dto.final_holdings, vec![Some(0.0), Some(1.0)]);
        assert_eq!(dto.initial_collateral, Some(1.0));
        assert_eq!(dto.final_collateral, Some(1.0));
        // initial_ev = 1.0 + 0·0.5 + 1·0.5 = 1.5; ev_gain = 0 (no trade).
        assert!((dto.initial_ev.unwrap() - 1.5).abs() < 1e-12);
        assert!((dto.final_ev.unwrap() - 1.5).abs() < 1e-12);
        assert_eq!(dto.ev_gain, Some(0.0));
        // split/merge plan is identity-zero.
        assert_eq!(dto.split_merge.mint, Some(0.0));
        assert_eq!(dto.split_merge.merge, Some(0.0));
        // Solver time preserved.
        assert_eq!(dto.solver_time_sec, Some(0.0125));
    }

    /// Downgrade-path regression at the compare-orchestrator level: when
    /// `SolveError::NeverCertified` fires for a branch, we expect the other
    /// branch to still emit a certified result and the compare to succeed as
    /// a whole. Uses the Julia-parity helper directly since crafting a
    /// fixture that genuinely hits `NeverCertified` requires adversarial
    /// problem data; end-to-end coverage lives in the downstream
    /// `forecastflows_doctor` fixture run.
    #[test]
    fn downgrade_helper_covers_both_modes_symmetrically() {
        let problem = problem_no_trade();
        for mode in [Mode::DirectOnly, Mode::MixedEnabled] {
            let dto =
                uncertified_no_trade_result(&problem, mode, "forced downgrade".to_string(), None);
            assert_eq!(dto.status, "uncertified");
            assert_eq!(dto.mode, mode_label(mode));
            assert!(dto.trades.is_empty());
            assert_eq!(
                dto.initial_holdings, dto.final_holdings,
                "holdings must round-trip for {mode:?}"
            );
            assert_eq!(dto.solver_time_sec, None);
        }
    }

    /// P2a regression: deep_trading consumes `solver_time_sec` for per-mode
    /// latency telemetry (`forecastflows/mod.rs:238-239`,
    /// `client.rs:1233-1234`). The field was always `None` prior to the
    /// Instant::now() plumbing; the protocol-parity subset harness missed
    /// it because the committed fixture doesn't include the field, so
    /// `assert_json_subset` never asserted on it.
    #[test]
    fn compare_emits_finite_nonnegative_solver_time_sec_per_mode() {
        let problem = problem_no_trade();
        let direct_opts = SolveOptions {
            mode: Mode::DirectOnly,
            ..Default::default()
        };
        let mixed_opts = SolveOptions {
            mode: Mode::MixedEnabled,
            ..Default::default()
        };
        let resp =
            compare_prediction_market_families(&problem, None, direct_opts, mixed_opts).unwrap();
        for (label, dto) in [
            ("direct_only", &resp.result.direct_only),
            ("mixed_enabled", &resp.result.mixed_enabled),
        ] {
            let t = dto
                .solver_time_sec
                .unwrap_or_else(|| panic!("{label}.solver_time_sec missing"));
            assert!(t.is_finite(), "{label}.solver_time_sec={t} not finite");
            assert!(t >= 0.0, "{label}.solver_time_sec={t} negative");
        }
    }
}
