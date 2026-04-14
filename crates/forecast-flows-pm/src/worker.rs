//! Stateless and workspace-cached protocol-v2 request handlers.
//!
//! Julia parity target: `handle_protocol_json` /
//! `_handle_protocol_json_with_workspace_cache` /
//! `_prediction_market_protocol_error_code` in
//! `src/prediction_market_api.jl` (lines 1945–2018).
//!
//! `handle_protocol_json` is stateless across calls; it delegates to the
//! cached handler with a throwaway `None` workspace slot. The cached variant
//! `handle_protocol_json_with_workspace_cache` reuses a
//! `PredictionMarketWorkspace` across lines for `compare_prediction_market_families`
//! requests with matching topology — that's the request kind Julia caches at
//! `prediction_market_api.jl:1962`. `solve_prediction_market` and `health`
//! requests bypass the cache, matching Julia's behavior of only entering the
//! cache branch for `CompareRequest`.

use std::time::Instant;

use serde_json::Value;

use crate::problem::PredictionMarketProblem;
use crate::problem::build_edges;
use crate::protocol::{
    ErrorBody, ErrorResponse, HealthResponse, PROTOCOL_VERSION, SolveResponse, error_codes,
};
use crate::request::{ParsedRequest, RequestError, parse_protocol_request};
use crate::result::{compare_prediction_market_families_with_workspace, extract_solve_result};
use crate::solve::{SolveError, SolveOptions, solve};
use crate::workspace::{PredictionMarketWorkspace, worker_compare_workspace};

/// Stable package version reported by the worker. Bumped together with the
/// `forecast-flows-pm` crate version and the Julia `Project.toml` version.
pub const PACKAGE_VERSION: &str = "2.0.0";

/// Parse one request line and return the rendered JSON response. Stateless —
/// no workspace reuse across calls. Mirrors Julia `handle_protocol_json`
/// (`prediction_market_api.jl:1999`) which delegates to the cached handler
/// with a throwaway `Ref{Any}(nothing)`.
#[must_use]
pub fn handle_protocol_json(line: &str) -> String {
    let mut workspace: Option<PredictionMarketWorkspace> = None;
    handle_protocol_json_with_workspace_cache(line, &mut workspace)
}

/// Workspace-cached variant. Compare requests whose topology matches the
/// cached workspace reuse its dual seeds; other requests (health, solve) and
/// incompatible compare requests bypass or rebuild the cache. Mirrors Julia
/// `_handle_protocol_json_with_workspace_cache`
/// (`prediction_market_api.jl:1949`).
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn handle_protocol_json_with_workspace_cache(
    line: &str,
    workspace: &mut Option<PredictionMarketWorkspace>,
) -> String {
    let mut request_id: Option<String> = None;

    let response = (|| -> Result<Value, ErrorResponse> {
        let payload: Value = serde_json::from_str(line).map_err(|e| {
            ErrorResponse::invalid_request(None, format!("ArgumentError: invalid JSON: {e}"))
        })?;

        // Pre-extract request_id so a later error preserves it on the wire.
        if let Some(v) = payload.get("request_id") {
            if let Some(s) = v.as_str() {
                request_id = Some(s.to_string());
            }
        }

        let parsed =
            parse_protocol_request(&payload).map_err(|RequestError { message }| ErrorResponse {
                protocol_version: PROTOCOL_VERSION,
                request_id: request_id.clone(),
                ok: false,
                error: ErrorBody {
                    code: error_codes::INVALID_REQUEST.to_string(),
                    message,
                },
            })?;

        // Override request_id with the typed parsed value (covers the case
        // where pre-extraction missed it because `request_id` was `null`).
        request_id = parsed.request_id().map(ToOwned::to_owned);

        match parsed {
            ParsedRequest::Health { request_id: rid } => {
                let resp = HealthResponse::stable(rid, PACKAGE_VERSION);
                Ok(serde_json::to_value(&resp).expect("HealthResponse always serializes"))
            }
            ParsedRequest::Solve {
                request_id: rid,
                problem,
                opts,
            } => handle_solve(rid, &problem, opts),
            ParsedRequest::Compare {
                request_id: rid,
                problem,
                direct_opts,
                mixed_opts,
            } => handle_compare(workspace, rid.as_ref(), &problem, direct_opts, mixed_opts),
        }
    })();

    let value = match response {
        Ok(v) => v,
        Err(err) => serde_json::to_value(&err).expect("ErrorResponse always serializes"),
    };
    serde_json::to_string(&value).expect("Value always serializes")
}

fn handle_solve(
    rid: Option<String>,
    problem: &PredictionMarketProblem,
    opts: SolveOptions,
) -> Result<Value, ErrorResponse> {
    let mode = opts.mode;
    // Julia parity: `PredictionMarketSolveResult.solver_time_sec`
    // (`prediction_market_api.jl:372`) is the wall-clock of the solve call.
    // Wrap only the `solve` invocation so DTO/edge construction time doesn't
    // leak into the number — matches Julia's `total_time += solve!(...)`
    // accounting at `solver.jl:1029`.
    let solve_start = Instant::now();
    let outcome = solve(problem, opts).map_err(|e| solve_error(rid.as_ref(), &e))?;
    let solve_secs = solve_start.elapsed().as_secs_f64();
    let edges = build_edges(problem, mode, Some(outcome.split_bound), 0.0)
        .map_err(|e| solve_error(rid.as_ref(), &SolveError::Problem(e)))?;
    let result = extract_solve_result(problem, &outcome, mode, &edges, Some(solve_secs));
    let resp = SolveResponse {
        protocol_version: PROTOCOL_VERSION,
        request_id: rid,
        ok: true,
        command: "solve_prediction_market".to_string(),
        result,
    };
    Ok(serde_json::to_value(&resp).expect("SolveResponse always serializes"))
}

fn handle_compare(
    workspace: &mut Option<PredictionMarketWorkspace>,
    rid: Option<&String>,
    problem: &PredictionMarketProblem,
    direct_opts: SolveOptions,
    mixed_opts: SolveOptions,
) -> Result<Value, ErrorResponse> {
    let ws = worker_compare_workspace(workspace, problem);
    match compare_prediction_market_families_with_workspace(
        ws,
        problem,
        rid.cloned(),
        direct_opts,
        mixed_opts,
    ) {
        Ok(resp) => Ok(serde_json::to_value(&resp).expect("CompareResponse always serializes")),
        Err(e) => Err(solve_error(rid, &e)),
    }
}

/// Map a solver/problem error into an `ErrorResponse`. Mirrors Julia
/// `_prediction_market_protocol_error_code`: problem-validation errors map
/// to `invalid_request`, solver failures map to `solve_failed`.
fn solve_error(request_id: Option<&String>, err: &SolveError) -> ErrorResponse {
    let (code, message) = match err {
        SolveError::Problem(e) => (error_codes::INVALID_REQUEST, format!("ArgumentError: {e}")),
        SolveError::Core(e) => (
            error_codes::SOLVE_FAILED,
            format!("PredictionMarketSolveFailed: {e}"),
        ),
        SolveError::NeverCertified(_) => (
            error_codes::SOLVE_FAILED,
            format!("PredictionMarketSolveFailed: {err}"),
        ),
    };
    ErrorResponse {
        protocol_version: PROTOCOL_VERSION,
        request_id: request_id.cloned(),
        ok: false,
        error: ErrorBody {
            code: code.to_string(),
            message,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn handles_health_request() {
        let req = json!({
            "protocol_version": 2,
            "request_id": "h-1",
            "command": "health"
        });
        let line = serde_json::to_string(&req).unwrap();
        let resp: Value = serde_json::from_str(&handle_protocol_json(&line)).unwrap();
        assert_eq!(resp["ok"], true);
        assert_eq!(resp["command"], "health");
        assert_eq!(resp["request_id"], "h-1");
        assert_eq!(resp["result"]["status"], "ok");
    }

    #[test]
    fn invalid_json_returns_invalid_request_error() {
        let resp: Value = serde_json::from_str(&handle_protocol_json("{not json")).unwrap();
        assert_eq!(resp["ok"], false);
        assert_eq!(resp["error"]["code"], "invalid_request");
        assert!(
            resp["error"]["message"]
                .as_str()
                .unwrap()
                .starts_with("ArgumentError: invalid JSON")
        );
    }

    fn compare_request_json(rid: &str) -> String {
        let req = json!({
            "protocol_version": 2,
            "request_id": rid,
            "command": "compare_prediction_market_families",
            "problem": {
                "outcomes": [
                    {"outcome_id": "a", "fair_value": 0.5, "initial_holding": 0.0},
                    {"outcome_id": "b", "fair_value": 0.5, "initial_holding": 1.0}
                ],
                "collateral_balance": 1.0,
                "markets": [
                    {
                        "type": "univ3",
                        "market_id": "m_a",
                        "outcome_id": "a",
                        "current_price": 0.5,
                        "fee_multiplier": 1.0,
                        "bands": [
                            {"lower_price": 1.0, "liquidity_L": 100.0},
                            {"lower_price": 0.4, "liquidity_L": 100.0}
                        ]
                    }
                ]
            }
        });
        req.to_string()
    }

    /// Second compare request over the same topology must reuse the cached
    /// workspace: the workspace's `direct_seed` and `mixed_seed` are populated
    /// after the first solve, so the second solve warm-starts from the prior
    /// converged dual. Parity contract: the wire response is structurally
    /// identical (modulo `request_id`) — the cache is a performance
    /// optimization, not a behavior change.
    #[test]
    fn workspace_cache_reuses_seeds_across_compatible_compares() {
        let mut ws: Option<PredictionMarketWorkspace> = None;
        let r1 = handle_protocol_json_with_workspace_cache(&compare_request_json("c-1"), &mut ws);
        assert!(ws.is_some(), "cache populated after first compare");
        let ws_after_first = ws.as_ref().unwrap();
        assert!(
            ws_after_first.direct_seed().is_some(),
            "direct_seed stored after first compare"
        );
        assert!(
            ws_after_first.mixed_seed().is_some(),
            "mixed_seed stored after first compare"
        );

        let r2 = handle_protocol_json_with_workspace_cache(&compare_request_json("c-2"), &mut ws);

        let v1: Value = serde_json::from_str(&r1).unwrap();
        let v2: Value = serde_json::from_str(&r2).unwrap();
        assert_eq!(v1["ok"], true);
        assert_eq!(v2["ok"], true);
        assert_eq!(v1["request_id"], "c-1");
        assert_eq!(v2["request_id"], "c-2");
        // Strip request_id and per-call wall-clock `solver_time_sec`, then
        // compare the full result payload — the cached warm-start must not
        // change the trades, EV, or certificate.
        let mut v1_stripped = v1.clone();
        let mut v2_stripped = v2.clone();
        v1_stripped["request_id"] = Value::Null;
        v2_stripped["request_id"] = Value::Null;
        for v in [&mut v1_stripped, &mut v2_stripped] {
            for mode in ["direct_only", "mixed_enabled"] {
                v["result"][mode]["solver_time_sec"] = Value::Null;
            }
        }
        assert_eq!(
            v1_stripped, v2_stripped,
            "cached compare response must match fresh response bit-for-bit"
        );
    }

    #[test]
    fn workspace_cache_rebuilds_on_incompatible_topology() {
        let mut ws: Option<PredictionMarketWorkspace> = None;
        let _ = handle_protocol_json_with_workspace_cache(&compare_request_json("c-1"), &mut ws);
        // Second request renames the market — layout compare must reject and
        // rebuild, so the freshly allocated workspace has no cached seeds
        // until the second solve finishes.
        let second = compare_request_json("c-2").replace("\"m_a\"", "\"m_b\"");
        let _ = handle_protocol_json_with_workspace_cache(&second, &mut ws);
        let ws_after = ws.as_ref().expect("workspace present");
        // After the rebuild+solve, the *new* workspace has its own seeds.
        assert!(ws_after.direct_seed().is_some());
        assert!(ws_after.mixed_seed().is_some());
    }

    #[test]
    fn unsupported_command_preserves_request_id() {
        let req = json!({
            "protocol_version": 2,
            "request_id": "invalid-fixture",
            "command": "wat"
        });
        let resp: Value = serde_json::from_str(&handle_protocol_json(&req.to_string())).unwrap();
        assert_eq!(resp["ok"], false);
        assert_eq!(resp["request_id"], "invalid-fixture");
        assert_eq!(
            resp["error"]["message"],
            "ArgumentError: unsupported command: wat"
        );
    }
}
