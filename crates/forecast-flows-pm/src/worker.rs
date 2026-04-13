//! Stateless protocol-v2 request handler.
//!
//! Julia parity target: `handle_protocol_json` /
//! `_handle_protocol_json_with_workspace_cache` /
//! `_prediction_market_protocol_error_code` in
//! `src/prediction_market_api.jl` (lines 1945–2018).
//!
//! `handle_protocol_json` is stateless across calls. The workspace-cached
//! variant from Julia (`_handle_protocol_json_with_workspace_cache`) is not
//! ported yet — the Rust solver has no `PredictionMarketWorkspace` to reuse,
//! so the stateful loop in `serve_protocol` would short-circuit to the same
//! "rebuild from scratch" path on every line. The cache lands together with
//! the workspace port in a follow-on phase.

use serde_json::Value;

use crate::protocol::{
    CompareResponse, CompareResult, ErrorBody, ErrorResponse, HealthResponse, PROTOCOL_VERSION,
    SolveResponse, error_codes,
};
use crate::request::{ParsedRequest, RequestError, parse_protocol_request};
use crate::result::extract_solve_result;
use crate::solve::{SolveError, solve};
use crate::{Mode, problem::build_edges};

/// Stable package version reported by the worker. Bumped together with the
/// `forecast-flows-pm` crate version and the Julia `Project.toml` version.
pub const PACKAGE_VERSION: &str = "2.0.0";

/// Parse one request line and return the rendered JSON response. Errors are
/// wrapped into `ErrorResponse` rather than returned as `Err` — the worker
/// loop keeps running across malformed lines.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn handle_protocol_json(line: &str) -> String {
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
            } => {
                let mode = opts.mode;
                let outcome = solve(&problem, opts).map_err(|e| solve_error(rid.as_ref(), &e))?;
                let edges = build_edges(&problem, mode, Some(outcome.split_bound), 0.0)
                    .map_err(|e| solve_error(rid.as_ref(), &SolveError::Problem(e)))?;
                let result = extract_solve_result(&problem, &outcome, mode, &edges, None);
                let resp = SolveResponse {
                    protocol_version: PROTOCOL_VERSION,
                    request_id: rid,
                    ok: true,
                    command: "solve_prediction_market".to_string(),
                    result,
                };
                Ok(serde_json::to_value(&resp).expect("SolveResponse always serializes"))
            }
            ParsedRequest::Compare {
                request_id: rid,
                problem,
                direct_opts,
                mixed_opts,
            } => {
                let direct_outcome =
                    solve(&problem, direct_opts).map_err(|e| solve_error(rid.as_ref(), &e))?;
                let direct_edges = build_edges(&problem, Mode::DirectOnly, None, 0.0)
                    .map_err(|e| solve_error(rid.as_ref(), &SolveError::Problem(e)))?;
                let direct_result = extract_solve_result(
                    &problem,
                    &direct_outcome,
                    Mode::DirectOnly,
                    &direct_edges,
                    None,
                );

                let mixed_outcome =
                    solve(&problem, mixed_opts).map_err(|e| solve_error(rid.as_ref(), &e))?;
                let mixed_edges = build_edges(
                    &problem,
                    Mode::MixedEnabled,
                    Some(mixed_outcome.split_bound),
                    0.0,
                )
                .map_err(|e| solve_error(rid.as_ref(), &SolveError::Problem(e)))?;
                let mixed_result = extract_solve_result(
                    &problem,
                    &mixed_outcome,
                    Mode::MixedEnabled,
                    &mixed_edges,
                    None,
                );

                let resp = CompareResponse {
                    protocol_version: PROTOCOL_VERSION,
                    request_id: rid,
                    ok: true,
                    command: "compare_prediction_market_families".to_string(),
                    result: CompareResult {
                        direct_only: direct_result,
                        mixed_enabled: mixed_result,
                    },
                };
                Ok(serde_json::to_value(&resp).expect("CompareResponse always serializes"))
            }
        }
    })();

    let value = match response {
        Ok(v) => v,
        Err(err) => serde_json::to_value(&err).expect("ErrorResponse always serializes"),
    };
    serde_json::to_string(&value).expect("Value always serializes")
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
