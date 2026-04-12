//! NDJSON protocol v2 DTOs for the prediction-market worker.
//!
//! Julia parity target: `src/prediction_market_api.jl` protocol types and
//! `StructTypes.lower` shapes — `HealthRequest`/`HealthResponse`,
//! `SolveRequest`/`SolveResponse`, `CompareRequest`/`CompareResponse`,
//! `ErrorResponse`, `PredictionMarketSolveResult`, `SolveCertificateSummary`,
//! `PredictionMarketTrade`, `SplitMergePlan`.
//!
//! This module only defines pure-data shapes and their JSON serialization.
//! It has no solver or orchestration logic; compare/solve response
//! construction will plug into these DTOs in a later phase.

use serde::{Deserialize, Serialize};

/// Stable NDJSON protocol version exposed by v2 workers.
pub const PROTOCOL_VERSION: u32 = 2;

// -- Responses --------------------------------------------------------------

/// Health response envelope. `ok` is always `true`, `command` is always
/// `"health"`. Matches Julia `StructTypes.lower(::HealthResponse)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub protocol_version: u32,
    pub request_id: Option<String>,
    pub ok: bool,
    pub command: String,
    pub result: HealthResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResult {
    pub status: String,
    pub package: String,
    pub package_version: String,
    pub supported_commands: Vec<String>,
    pub supported_modes: Vec<String>,
    pub supported_market_types: Vec<String>,
    pub stable_interfaces: Vec<String>,
    pub public_interfaces: Vec<String>,
    pub numeric_units: String,
    pub execution_model: String,
}

impl HealthResponse {
    /// Build a v2 health response with the stable `ForecastFlows` capability
    /// list, matching the committed `health_response.json` fixture exactly.
    #[must_use]
    pub fn stable(request_id: Option<String>, package_version: impl Into<String>) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            request_id,
            ok: true,
            command: "health".to_string(),
            result: HealthResult {
                status: "ok".to_string(),
                package: "ForecastFlows".to_string(),
                package_version: package_version.into(),
                supported_commands: vec![
                    "health".to_string(),
                    "solve_prediction_market".to_string(),
                    "compare_prediction_market_families".to_string(),
                ],
                supported_modes: vec!["direct_only".to_string(), "mixed_enabled".to_string()],
                supported_market_types: vec!["constant_product".to_string(), "univ3".to_string()],
                stable_interfaces: vec![
                    "prediction_market_facade".to_string(),
                    "ndjson_protocol".to_string(),
                ],
                public_interfaces: vec![
                    "PredictionMarketWorkspace".to_string(),
                    "PredictionMarketFixedGasModel".to_string(),
                    "solve_prediction_market!".to_string(),
                    "compare_prediction_market_families!".to_string(),
                    "PREDICTION_MARKET_PROTOCOL_VERSION".to_string(),
                    "HealthRequest".to_string(),
                    "SolveRequest".to_string(),
                    "CompareRequest".to_string(),
                    "HealthResponse".to_string(),
                    "SolveResponse".to_string(),
                    "CompareResponse".to_string(),
                    "ErrorResponse".to_string(),
                    "parse_protocol_request".to_string(),
                    "handle_protocol_request".to_string(),
                    "render_protocol_response".to_string(),
                    "handle_protocol_json".to_string(),
                    "serve_protocol".to_string(),
                ],
                numeric_units: "decimal collateral and outcome token units".to_string(),
                execution_model:
                    "NDJSON; one request at a time per worker process; serve_protocol reuses \
                     compatible compare workspaces; handle_protocol_json is stateless"
                        .to_string(),
            },
        }
    }
}

/// Solve response envelope for `solve_prediction_market`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveResponse {
    pub protocol_version: u32,
    pub request_id: Option<String>,
    pub ok: bool,
    pub command: String,
    pub result: SolveResultDto,
}

/// Compare response envelope for `compare_prediction_market_families`. The
/// result carries both the `direct_only` and `mixed_enabled` solve results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareResponse {
    pub protocol_version: u32,
    pub request_id: Option<String>,
    pub ok: bool,
    pub command: String,
    pub result: CompareResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareResult {
    pub direct_only: SolveResultDto,
    pub mixed_enabled: SolveResultDto,
}

/// Pure-data solve result mirroring `PredictionMarketSolveResult`.
///
/// Non-finite float fields are serialized as JSON `null` by storing them as
/// `Option<f64>`, matching Julia's `_prediction_market_json_number` helper.
/// Production-side construction will replace non-finite values with `None`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveResultDto {
    pub status: String,
    pub mode: String,
    pub certificate: Option<CertificateDto>,
    pub solver_time_sec: Option<f64>,
    pub initial_ev: Option<f64>,
    pub final_ev: Option<f64>,
    pub ev_gain: Option<f64>,
    pub estimated_execution_cost: Option<f64>,
    pub net_ev: Option<f64>,
    pub outcome_ids: Vec<String>,
    pub initial_collateral: Option<f64>,
    pub final_collateral: Option<f64>,
    pub initial_holdings: Vec<Option<f64>>,
    pub final_holdings: Vec<Option<f64>>,
    pub trades: Vec<TradeDto>,
    pub split_merge: SplitMergePlanDto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateDto {
    pub passed: bool,
    pub message: String,
    pub primal_value: Option<f64>,
    pub dual_value: Option<f64>,
    pub duality_gap: Option<f64>,
    pub target_residual: Option<f64>,
    pub bound_residual: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeDto {
    pub market_id: String,
    pub outcome_id: String,
    pub collateral_delta: Option<f64>,
    pub outcome_delta: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitMergePlanDto {
    pub mint: Option<f64>,
    pub merge: Option<f64>,
}

/// Error envelope (worker validation / solve failures). Matches Julia
/// `StructTypes.lower(::ErrorResponse)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub protocol_version: u32,
    pub request_id: Option<String>,
    pub ok: bool,
    pub error: ErrorBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
}

/// Stable machine-readable error codes used by the v2 protocol.
pub mod error_codes {
    pub const INVALID_REQUEST: &str = "invalid_request";
    pub const SOLVE_FAILED: &str = "solve_failed";
    pub const INTERNAL_ERROR: &str = "internal_error";
}

impl ErrorResponse {
    #[must_use]
    pub fn invalid_request(request_id: Option<String>, message: impl Into<String>) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            request_id,
            ok: false,
            error: ErrorBody {
                code: error_codes::INVALID_REQUEST.to_string(),
                message: message.into(),
            },
        }
    }
}

// -- Convert non-finite f64 -> None -----------------------------------------

/// Convert a raw float into the `Option<f64>` form used on the wire: finite
/// values stay as `Some`, non-finite values (`NaN`, `±∞`) become `None` and
/// serialize to JSON `null`. Matches Julia `_prediction_market_json_number`.
#[inline]
#[must_use]
pub fn finite_or_null(x: f64) -> Option<f64> {
    if x.is_finite() { Some(x) } else { None }
}

#[inline]
#[must_use]
pub fn finite_or_null_vec(xs: &[f64]) -> Vec<Option<f64>> {
    xs.iter().copied().map(finite_or_null).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_or_null_passes_finite() {
        assert_eq!(finite_or_null(1.5), Some(1.5));
        assert_eq!(finite_or_null(0.0), Some(0.0));
        assert_eq!(finite_or_null(-1e308), Some(-1e308));
    }

    #[test]
    fn finite_or_null_drops_nonfinite() {
        assert_eq!(finite_or_null(f64::NAN), None);
        assert_eq!(finite_or_null(f64::INFINITY), None);
        assert_eq!(finite_or_null(f64::NEG_INFINITY), None);
    }

    #[test]
    fn health_response_serializes_with_stable_top_level_fields() {
        let resp = HealthResponse::stable(Some("x".to_string()), "2.0.0");
        let value = serde_json::to_value(&resp).unwrap();
        assert_eq!(value["protocol_version"], 2);
        assert_eq!(value["request_id"], "x");
        assert_eq!(value["ok"], true);
        assert_eq!(value["command"], "health");
        assert_eq!(value["result"]["status"], "ok");
        assert_eq!(value["result"]["package"], "ForecastFlows");
    }

    #[test]
    fn error_response_shape_matches_invalid_request_fixture() {
        let e = ErrorResponse::invalid_request(
            Some("invalid-fixture".to_string()),
            "ArgumentError: unsupported command: wat",
        );
        let value = serde_json::to_value(&e).unwrap();
        assert_eq!(value["protocol_version"], 2);
        assert_eq!(value["request_id"], "invalid-fixture");
        assert_eq!(value["ok"], false);
        assert_eq!(value["error"]["code"], "invalid_request");
        assert_eq!(
            value["error"]["message"],
            "ArgumentError: unsupported command: wat"
        );
    }
}
