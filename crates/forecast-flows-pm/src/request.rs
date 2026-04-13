//! NDJSON protocol v2 request parsing.
//!
//! Julia parity target: `_parse_protocol_request` and the
//! `_prediction_market_protocol_*_from_json` family in
//! `src/prediction_market_api.jl` (lines 1620–1845). Every error message
//! mirrors Julia's `ArgumentError(...)` text byte-for-byte so that
//! `ErrorResponse.error.message` matches the committed protocol-v2
//! fixtures.
//!
//! Narrowed v1 scope: only `health` and `compare_prediction_market_families`
//! are accepted. `solve_prediction_market` is parsed and dispatched via the
//! same code path the parity tests use, even though it is not part of the
//! deep_trading consumer surface — adding it here is two extra match arms
//! and keeps the worker byte-compatible with Julia for solve requests if a
//! second consumer ever needs them. `gas_model` is rejected with a clear
//! "not implemented" message so a caller sending one fails loudly instead
//! of silently dropping it.

use serde_json::Value;

use crate::Mode;
use crate::problem::{OutcomeSpec, PredictionMarketProblem, UniV3Band, UniV3MarketSpec};
use crate::protocol::PROTOCOL_VERSION;
use crate::solve::SolveOptions;
use forecast_flows_core::SolverOptions;

/// Decimal-scaled "quantity" fields (token amounts, collateral) are clamped
/// against this safe integer cap so callers cannot accidentally send raw
/// base-unit integers and lose precision in `f64`. Mirrors Julia's
/// `PREDICTION_MARKET_WORKER_SAFE_INTEGER_LIMIT` (2^53).
const SAFE_INTEGER_LIMIT: f64 = 9_007_199_254_740_992.0;

/// One parsed protocol request. Mirrors Julia's
/// `Union{HealthRequest,SolveRequest,CompareRequest}`.
#[derive(Debug, Clone)]
pub enum ParsedRequest {
    Health {
        request_id: Option<String>,
    },
    Solve {
        request_id: Option<String>,
        problem: PredictionMarketProblem,
        opts: SolveOptions,
    },
    Compare {
        request_id: Option<String>,
        problem: PredictionMarketProblem,
        direct_opts: SolveOptions,
        mixed_opts: SolveOptions,
    },
}

impl ParsedRequest {
    #[must_use]
    pub fn request_id(&self) -> Option<&str> {
        match self {
            Self::Health { request_id }
            | Self::Solve { request_id, .. }
            | Self::Compare { request_id, .. } => request_id.as_deref(),
        }
    }
}

/// Parser error. The `message` payload matches Julia's `ArgumentError` text
/// exactly so the worker's `ErrorResponse.error.message` is fixture-stable.
#[derive(Debug, Clone)]
pub struct RequestError {
    pub message: String,
}

impl RequestError {
    fn argument_error(msg: impl Into<String>) -> Self {
        Self {
            message: format!("ArgumentError: {}", msg.into()),
        }
    }
}

/// Parse a raw protocol payload (already-decoded `serde_json::Value`).
/// Mirrors Julia `_parse_protocol_request`.
///
/// # Errors
///
/// Returns a `RequestError` if any required field is missing, has the wrong
/// type, or carries a value the parser rejects (unsupported version, unknown
/// command, non-finite quantity, etc.).
pub fn parse_protocol_request(payload: &Value) -> Result<ParsedRequest, RequestError> {
    let request_id = optional_request_id(payload)?;
    let protocol_version = require_int(payload, "protocol_version")?;
    if protocol_version != i64::from(PROTOCOL_VERSION) {
        return Err(RequestError::argument_error(format!(
            "unsupported protocol_version {protocol_version}"
        )));
    }
    let command = require_string(payload, "command")?;

    match command.as_str() {
        "health" => Ok(ParsedRequest::Health { request_id }),
        "solve_prediction_market" => {
            reject_gas_model(payload)?;
            let problem = parse_problem(payload)?;
            let mode = parse_mode(payload)?;
            let opts = build_solve_options(payload, mode)?;
            Ok(ParsedRequest::Solve {
                request_id,
                problem,
                opts,
            })
        }
        "compare_prediction_market_families" => {
            reject_gas_model(payload)?;
            let problem = parse_problem(payload)?;
            let direct_opts = build_solve_options(payload, Mode::DirectOnly)?;
            let mixed_opts = build_solve_options(payload, Mode::MixedEnabled)?;
            Ok(ParsedRequest::Compare {
                request_id,
                problem,
                direct_opts,
                mixed_opts,
            })
        }
        other => Err(RequestError::argument_error(format!(
            "unsupported command: {other}"
        ))),
    }
}

// -- field accessors --------------------------------------------------------

fn require<'a>(obj: &'a Value, key: &str, path: &str) -> Result<&'a Value, RequestError> {
    obj.get(key)
        .ok_or_else(|| RequestError::argument_error(format!("missing required field {path}")))
}

fn require_object<'a>(value: &'a Value, path: &str) -> Result<&'a Value, RequestError> {
    if value.is_object() {
        Ok(value)
    } else {
        Err(RequestError::argument_error(format!(
            "{path} must be an object"
        )))
    }
}

fn require_array<'a>(value: &'a Value, path: &str) -> Result<&'a Vec<Value>, RequestError> {
    value
        .as_array()
        .ok_or_else(|| RequestError::argument_error(format!("{path} must be an array")))
}

fn require_string(obj: &Value, key: &str) -> Result<String, RequestError> {
    let value = require(obj, key, key)?;
    string_field(value, key)
}

fn string_field(value: &Value, path: &str) -> Result<String, RequestError> {
    value
        .as_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| RequestError::argument_error(format!("{path} must be a string")))
}

fn optional_request_id(obj: &Value) -> Result<Option<String>, RequestError> {
    match obj.get("request_id") {
        None | Some(Value::Null) => Ok(None),
        Some(v) => Ok(Some(string_field(v, "request_id")?)),
    }
}

fn parse_mode(payload: &Value) -> Result<Mode, RequestError> {
    let Some(v) = payload.get("mode") else {
        return Ok(Mode::DirectOnly);
    };
    if v.is_null() {
        return Ok(Mode::DirectOnly);
    }
    let s = string_field(v, "mode")?;
    match s.as_str() {
        "direct_only" => Ok(Mode::DirectOnly),
        "mixed_enabled" => Ok(Mode::MixedEnabled),
        other => Err(RequestError::argument_error(format!(
            "mode has unsupported value: {other}"
        ))),
    }
}

fn require_number(value: &Value, path: &str) -> Result<f64, RequestError> {
    let parsed = match value {
        Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| RequestError::argument_error(format!("{path} must be numeric")))?,
        Value::String(s) => s.parse::<f64>().map_err(|_| {
            RequestError::argument_error(format!("{path} must be parseable as Float64"))
        })?,
        Value::Bool(_) => {
            return Err(RequestError::argument_error(format!(
                "{path} must be numeric, not boolean"
            )));
        }
        _ => {
            return Err(RequestError::argument_error(format!(
                "{path} must be numeric"
            )));
        }
    };
    if !parsed.is_finite() {
        return Err(RequestError::argument_error(format!(
            "{path} must be finite"
        )));
    }
    Ok(parsed)
}

/// Same as `require_number` but additionally rejects integer-valued payloads
/// above the f64 safe-integer cap, matching Julia's `quantity=true` guard.
fn require_quantity(value: &Value, path: &str) -> Result<f64, RequestError> {
    let parsed = require_number(value, path)?;
    if parsed.abs() > SAFE_INTEGER_LIMIT && parsed.fract() == 0.0 {
        return Err(RequestError::argument_error(format!(
            "{path} exceeds Float64's exact integer range; send decimal-scaled \
             token units instead of raw base-unit integers"
        )));
    }
    Ok(parsed)
}

fn require_int(obj: &Value, key: &str) -> Result<i64, RequestError> {
    let value = require(obj, key, key)?;
    int_field(value, key)
}

fn int_field(value: &Value, path: &str) -> Result<i64, RequestError> {
    let parsed = require_number(value, path)?;
    if parsed.fract() != 0.0 {
        return Err(RequestError::argument_error(format!(
            "{path} must be an integer"
        )));
    }
    // i64::MAX is 2^63-1; representable exactly only as the next f64 value
    // (2^63), so the cap is `< 2^63` in f64 space.
    if !(-9_223_372_036_854_775_808.0_f64..9_223_372_036_854_775_808.0_f64).contains(&parsed) {
        return Err(RequestError::argument_error(format!(
            "{path} is outside Int range"
        )));
    }
    #[allow(clippy::cast_possible_truncation)]
    Ok(parsed as i64)
}

fn require_bool(value: &Value, path: &str) -> Result<bool, RequestError> {
    value
        .as_bool()
        .ok_or_else(|| RequestError::argument_error(format!("{path} must be boolean")))
}

// -- problem/options parsers ------------------------------------------------

fn reject_gas_model(payload: &Value) -> Result<(), RequestError> {
    match payload.get("gas_model") {
        None | Some(Value::Null) => Ok(()),
        Some(_) => Err(RequestError::argument_error(
            "gas_model is not supported by the v1 Rust worker; omit it or send null",
        )),
    }
}

fn parse_problem(payload: &Value) -> Result<PredictionMarketProblem, RequestError> {
    let problem_value = require(payload, "problem", "problem")?;
    let problem = require_object(problem_value, "problem")?;

    if problem.get("outcome_values").is_some()
        || problem.get("initial_cash").is_some()
        || problem.get("initial_holdings").is_some()
    {
        return Err(RequestError::argument_error(
            "protocol v2 requires problem.outcomes and problem.collateral_balance; \
             v1 outcome_values/initial_cash payloads are not supported",
        ));
    }

    let outcomes_value = require(problem, "outcomes", "problem.outcomes")?;
    let outcomes_array = require_array(outcomes_value, "problem.outcomes")?;
    let markets_value = require(problem, "markets", "problem.markets")?;
    let markets_array = require_array(markets_value, "problem.markets")?;

    let split_bound = match problem.get("split_bound") {
        None | Some(Value::Null) => None,
        Some(v) => Some(require_quantity(v, "problem.split_bound")?),
    };

    let mut outcomes = Vec::with_capacity(outcomes_array.len());
    for (i, entry) in outcomes_array.iter().enumerate() {
        let path = format!("problem.outcomes[{i}]");
        let obj = require_object(entry, &path)?;
        let outcome_id = string_field(
            require(obj, "outcome_id", &format!("{path}.outcome_id"))?,
            &format!("{path}.outcome_id"),
        )?;
        let fair_value = require_number(
            require(obj, "fair_value", &format!("{path}.fair_value"))?,
            &format!("{path}.fair_value"),
        )?;
        let initial_holding = require_quantity(
            require(obj, "initial_holding", &format!("{path}.initial_holding"))?,
            &format!("{path}.initial_holding"),
        )?;
        outcomes.push(
            OutcomeSpec::new(outcome_id, fair_value, initial_holding)
                .map_err(|e| RequestError::argument_error(e.to_string()))?,
        );
    }

    let mut markets = Vec::with_capacity(markets_array.len());
    for (i, entry) in markets_array.iter().enumerate() {
        let path = format!("problem.markets[{i}]");
        markets.push(parse_market(entry, &path)?);
    }

    let collateral_balance = require_quantity(
        require(problem, "collateral_balance", "problem.collateral_balance")?,
        "problem.collateral_balance",
    )?;

    PredictionMarketProblem::new(outcomes, collateral_balance, markets, split_bound)
        .map_err(|e| RequestError::argument_error(e.to_string()))
}

fn parse_market(value: &Value, path: &str) -> Result<UniV3MarketSpec, RequestError> {
    let spec = require_object(value, path)?;
    if spec.get("outcome_index").is_some() {
        return Err(RequestError::argument_error(format!(
            "{path}.outcome_index is not supported in protocol v2; send outcome_id \
             and problem.outcomes instead"
        )));
    }
    let kind = string_field(
        require(spec, "type", &format!("{path}.type"))?,
        &format!("{path}.type"),
    )?;
    let market_id = string_field(
        require(spec, "market_id", &format!("{path}.market_id"))?,
        &format!("{path}.market_id"),
    )?;
    let outcome_id = string_field(
        require(spec, "outcome_id", &format!("{path}.outcome_id"))?,
        &format!("{path}.outcome_id"),
    )?;

    if kind != "univ3" {
        return Err(RequestError::argument_error(format!(
            "{path}.type has unsupported value: {kind}"
        )));
    }

    let current_price = require_number(
        require(spec, "current_price", &format!("{path}.current_price"))?,
        &format!("{path}.current_price"),
    )?;
    let fee_multiplier = require_number(
        require(spec, "fee_multiplier", &format!("{path}.fee_multiplier"))?,
        &format!("{path}.fee_multiplier"),
    )?;
    let bands_value = require(spec, "bands", &format!("{path}.bands"))?;
    let bands_array = require_array(bands_value, &format!("{path}.bands"))?;
    let mut bands = Vec::with_capacity(bands_array.len());
    for (i, entry) in bands_array.iter().enumerate() {
        let band_path = format!("{path}.bands[{i}]");
        let band = require_object(entry, &band_path)?;
        let lower_price = require_number(
            require(band, "lower_price", &format!("{band_path}.lower_price"))?,
            &format!("{band_path}.lower_price"),
        )?;
        let liquidity_l = require_number(
            require(band, "liquidity_L", &format!("{band_path}.liquidity_L"))?,
            &format!("{band_path}.liquidity_L"),
        )?;
        bands.push(
            UniV3Band::new(lower_price, liquidity_l)
                .map_err(|e| RequestError::argument_error(e.to_string()))?,
        );
    }

    UniV3MarketSpec::new(market_id, outcome_id, current_price, bands, fee_multiplier)
        .map_err(|e| RequestError::argument_error(e.to_string()))
}

/// Solver options carried by `solve_options`. Mirrors Julia
/// `_prediction_market_protocol_options`, narrowed to the knobs that the Rust
/// solver actually exposes today (`memory`, `factr`, `pgtol`, `max_iter`,
/// `max_doublings`). Unknown fields are silently ignored to match the Julia
/// helper which only forwards the recognized keys to LBFGS-B. `max_fun`,
/// `max_restarts`, and `method` are accepted-but-ignored for parity.
fn build_solve_options(payload: &Value, mode: Mode) -> Result<SolveOptions, RequestError> {
    let mut solver = SolverOptions::default();
    let mut max_doublings: usize = SolveOptions::default().max_doublings;

    if let Some(opts_value) = payload.get("solve_options") {
        if !opts_value.is_null() {
            let opts = require_object(opts_value, "solve_options")?;
            if let Some(v) = opts.get("memory") {
                solver.memory = usize_field(v, "solve_options.memory")?;
            }
            if let Some(v) = opts.get("factr") {
                solver.factr = require_number(v, "solve_options.factr")?;
            }
            if let Some(v) = opts.get("pgtol") {
                solver.pgtol = require_number(v, "solve_options.pgtol")?;
            }
            if let Some(v) = opts.get("max_iter") {
                solver.max_iter = usize_field(v, "solve_options.max_iter")?;
            }
            // `max_fun`, `max_restarts`, `method`, `certify`, `throw_on_fail`
            // are accepted-but-ignored: the Rust solver derives function-eval
            // budget from `max_iter`, has no warm restarts, only ships LBFGS-B,
            // and always returns a `SolveOutcome` with the certificate flag —
            // the caller never throws on a failed certificate.
            if let Some(v) = opts.get("max_fun") {
                let _ = usize_field(v, "solve_options.max_fun")?;
            }
            if let Some(v) = opts.get("max_restarts") {
                let _ = usize_field(v, "solve_options.max_restarts")?;
            }
            if let Some(v) = opts.get("method") {
                let _ = string_field(v, "solve_options.method")?;
            }
            if let Some(v) = opts.get("certify") {
                let _ = require_bool(v, "solve_options.certify")?;
            }
            if let Some(v) = opts.get("throw_on_fail") {
                let _ = require_bool(v, "solve_options.throw_on_fail")?;
            }
            if let Some(v) = opts.get("max_doublings") {
                max_doublings = usize_field(v, "solve_options.max_doublings")?;
            }
        }
    }

    Ok(SolveOptions {
        mode,
        max_doublings,
        solver,
    })
}

fn usize_field(value: &Value, path: &str) -> Result<usize, RequestError> {
    let parsed = int_field(value, path)?;
    usize::try_from(parsed)
        .map_err(|_| RequestError::argument_error(format!("{path} must be nonnegative")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture_problem_json() -> Value {
        json!({
            "outcomes": [
                {"outcome_id": "1", "fair_value": 0.55, "initial_holding": 0.0},
                {"outcome_id": "2", "fair_value": 0.45, "initial_holding": 0.0}
            ],
            "collateral_balance": 1.0,
            "split_bound": 5.0,
            "markets": [
                {
                    "type": "univ3",
                    "market_id": "m1",
                    "outcome_id": "1",
                    "current_price": 0.4,
                    "fee_multiplier": 1.0,
                    "bands": [
                        {"lower_price": 0.99, "liquidity_L": 63.246},
                        {"lower_price": 0.20, "liquidity_L": 63.246}
                    ]
                },
                {
                    "type": "univ3",
                    "market_id": "m2",
                    "outcome_id": "2",
                    "current_price": 0.7,
                    "fee_multiplier": 1.0,
                    "bands": [
                        {"lower_price": 0.99, "liquidity_L": 63.246},
                        {"lower_price": 0.20, "liquidity_L": 63.246}
                    ]
                }
            ]
        })
    }

    #[test]
    fn parses_health_request() {
        let payload = json!({
            "protocol_version": 2,
            "request_id": "h-1",
            "command": "health"
        });
        let parsed = parse_protocol_request(&payload).unwrap();
        match parsed {
            ParsedRequest::Health { request_id } => assert_eq!(request_id.as_deref(), Some("h-1")),
            _ => panic!("expected Health"),
        }
    }

    #[test]
    fn parses_compare_request_with_solve_options() {
        let payload = json!({
            "protocol_version": 2,
            "request_id": "c-1",
            "command": "compare_prediction_market_families",
            "problem": fixture_problem_json(),
            "solve_options": {
                "pgtol": 1e-8,
                "max_iter": 5000,
                "max_fun": 10000,
                "throw_on_fail": false,
                "max_doublings": 0
            }
        });
        let parsed = parse_protocol_request(&payload).unwrap();
        match parsed {
            ParsedRequest::Compare {
                request_id,
                problem,
                direct_opts,
                mixed_opts,
            } => {
                assert_eq!(request_id.as_deref(), Some("c-1"));
                assert_eq!(problem.markets().len(), 2);
                assert_eq!(direct_opts.solver.pgtol, 1e-8);
                assert_eq!(direct_opts.solver.max_iter, 5000);
                assert_eq!(direct_opts.max_doublings, 0);
                assert_eq!(mixed_opts.solver.pgtol, 1e-8);
                assert_eq!(mixed_opts.max_doublings, 0);
            }
            _ => panic!("expected Compare"),
        }
    }

    #[test]
    fn rejects_unsupported_protocol_version() {
        let payload = json!({
            "protocol_version": 3,
            "request_id": "bad-version-fixture",
            "command": "health"
        });
        let err = parse_protocol_request(&payload).unwrap_err();
        assert_eq!(err.message, "ArgumentError: unsupported protocol_version 3");
    }

    #[test]
    fn rejects_unknown_command() {
        let payload = json!({
            "protocol_version": 2,
            "request_id": "invalid-fixture",
            "command": "wat"
        });
        let err = parse_protocol_request(&payload).unwrap_err();
        assert_eq!(err.message, "ArgumentError: unsupported command: wat");
    }

    #[test]
    fn rejects_constant_product_market() {
        let mut problem = fixture_problem_json();
        problem["markets"][0]["type"] = json!("constant_product");
        let payload = json!({
            "protocol_version": 2,
            "command": "compare_prediction_market_families",
            "problem": problem
        });
        let err = parse_protocol_request(&payload).unwrap_err();
        assert!(
            err.message
                .contains("problem.markets[0].type has unsupported value: constant_product"),
            "{}",
            err.message
        );
    }

    #[test]
    fn rejects_gas_model_payload() {
        let payload = json!({
            "protocol_version": 2,
            "command": "compare_prediction_market_families",
            "problem": fixture_problem_json(),
            "gas_model": {"market_action_costs": [0.0, 0.0], "split_merge_action_cost": 0.0}
        });
        let err = parse_protocol_request(&payload).unwrap_err();
        assert!(
            err.message.contains("gas_model is not supported"),
            "{}",
            err.message
        );
    }

    #[test]
    fn rejects_quantity_above_safe_integer_limit() {
        // 2^54 = 18_014_398_509_481_984 — comfortably above the 2^53 cap and
        // representable exactly as f64, so the rejection fires before the
        // problem builder gets a chance to silently truncate.
        let mut problem = fixture_problem_json();
        problem["collateral_balance"] = json!(18_014_398_509_481_984_i64);
        let payload = json!({
            "protocol_version": 2,
            "command": "compare_prediction_market_families",
            "problem": problem
        });
        let err = parse_protocol_request(&payload).unwrap_err();
        assert!(
            err.message
                .contains("exceeds Float64's exact integer range"),
            "{}",
            err.message
        );
    }
}
