//! Parity harness against the Julia-owned `ForecastFlows.jl` protocol v2
//! fixtures (`test/fixtures/protocol_v2/*_response.json`).
//!
//! These tests mirror Julia's `assert_json_subset` helper in
//! `ForecastFlows.jl/test/protocol_fixtures.jl`. The Julia test harness treats
//! the committed fixtures as a *structural subset* of the full runtime
//! response: every field present in the fixture must appear identically in
//! the Rust-produced response, but the Rust response is allowed to carry
//! additional fields.
//!
//! At this phase we only cover the DTOs that can be constructed without the
//! solver (`health` and `invalid_request`/`unsupported_version`). The solve
//! and compare fixtures will be exercised against real solver output once
//! the bounded dual solver lands in Phase 5 and the PM facade/orchestration
//! lands in Phase 6.

use std::path::PathBuf;

use forecast_flows_pm::{ErrorResponse, HealthResponse};
use serde_json::Value;

// -- Fixture loader ---------------------------------------------------------

/// Path to the Julia-owned protocol v2 fixture directory, resolved relative
/// to this crate's `CARGO_MANIFEST_DIR` and expecting the default sibling-
/// repo layout (`ForecastFlows.rs` next to `ForecastFlows.jl`).
fn fixture_dir() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // crates/forecast-flows-pm -> ../../../ForecastFlows.jl/test/fixtures/protocol_v2
    manifest.join("../../../ForecastFlows.jl/test/fixtures/protocol_v2")
}

fn load_fixture(name: &str) -> Value {
    let path = fixture_dir().join(format!("{name}_response.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("fixture {} is not valid JSON: {e}", path.display()))
}

// -- Subset assertion -------------------------------------------------------

/// Assert that every leaf in `expected` appears identically in `actual`.
/// Mirrors Julia's `assert_json_subset` in `test/protocol_fixtures.jl`.
///
/// - Objects: every key in `expected` must exist in `actual`, recurse.
/// - Arrays: same length, elementwise recurse.
/// - Numbers: compared with `atol=1e-9, rtol=1e-9`.
/// - Everything else: strict equality.
fn assert_json_subset(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Object(a), Value::Object(e)) => {
            for (k, ev) in e {
                let av = a
                    .get(k)
                    .unwrap_or_else(|| panic!("missing key {path}.{k} in actual"));
                assert_json_subset(av, ev, &format!("{path}.{k}"));
            }
        }
        (Value::Array(a), Value::Array(e)) => {
            assert_eq!(
                a.len(),
                e.len(),
                "array length mismatch at {path}: actual={} expected={}",
                a.len(),
                e.len(),
            );
            for (i, (av, ev)) in a.iter().zip(e).enumerate() {
                assert_json_subset(av, ev, &format!("{path}[{i}]"));
            }
        }
        (Value::Number(a), Value::Number(e)) => {
            let af = a.as_f64().unwrap_or(f64::NAN);
            let ef = e.as_f64().unwrap_or(f64::NAN);
            let tol = 1e-9_f64.max(1e-9_f64 * ef.abs());
            assert!(
                (af - ef).abs() <= tol,
                "number mismatch at {path}: actual={af} expected={ef}",
            );
        }
        (a, e) => {
            assert_eq!(a, e, "leaf mismatch at {path}");
        }
    }
}

// -- Tests ------------------------------------------------------------------

#[test]
fn health_response_is_structural_superset_of_fixture() {
    let expected = load_fixture("health");
    let rust = HealthResponse::stable(Some("health-fixture".to_string()), "2.0.0");
    let actual = serde_json::to_value(&rust).unwrap();
    assert_json_subset(&actual, &expected, "$");
}

#[test]
fn invalid_request_error_matches_fixture() {
    let expected = load_fixture("invalid_request");
    let rust = ErrorResponse::invalid_request(
        Some("invalid-fixture".to_string()),
        "ArgumentError: unsupported command: wat",
    );
    let actual = serde_json::to_value(&rust).unwrap();
    assert_json_subset(&actual, &expected, "$");
}

#[test]
fn unsupported_version_error_matches_fixture() {
    let expected = load_fixture("unsupported_version");
    let rust = ErrorResponse::invalid_request(
        Some("bad-version-fixture".to_string()),
        "ArgumentError: unsupported protocol_version 3",
    );
    let actual = serde_json::to_value(&rust).unwrap();
    assert_json_subset(&actual, &expected, "$");
}

#[test]
fn fixture_directory_resolves_to_julia_oracle() {
    // Guard against cross-repo layout drift: if someone moves either repo and
    // forgets to update the parity harness, this test fires before the real
    // assertions produce confusing "missing key $.foo" errors.
    let dir = fixture_dir();
    assert!(
        dir.exists(),
        "expected Julia protocol_v2 fixture dir at {}; is the ForecastFlows.jl repo at the \
         expected sibling path?",
        dir.display()
    );
    for name in [
        "health",
        "compare",
        "solve_direct",
        "solve_mixed",
        "invalid_request",
        "unsupported_version",
    ] {
        let path = dir.join(format!("{name}_response.json"));
        assert!(path.exists(), "missing fixture {}", path.display());
    }
}
