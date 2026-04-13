//! End-to-end NDJSON replay test for the worker binary.
//!
//! Spawns the compiled `forecast-flows-worker` binary, pipes a small NDJSON
//! request stream into stdin (health + compare + invalid), and asserts that
//! each response line is well-formed JSON, the compare response structurally
//! matches the committed protocol-v2 fixture (subset semantics matching
//! Julia's `assert_json_subset`), and a blank line in the middle of the
//! stream is silently skipped.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde_json::{Value, json};

fn fixture_dir() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.join("../../../ForecastFlows.jl/test/fixtures/protocol_v2")
}

fn load_fixture(name: &str) -> Value {
    let path = fixture_dir().join(format!("{name}_response.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

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
        (a, e) => assert_eq!(a, e, "leaf mismatch at {path}"),
    }
}

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
fn worker_binary_replays_health_compare_and_invalid_request() {
    let bin = env!("CARGO_BIN_EXE_forecast-flows-worker");

    let health_req = json!({
        "protocol_version": 2,
        "request_id": "h-1",
        "command": "health"
    });
    let compare_req = json!({
        "protocol_version": 2,
        "request_id": "compare-fixture",
        "command": "compare_prediction_market_families",
        "problem": fixture_problem_json(),
        "solve_options": {
            "pgtol": 1e-8,
            "max_iter": 5_000,
            "max_fun": 10_000,
            "throw_on_fail": false,
            "max_doublings": 0
        }
    });
    let invalid_req = json!({
        "protocol_version": 2,
        "request_id": "invalid-fixture",
        "command": "wat"
    });

    // Stream four lines: health, blank (must be skipped), compare, invalid.
    let mut input = String::new();
    input.push_str(&health_req.to_string());
    input.push('\n');
    input.push('\n');
    input.push_str(&compare_req.to_string());
    input.push('\n');
    input.push_str(&invalid_req.to_string());
    input.push('\n');

    let mut child = Command::new(bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn worker binary");
    child
        .stdin
        .as_mut()
        .expect("worker stdin")
        .write_all(input.as_bytes())
        .expect("write to worker stdin");
    let output = child.wait_with_output().expect("wait worker");
    assert!(
        output.status.success(),
        "worker exited non-zero: status={:?} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );

    let stdout = String::from_utf8(output.stdout).expect("worker stdout utf8");
    let lines: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(
        lines.len(),
        3,
        "expected 3 response lines (blank line skipped), got {}: {stdout}",
        lines.len()
    );

    let health_resp: Value = serde_json::from_str(lines[0]).expect("health line is JSON");
    assert_eq!(health_resp["ok"], true);
    assert_eq!(health_resp["command"], "health");
    assert_eq!(health_resp["request_id"], "h-1");

    let compare_resp: Value = serde_json::from_str(lines[1]).expect("compare line is JSON");
    let expected_compare = load_fixture("compare");
    assert_json_subset(&compare_resp, &expected_compare, "$");
    assert_eq!(compare_resp["request_id"], "compare-fixture");

    let invalid_resp: Value = serde_json::from_str(lines[2]).expect("invalid line is JSON");
    let expected_invalid = load_fixture("invalid_request");
    assert_json_subset(&invalid_resp, &expected_invalid, "$");
}
