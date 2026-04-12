//! NDJSON stdio worker for `ForecastFlows` protocol v2.
//!
//! Narrow v1 scope: line-buffered stdin, one JSON request per line, one
//! JSON response per line, flush after each line. Commands: `health`,
//! `compare_prediction_market_families`. Workspace cache mirrors the
//! Julia worker's compatible-compare cache behavior.

fn main() {}
