//! NDJSON stdio worker for the `ForecastFlows` protocol v2.
//!
//! Julia parity target: `serve_protocol` in
//! `ForecastFlows.jl/src/prediction_market_api.jl:2010`.
//!
//! Line-buffered stdin → one JSON request per non-blank line → one JSON
//! response per line → flush after each response. The handler itself lives
//! in `forecast_flows_pm::handle_protocol_json`; this binary is just the
//! stdio loop. Workspace-cache reuse from the Julia side will land together
//! with the `PredictionMarketWorkspace` port — see plan §0 for the
//! follow-on.

use std::io::{self, BufRead, BufWriter, Write};
use std::process::ExitCode;

use forecast_flows_pm::handle_protocol_json;

fn main() -> ExitCode {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let stderr = io::stderr();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                let _ = writeln!(
                    stderr.lock(),
                    "forecast-flows-worker: stdin read error: {e}"
                );
                return ExitCode::FAILURE;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let response = handle_protocol_json(&line);
        if let Err(e) = writeln!(out, "{response}").and_then(|()| out.flush()) {
            let _ = writeln!(
                stderr.lock(),
                "forecast-flows-worker: stdout write error: {e}"
            );
            return ExitCode::FAILURE;
        }
    }

    ExitCode::SUCCESS
}
