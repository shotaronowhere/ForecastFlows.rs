//! NDJSON stdio worker for the `ForecastFlows` protocol v2.
//!
//! Julia parity target: `serve_protocol` in
//! `ForecastFlows.jl/src/prediction_market_api.jl:2010`.
//!
//! Line-buffered stdin → one JSON request per non-blank line → one JSON
//! response per line → flush after each response. A single
//! `PredictionMarketWorkspace` slot is held across lines so
//! `compare_prediction_market_families` requests with matching topology can
//! warm-start L-BFGS-B from the prior solve's converged dual — mirroring
//! Julia's `compare_workspace_ref` at `prediction_market_api.jl:2011`.

use std::io::{self, BufRead, BufWriter, Write};
use std::process::ExitCode;

use forecast_flows_pm::{PredictionMarketWorkspace, handle_protocol_json_with_workspace_cache};

fn main() -> ExitCode {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let stderr = io::stderr();
    let mut workspace: Option<PredictionMarketWorkspace> = None;

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
        let response = handle_protocol_json_with_workspace_cache(&line, &mut workspace);
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
