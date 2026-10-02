# ForecastFlows.rs

Rust port of [`ForecastFlows.jl`](https://github.com/shotaronowhere/ForecastFlows.jl) — an optimal-order router for a prediction market, built on a convex-flow (dual-decomposition) formulation with a custom mint/merge hyperedge.

The Rust worker speaks the same protocol-v2 NDJSON stdio interface as the Julia oracle and is a drop-in replacement for the Julia worker.

## Status

Implements the workspace-cached NDJSON worker, with warm-start dual seeds across compatible compare requests.

Julia parity is enforced via a subset-JSON fixture harness against committed Julia oracle outputs. The gas model and `constant_product` markets are not yet ported.

## Crates

- **`forecast-flows-core`** — L-BFGS-B `BoundedDualProblem` adapter over the `lbfgsb` C-port, certification + primal recovery, Moreau–Yosida μ smoothing.
- **`forecast-flows-pm`** — prediction-market problem/edges, doubling+bisection mixed solver, workspace cache, NDJSON protocol v2 DTOs and request handlers.
- **`forecast-flows-worker`** — stdio binary: reads NDJSON requests line-by-line, holds a single cached `PredictionMarketWorkspace` across the loop.

## Build & test

Toolchain is pinned to `1.93.0` via [`rust-toolchain.toml`](rust-toolchain.toml).

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test  --workspace --release -- --test-threads=1
```

`--test-threads=1` is required because the `lbfgsb` C port is not reentrant; CI runs the tests serially.

## CI

`.github/workflows/rust.yml` runs the three commands above on every push to `main` and every pull request. Jobs are split so `fmt` / `clippy` fast-fail before the slower release test build.

## License

MIT OR Apache-2.0, as declared in `Cargo.toml`.
