# ForecastFlows.rs

Rust port of [`ForecastFlows.jl`](https://github.com/shotaronowhere/ForecastFlows.jl) — an optimal-order router for a prediction market, built on a convex-flow (dual-decomposition) formulation with a custom mint/merge hyperedge.

The Rust worker speaks the same protocol-v2 NDJSON stdio interface as the Julia oracle and is a drop-in replacement behind the [`deep_trading`](https://github.com/shotaronowhere/deep_trading) consumer's `PredictionMarketFacade`.

## Status

Phase 8 complete: workspace-cached NDJSON worker with warm-start dual seeds across compatible compare requests. See the canonical migration plan in the Julia repo:

- [`ForecastFlows.jl/docs/2026-04-12-rust-migration-plan.md`](https://github.com/shotaronowhere/ForecastFlows.jl/blob/main/docs/2026-04-12-rust-migration-plan.md) — live status, phase-by-phase spec, and progress log.

Julia parity is enforced via a subset-JSON fixture harness against committed Julia oracle outputs. Non-parity scope (gas model, `constant_product` markets) is deferred under migration plan §0.2.

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

`--test-threads=1` is **mandatory**: the `lbfgsb` C-port is not reentrant. Tests take a global `LBFGSB_LOCK` mutex, but parallel runs still occasionally flake under shared runners. CI enforces this.

## CI

`.github/workflows/rust.yml` runs the three commands above on every push to `main` and every pull request. Jobs are split so `fmt` / `clippy` fast-fail before the slower release test build.

## License

`Cargo.toml` declares `MIT OR Apache-2.0`. License text files are not yet checked in.
