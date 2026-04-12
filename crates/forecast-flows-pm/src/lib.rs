//! Prediction-market facade and protocol v2 DTOs.
//!
//! Narrow v1 scope: `UniV3MarketSpec` only, `EndowmentLinear` objective,
//! `compare_prediction_market_families` path (`direct_only` +
//! `mixed_enabled`) with workspace caching. No `solve_prediction_market`
//! single-mode facade, no `ConstantProductMarketSpec`, no gas model.
//!
//! Current session (Phase 3 narrowed slice): only the protocol v2 DTOs and
//! the parity fixture harness. Solver glue, workspace caching, problem
//! validation, and orchestration land in Phase 6.

pub mod protocol;

pub use protocol::{
    CertificateDto, CompareResponse, CompareResult, ErrorBody, ErrorResponse, HealthResponse,
    HealthResult, PROTOCOL_VERSION, SolveResponse, SolveResultDto, SplitMergePlanDto, TradeDto,
    finite_or_null, finite_or_null_vec,
};
