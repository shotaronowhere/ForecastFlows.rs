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

#![allow(
    clippy::missing_panics_doc,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::collapsible_if,
    clippy::match_wildcard_for_single_variants,
    clippy::too_many_arguments
)]

pub mod problem;
pub mod protocol;
pub mod request;
pub mod result;
pub mod solve;
pub mod worker;
pub mod workspace;

pub use problem::{
    Mode, OutcomeSpec, PredictionMarketProblem, ProblemError, UniV3Band, UniV3MarketSpec,
    analytical_split_bound, build_edges, build_objective, default_split_bound,
};
pub use protocol::{
    CertificateDto, CompareResponse, CompareResult, ErrorBody, ErrorResponse, HealthResponse,
    HealthResult, PROTOCOL_VERSION, SolveResponse, SolveResultDto, SplitMergePlanDto, TradeDto,
    finite_or_null, finite_or_null_vec,
};
pub use request::{ParsedRequest, RequestError, parse_protocol_request};
pub use result::{
    compare_prediction_market_families, compare_prediction_market_families_with_workspace,
    extract_solve_result,
};
pub use solve::{SolveError, SolveOptions, SolveOutcome, solve, solve_with_seed};
pub use worker::{
    PACKAGE_VERSION, handle_protocol_json, handle_protocol_json_with_workspace_cache,
};
pub use workspace::{Layout, PredictionMarketWorkspace, worker_compare_workspace};
