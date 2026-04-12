//! Convex-flow solver core for `ForecastFlows`.
//!
//! Narrow v1 scope: closed-enum edges (`UniV3`, `SplitMerge`), a single
//! `EndowmentLinear` objective, bounded L-BFGS-B dual solve with
//! Moreau-Yosida smoothing, doubling + bisection on the split bound,
//! certification, and primal recovery. No generic AD, no exact BFGS,
//! no `Markowitz`, no `ConstantProduct`.
//!
//! Current session (Phase 3 narrowed slice): data-layer ports only — edges
//! and the objective. The bounded dual solver and smoothing/certification
//! pipeline land in Phase 5.

#![allow(
    clippy::missing_panics_doc,
    clippy::doc_markdown,
    clippy::comparison_chain,
    clippy::needless_range_loop,
    clippy::float_cmp,
    clippy::cast_precision_loss
)]

pub mod edge;
pub mod objective;
pub mod solver;
pub mod split_merge;
pub mod tolerances;
pub mod uni_v3;

pub use edge::{Edge, EdgeError, SplitMergeEdge as SplitMergeEdgeHandle, UniV3Edge};
pub use objective::{EndowmentLinear, Objective, ObjectiveError};
pub use solver::{
    BoundedDualProblem, CertifyTolerances, DualSolution, SolveCertificate, SolveError,
    SolverOptions, certify, moreau_yosida_mu, recover_primal, splitmerge_bound_residual,
};
pub use split_merge::{SplitMerge, SplitMergeError};
pub use uni_v3::{UniV3, UniV3Error};
