//! Convex-flow solver core for `ForecastFlows`.
//!
//! Narrow v1 scope: closed-enum edges (`UniV3`, `SplitMerge`), a single
//! `EndowmentLinear` objective, bounded L-BFGS-B dual solve with
//! Moreau-Yosida smoothing, doubling + bisection on the split bound,
//! certification, and primal recovery. No generic AD, no exact BFGS,
//! no `Markowitz`, no `ConstantProduct`.
