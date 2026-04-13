//! Workspace cache for repeated solves over a fixed prediction-market topology.
//!
//! Julia parity target: `PredictionMarketWorkspace`,
//! `_PredictionMarketLayout`, `_prediction_market_layout`,
//! `_prediction_market_compatible_layout`, `_workspace_seed`,
//! `_store_workspace_seed!`, `_worker_compare_workspace!` in
//! `src/prediction_market_api.jl`.
//!
//! Scope difference vs. Julia. The Julia workspace caches both the dual
//! seed and the pre-allocated `Solver` buffer (`direct_solver` /
//! `mixed_solver`). The Rust solver has no equivalent stateful buffer —
//! `BoundedDualProblem` is a lightweight value type rebuilt cheaply per
//! call — so this port caches only the layout snapshot and the dual seeds.
//! That is the load-bearing optimization in the compare path: subsequent
//! compatible compares warm-start L-BFGS-B from the prior solve's converged
//! `ν`, cutting iterations on the second-and-later request.
//!
//! The layout snapshot enables the `_worker_compare_workspace!` reuse
//! check: only requests with byte-identical outcome ids, market ids,
//! market→outcome wiring, and band counts may reuse the cached seeds.
//! Anything else triggers a fresh workspace.

use crate::problem::PredictionMarketProblem;

/// Topology snapshot used for compatible-compare reuse. Matches Julia
/// `_PredictionMarketLayout` minus `market_kinds` (v1 has only `UniV3`) and
/// minus `outcome_index_by_id` (derivable from `outcome_ids`, never read by
/// the compatibility check).
#[derive(Debug, Clone)]
pub struct Layout {
    outcome_ids: Vec<String>,
    market_ids: Vec<String>,
    market_outcome_ids: Vec<String>,
    market_band_counts: Vec<usize>,
}

impl Layout {
    /// Snapshot the topology of `problem`. Mirrors Julia
    /// `_prediction_market_layout`.
    pub fn from_problem(problem: &PredictionMarketProblem) -> Self {
        Self {
            outcome_ids: problem.outcomes().iter().map(|o| o.outcome_id().to_string()).collect(),
            market_ids: problem.markets().iter().map(|m| m.market_id().to_string()).collect(),
            market_outcome_ids: problem
                .markets()
                .iter()
                .map(|m| m.outcome_id().to_string())
                .collect(),
            market_band_counts: problem.markets().iter().map(|m| m.bands().len()).collect(),
        }
    }

    /// True iff the cached layout matches `problem`'s topology. Mirrors
    /// Julia `_prediction_market_compatible_layout`.
    pub fn is_compatible_with(&self, problem: &PredictionMarketProblem) -> bool {
        let other = Self::from_problem(problem);
        self.outcome_ids == other.outcome_ids
            && self.market_ids == other.market_ids
            && self.market_outcome_ids == other.market_outcome_ids
            && self.market_band_counts == other.market_band_counts
    }

    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.outcome_ids.len() + 1
    }
}

/// Reusable workspace for repeated solves over a fixed prediction-market
/// topology. Holds the topology snapshot plus per-mode dual seeds promoted
/// from prior solves.
#[derive(Debug, Clone)]
pub struct PredictionMarketWorkspace {
    layout: Layout,
    direct_seed: Option<Vec<f64>>,
    mixed_seed: Option<Vec<f64>>,
}

impl PredictionMarketWorkspace {
    /// Construct an empty workspace pinned to the topology of
    /// `problem_template`. Seeds populate after the first solve in each mode.
    pub fn new(problem_template: &PredictionMarketProblem) -> Self {
        Self {
            layout: Layout::from_problem(problem_template),
            direct_seed: None,
            mixed_seed: None,
        }
    }

    #[inline]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    #[inline]
    pub fn is_compatible_with(&self, problem: &PredictionMarketProblem) -> bool {
        self.layout.is_compatible_with(problem)
    }

    #[inline]
    pub fn direct_seed(&self) -> Option<&[f64]> {
        self.direct_seed.as_deref()
    }

    #[inline]
    pub fn mixed_seed(&self) -> Option<&[f64]> {
        self.mixed_seed.as_deref()
    }

    /// Store `nu` as the next direct-mode seed. Mirrors Julia
    /// `_store_workspace_seed!(workspace, :direct_only, nu)`.
    pub fn store_direct_seed(&mut self, nu: &[f64]) {
        self.direct_seed = Some(nu.to_vec());
    }

    /// Store `nu` as the next mixed-mode seed. Mirrors Julia
    /// `_store_workspace_seed!(workspace, :mixed_enabled, nu)`.
    pub fn store_mixed_seed(&mut self, nu: &[f64]) {
        self.mixed_seed = Some(nu.to_vec());
    }
}

/// Get-or-rebuild a workspace for the compare path. Mirrors Julia
/// `_worker_compare_workspace!`: returns the cached workspace if its layout
/// matches `problem`, else replaces it with a fresh one and returns that.
pub fn worker_compare_workspace<'a>(
    cached: &'a mut Option<PredictionMarketWorkspace>,
    problem: &PredictionMarketProblem,
) -> &'a mut PredictionMarketWorkspace {
    let needs_rebuild = match cached {
        Some(ws) => !ws.is_compatible_with(problem),
        None => true,
    };
    if needs_rebuild {
        *cached = Some(PredictionMarketWorkspace::new(problem));
    }
    cached.as_mut().expect("just initialized")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{OutcomeSpec, UniV3Band, UniV3MarketSpec};

    fn problem(market_id: &str, band_count: usize) -> PredictionMarketProblem {
        let outcomes = vec![
            OutcomeSpec::new("a", 0.5, 0.0).unwrap(),
            OutcomeSpec::new("b", 0.5, 0.0).unwrap(),
        ];
        // Always include the top (0.99) and bottom (0.20) ticks so
        // current_price=0.4 sits inside the band range regardless of how
        // many middle ticks are present.
        let bands: Vec<UniV3Band> = match band_count {
            2 => vec![
                UniV3Band::new(0.99, 100.0).unwrap(),
                UniV3Band::new(0.20, 100.0).unwrap(),
            ],
            3 => vec![
                UniV3Band::new(0.99, 100.0).unwrap(),
                UniV3Band::new(0.50, 100.0).unwrap(),
                UniV3Band::new(0.20, 100.0).unwrap(),
            ],
            _ => panic!("test fixture only supports band_count in {{2, 3}}"),
        };
        let market = UniV3MarketSpec::new(market_id, "a", 0.4, bands, 1.0).unwrap();
        PredictionMarketProblem::new(outcomes, 1.0, vec![market], None).unwrap()
    }

    #[test]
    fn layout_matches_same_topology() {
        let p1 = problem("m", 2);
        let p2 = problem("m", 2);
        let layout = Layout::from_problem(&p1);
        assert!(layout.is_compatible_with(&p2));
    }

    #[test]
    fn layout_rejects_different_market_id() {
        let p1 = problem("m1", 2);
        let p2 = problem("m2", 2);
        assert!(!Layout::from_problem(&p1).is_compatible_with(&p2));
    }

    #[test]
    fn layout_rejects_different_band_count() {
        let p1 = problem("m", 2);
        let p2 = problem("m", 3);
        assert!(!Layout::from_problem(&p1).is_compatible_with(&p2));
    }

    #[test]
    fn worker_compare_workspace_rebuilds_on_mismatch() {
        let p1 = problem("m1", 2);
        let p2 = problem("m2", 2);
        let mut cached: Option<PredictionMarketWorkspace> = None;
        let ws1 = worker_compare_workspace(&mut cached, &p1);
        ws1.store_mixed_seed(&[1.0, 0.5, 0.5]);
        // Mismatched problem must drop the cached seeds.
        let ws2 = worker_compare_workspace(&mut cached, &p2);
        assert!(ws2.mixed_seed().is_none());
    }

    #[test]
    fn worker_compare_workspace_reuses_on_match() {
        let p1 = problem("m", 2);
        let p2 = problem("m", 2);
        let mut cached: Option<PredictionMarketWorkspace> = None;
        worker_compare_workspace(&mut cached, &p1).store_mixed_seed(&[1.0, 0.5, 0.5]);
        let ws2 = worker_compare_workspace(&mut cached, &p2);
        assert_eq!(ws2.mixed_seed(), Some(&[1.0, 0.5, 0.5][..]));
    }
}
