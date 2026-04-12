//! Closed-enum hyperedge dispatch for the bounded dual solver.
//!
//! Julia parity target: `src/convex_flows.jl` `Edge{T}` + the per-edge
//! dispatch inside `src/solver.jl` `find_arb!` (Vis_zero branch at
//! line 226) and `_netflow_from_edges!` (line 235).
//!
//! v1 carries only the two edge kinds the `compare_prediction_market_families`
//! benchmark path uses: a two-node `UniV3` CFMM and the `(n+1)`-node
//! `SplitMerge` hyperedge. Each variant owns its own node index list
//! (`Aᵢ` in Julia). The adapter is intentionally non-generic and does a
//! `match` in the gather/scatter hot path; the closed enum keeps the
//! dispatch inline-friendly and avoids trait-object overhead.

use crate::split_merge::SplitMerge;
use crate::uni_v3::UniV3;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EdgeError {
    #[error("UniV3 edge must have exactly 2 incident nodes")]
    UniV3NodeCount,
    #[error("SplitMerge edge node count must match its SplitMerge shape")]
    SplitMergeNodeMismatch,
    #[error("edge node index {0} is out of range for an n = {1} problem")]
    NodeOutOfRange(usize, usize),
    #[error("edge has duplicate node indices")]
    DuplicateNodes,
}

/// A two-node UniV3 edge: nodes are `[collateral_node, outcome_node]`
/// (in that order — matches the `[ν₀, ν₁]` convention of `UniV3::find_arb`).
#[derive(Debug, Clone)]
pub struct UniV3Edge {
    nodes: [usize; 2],
    inner: UniV3,
}

impl UniV3Edge {
    pub fn new(nodes: [usize; 2], inner: UniV3) -> Result<Self, EdgeError> {
        if nodes[0] == nodes[1] {
            return Err(EdgeError::DuplicateNodes);
        }
        Ok(Self { nodes, inner })
    }

    #[inline]
    pub fn nodes(&self) -> &[usize; 2] {
        &self.nodes
    }

    #[inline]
    pub fn inner(&self) -> &UniV3 {
        &self.inner
    }
}

/// An `(n+1)`-node split/merge hyperedge: nodes are
/// `[collateral_node, outcome_nodes...]`.
#[derive(Debug, Clone)]
pub struct SplitMergeEdge {
    nodes: Vec<usize>,
    inner: SplitMerge,
}

impl SplitMergeEdge {
    pub fn new(nodes: Vec<usize>, inner: SplitMerge) -> Result<Self, EdgeError> {
        if nodes.len() != inner.n_nodes() {
            return Err(EdgeError::SplitMergeNodeMismatch);
        }
        let mut seen = nodes.clone();
        seen.sort_unstable();
        for w in seen.windows(2) {
            if w[0] == w[1] {
                return Err(EdgeError::DuplicateNodes);
            }
        }
        Ok(Self { nodes, inner })
    }

    #[inline]
    pub fn nodes(&self) -> &[usize] {
        &self.nodes
    }

    #[inline]
    pub fn inner(&self) -> &SplitMerge {
        &self.inner
    }
}

/// Closed-enum edge dispatch used by `BoundedDualProblem`.
#[derive(Debug, Clone)]
pub enum Edge {
    UniV3(UniV3Edge),
    SplitMerge(SplitMergeEdge),
}

impl Edge {
    /// Incident node indices (`Aᵢ` in Julia). Ordering matters: it must
    /// match the per-variant convention of the wrapped `find_arb` so the
    /// gather/scatter is index-consistent.
    pub fn nodes(&self) -> &[usize] {
        match self {
            Edge::UniV3(e) => &e.nodes,
            Edge::SplitMerge(e) => &e.nodes,
        }
    }

    /// Number of incident nodes (degree of the hyperedge).
    #[inline]
    pub fn degree(&self) -> usize {
        self.nodes().len()
    }

    /// Per-edge closed-form oracle. Writes the optimal net flow into `x`
    /// for the gathered shadow prices `eta = ν[Aᵢ]`. Matches Julia
    /// `find_arb!(x, edge, view(ν, edge.Ai))`.
    pub fn find_arb(&self, x: &mut [f64], eta: &[f64]) {
        debug_assert_eq!(x.len(), self.degree());
        debug_assert_eq!(eta.len(), self.degree());
        match self {
            Edge::UniV3(e) => {
                let mut buf = [0.0_f64; 2];
                e.inner.find_arb(&mut buf, [eta[0], eta[1]]);
                x[0] = buf[0];
                x[1] = buf[1];
            }
            Edge::SplitMerge(e) => {
                e.inner.find_arb(x, eta);
            }
        }
    }

    /// Validate that every incident node index lies in `0..n`.
    pub fn validate(&self, n: usize) -> Result<(), EdgeError> {
        for &i in self.nodes() {
            if i >= n {
                return Err(EdgeError::NodeOutOfRange(i, n));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn univ3_edge_rejects_duplicate_nodes() {
        let pool = UniV3::new(1.0, vec![2.0], vec![100.0], 1.0).unwrap();
        assert!(matches!(
            UniV3Edge::new([0, 0], pool).unwrap_err(),
            EdgeError::DuplicateNodes
        ));
    }

    #[test]
    fn splitmerge_edge_rejects_node_count_mismatch() {
        let sm = SplitMerge::new(3, 1.0, 0.0).unwrap();
        assert!(matches!(
            SplitMergeEdge::new(vec![0, 1], sm).unwrap_err(),
            EdgeError::SplitMergeNodeMismatch
        ));
    }

    #[test]
    fn splitmerge_edge_rejects_duplicate_nodes() {
        let sm = SplitMerge::new(3, 1.0, 0.0).unwrap();
        assert!(matches!(
            SplitMergeEdge::new(vec![0, 1, 1], sm).unwrap_err(),
            EdgeError::DuplicateNodes
        ));
    }

    #[test]
    fn edge_validate_flags_out_of_range_nodes() {
        let pool = UniV3::new(1.0, vec![2.0], vec![100.0], 1.0).unwrap();
        let edge = Edge::UniV3(UniV3Edge::new([0, 5], pool).unwrap());
        assert!(matches!(
            edge.validate(3).unwrap_err(),
            EdgeError::NodeOutOfRange(5, 3)
        ));
    }

    #[test]
    fn univ3_edge_find_arb_roundtrips_through_enum() {
        let pool = UniV3::new(1.0, vec![2.0], vec![100.0], 1.0).unwrap();
        let edge = Edge::UniV3(UniV3Edge::new([0, 1], pool).unwrap());
        let mut x = [0.0_f64, 0.0];
        edge.find_arb(&mut x, &[1.0, 1.0]);
        // γ = 1, p = 1 = current_price → no-trade.
        assert!(x[0].abs() < 1e-15 && x[1].abs() < 1e-15);
    }

    #[test]
    fn splitmerge_edge_find_arb_roundtrips_through_enum() {
        let sm = SplitMerge::new(3, 5.0, 0.0).unwrap();
        let edge = Edge::SplitMerge(SplitMergeEdge::new(vec![0, 1, 2], sm).unwrap());
        let mut x = vec![0.0; 3];
        // gap = 0.6 + 0.6 - 1.0 = 0.2 > tol → mint at B = 5.
        edge.find_arb(&mut x, &[1.0, 0.6, 0.6]);
        assert!((x[0] + 5.0).abs() < 1e-12);
        assert!((x[1] - 5.0).abs() < 1e-12);
        assert!((x[2] - 5.0).abs() < 1e-12);
    }
}
