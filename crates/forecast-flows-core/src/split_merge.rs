//! Fee-free mint/merge hyperedge for the prediction-market split/merge family.
//!
//! Julia parity target: `src/prediction_markets.jl` `SplitMergeEdge{T}`,
//! `splitmerge_gap`, `splitmerge_flow!`, `find_arb!(x, e::SplitMergeEdge, η)`.
//!
//! Node ordering is `[collateral, outcomes...]`. A flow parameter `w ∈ [-B, B]`
//! represents: `w > 0` = mint `w` complete sets (spend `w` collateral, receive
//! `w` of each outcome); `w < 0` = merge.
//!
//! When `μ > 0` the support function is Moreau-Yosida smoothed:
//! `fₛₘᵘ(η) = max_{w ∈ [-B, B]} { w·gap − (μ/2)·w² }`, giving the closed-form
//! oracle `w* = clamp(gap/μ, -B, B)`. This makes the dual C¹ and lets L-BFGS-B
//! converge without the nonsmooth bang-bang discontinuity.

use crate::tolerances::sqrt_eps_f64;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SplitMergeError {
    #[error("SplitMergeEdge requires at least two outcomes (n_nodes >= 3)")]
    TooFewNodes,
    #[error("bound B must be finite and positive")]
    BadBound,
    #[error("smoothing parameter μ must be finite and nonnegative")]
    BadSmoothing,
}

/// Split/merge hyperedge. `n_nodes = 1 + n_outcomes` (collateral + outcomes).
#[derive(Debug, Clone, Copy)]
pub struct SplitMerge {
    n_nodes: usize,
    b: f64,
    mu: f64,
}

impl SplitMerge {
    pub fn new(n_nodes: usize, b: f64, mu: f64) -> Result<Self, SplitMergeError> {
        if n_nodes < 3 {
            return Err(SplitMergeError::TooFewNodes);
        }
        if !(b.is_finite() && b > 0.0) {
            return Err(SplitMergeError::BadBound);
        }
        if !(mu.is_finite() && mu >= 0.0) {
            return Err(SplitMergeError::BadSmoothing);
        }
        Ok(Self { n_nodes, b, mu })
    }

    #[inline]
    pub fn n_nodes(self) -> usize {
        self.n_nodes
    }

    #[inline]
    pub fn bound(self) -> f64 {
        self.b
    }

    #[inline]
    pub fn smoothing(self) -> f64 {
        self.mu
    }

    #[inline]
    pub fn is_nonsmooth(self) -> bool {
        self.mu == 0.0
    }

    /// Dual gap `Σᵢ>0 ηᵢ − η₀`. Matches Julia `splitmerge_gap(η)`.
    #[inline]
    pub fn gap(self, eta: &[f64]) -> f64 {
        assert_eq!(eta.len(), self.n_nodes, "eta length mismatch");
        let mut acc = 0.0;
        for v in &eta[1..] {
            acc += *v;
        }
        acc - eta[0]
    }

    /// Writes the `w`-parameterized flow into `x`: `x[0] = -w`, `x[1..] = w`.
    pub fn set_flow(self, x: &mut [f64], w: f64) {
        assert_eq!(x.len(), self.n_nodes, "x length mismatch");
        x[0] = -w;
        for slot in &mut x[1..] {
            *slot = w;
        }
    }

    /// Closed-form dual arbitrage oracle. For `μ = 0` it's the bang-bang
    /// support function of the `[-B, B]` box; for `μ > 0` it's the smoothed
    /// clamp. Matches Julia `find_arb!(x, e::SplitMergeEdge, η)`.
    pub fn find_arb(self, x: &mut [f64], eta: &[f64]) {
        let gap = self.gap(eta);
        if self.mu == 0.0 {
            let tol = sqrt_eps_f64();
            if gap > tol {
                self.set_flow(x, self.b);
            } else if gap < -tol {
                self.set_flow(x, -self.b);
            } else {
                x.fill(0.0);
            }
        } else {
            let w = (gap / self.mu).clamp(-self.b, self.b);
            if w.abs() < sqrt_eps_f64() {
                x.fill(0.0);
            } else {
                self.set_flow(x, w);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < 1e-12, "slot {i}: actual={a} expected={e}");
        }
    }

    #[test]
    fn rejects_too_few_nodes() {
        assert!(matches!(
            SplitMerge::new(2, 1.0, 0.0).unwrap_err(),
            SplitMergeError::TooFewNodes
        ));
    }

    #[test]
    fn rejects_non_positive_bound() {
        assert!(matches!(
            SplitMerge::new(3, 0.0, 0.0).unwrap_err(),
            SplitMergeError::BadBound
        ));
        assert!(matches!(
            SplitMerge::new(3, -1.0, 0.0).unwrap_err(),
            SplitMergeError::BadBound
        ));
    }

    #[test]
    fn nonsmooth_mints_when_outcome_prices_exceed_collateral() {
        let e = SplitMerge::new(3, 5.0, 0.0).unwrap();
        // eta = (collateral, out1, out2) with sum(outs) > eta[0] → mint.
        let mut x = [0.0; 3];
        e.find_arb(&mut x, &[1.0, 0.6, 0.6]);
        approx_eq(&x, &[-5.0, 5.0, 5.0]);
    }

    #[test]
    fn nonsmooth_merges_when_outcome_prices_below_collateral() {
        let e = SplitMerge::new(3, 5.0, 0.0).unwrap();
        let mut x = [0.0; 3];
        e.find_arb(&mut x, &[1.0, 0.3, 0.3]);
        approx_eq(&x, &[5.0, -5.0, -5.0]);
    }

    #[test]
    fn nonsmooth_idle_on_zero_gap() {
        let e = SplitMerge::new(3, 5.0, 0.0).unwrap();
        let mut x = [0.0; 3];
        e.find_arb(&mut x, &[1.0, 0.5, 0.5]); // gap = 0
        approx_eq(&x, &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn smoothed_scales_with_mu() {
        let e = SplitMerge::new(3, 10.0, 1e-3).unwrap();
        // gap = 0.2 - 0.0 = 0.2, μ = 1e-3 → w = clamp(200, -10, 10) = 10.
        let mut x = [0.0; 3];
        e.find_arb(&mut x, &[0.0, 0.1, 0.1]);
        approx_eq(&x, &[-10.0, 10.0, 10.0]);
    }

    #[test]
    fn smoothed_interior_clamp() {
        let e = SplitMerge::new(3, 10.0, 1.0).unwrap();
        // gap = 0.4, μ = 1.0 → w = 0.4 (interior, not clamped).
        let mut x = [0.0; 3];
        e.find_arb(&mut x, &[0.0, 0.3, 0.1]);
        assert!((x[0] + 0.4).abs() < 1e-12);
        assert!((x[1] - 0.4).abs() < 1e-12);
        assert!((x[2] - 0.4).abs() < 1e-12);
    }

    #[test]
    fn smoothed_snaps_tiny_to_zero() {
        let e = SplitMerge::new(3, 10.0, 1.0).unwrap();
        // gap ≈ 1e-12, μ=1 → |w| = 1e-12 < sqrt_eps → snap to zero.
        let mut x = [0.0; 3];
        e.find_arb(&mut x, &[0.0, 1e-13, 1e-13]);
        approx_eq(&x, &[0.0, 0.0, 0.0]);
    }
}
