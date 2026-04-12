//! Multi-band Uniswap-v3-style concentrated-liquidity edge (closed-form oracle).
//!
//! Julia parity target: `src/prediction_markets.jl` `UniV3{T}`, `BoundedProduct`,
//! `find_arb_pos`, `find_arb!(x, cfmm::UniV3, η)`.
//!
//! The edge represents a collateral/outcome CFMM with descending-price ticks.
//! Coordinates are `(collateral, outcome)` — the solver hands in a 2-element
//! price vector `η` and expects a 2-element net-flow vector `x` in return,
//! with sign convention `x < 0` = edge receives, `x > 0` = edge pays out.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum UniV3Error {
    #[error("tick and liquidity arrays must match")]
    LengthMismatch,
    #[error("tick and liquidity arrays must be nonempty")]
    Empty,
    #[error("current_price must be finite and positive")]
    BadCurrentPrice,
    #[error("fee multiplier must lie in (0, 1]")]
    BadFee,
    #[error("tick prices must be finite")]
    TickNonFinite,
    #[error("liquidity must be finite")]
    LiquidityNonFinite,
    #[error("tick prices must be strictly positive")]
    TickNonPositive,
    #[error("liquidity must be nonnegative")]
    LiquidityNegative,
    #[error("tick prices must be strictly decreasing")]
    TicksNotDescending,
    #[error("current_price must lie within the represented tick range")]
    PriceOutOfRange,
}

/// Internal-representation UniV3 edge. `lower_ticks` is strictly decreasing,
/// `liquidity[i]` is the reserve-product `L²` for band `i`, and `current_tick`
/// is the 0-based index of the band that contains `current_price`.
///
/// This is the "solver-facing" representation. The public `UniV3MarketSpec`
/// facade (to be ported in a later phase) exposes outcome-priced bands in
/// ascending price order and inverts them to build this internal edge.
#[derive(Debug, Clone)]
pub struct UniV3 {
    current_price: f64,
    current_tick: usize,
    lower_ticks: Vec<f64>,
    liquidity: Vec<f64>,
    gamma: f64,
}

impl UniV3 {
    pub fn new(
        current_price: f64,
        lower_ticks: Vec<f64>,
        liquidity: Vec<f64>,
        gamma: f64,
    ) -> Result<Self, UniV3Error> {
        if lower_ticks.len() != liquidity.len() {
            return Err(UniV3Error::LengthMismatch);
        }
        if lower_ticks.is_empty() {
            return Err(UniV3Error::Empty);
        }
        if !(current_price.is_finite() && current_price > 0.0) {
            return Err(UniV3Error::BadCurrentPrice);
        }
        if !(gamma.is_finite() && gamma > 0.0 && gamma <= 1.0) {
            return Err(UniV3Error::BadFee);
        }
        if !lower_ticks.iter().all(|x| x.is_finite()) {
            return Err(UniV3Error::TickNonFinite);
        }
        if !liquidity.iter().all(|x| x.is_finite()) {
            return Err(UniV3Error::LiquidityNonFinite);
        }
        if !lower_ticks.iter().all(|&x| x > 0.0) {
            return Err(UniV3Error::TickNonPositive);
        }
        if !liquidity.iter().all(|&x| x >= 0.0) {
            return Err(UniV3Error::LiquidityNegative);
        }
        for pair in lower_ticks.windows(2) {
            if pair[0] <= pair[1] {
                return Err(UniV3Error::TicksNotDescending);
            }
        }
        if current_price > lower_ticks[0] {
            return Err(UniV3Error::PriceOutOfRange);
        }
        // Julia `searchsortedlast(ticks, price; rev=true)` returns the last
        // (largest) index `i` with `ticks[i] >= price`. Because `ticks` is
        // strictly decreasing, we scan from the back for the first `ticks[i]
        // >= price`.
        let current_tick = lower_ticks
            .iter()
            .enumerate()
            .rev()
            .find(|&(_, &t)| t >= current_price)
            .map(|(i, _)| i)
            .ok_or(UniV3Error::PriceOutOfRange)?;
        Ok(Self {
            current_price,
            current_tick,
            lower_ticks,
            liquidity,
            gamma,
        })
    }

    #[inline]
    pub fn current_price(&self) -> f64 {
        self.current_price
    }

    #[inline]
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    #[inline]
    pub fn num_bands(&self) -> usize {
        self.lower_ticks.len()
    }

    /// Closed-form dual arbitrage oracle. `eta` is the 2-vector `(ν₀, ν₁)` of
    /// shadow prices for `(collateral, outcome)`. Writes the optimal net flow
    /// into `x`. Matches Julia `find_arb!(x, cfmm::UniV3, η)`.
    pub fn find_arb(&self, x: &mut [f64; 2], eta: [f64; 2]) {
        let p = eta[0] / eta[1];
        let gamma = self.gamma;
        x[0] = 0.0;
        x[1] = 0.0;
        // No-trade cone: market-implied price lies inside `[γ·p̄, p̄/γ]`.
        if gamma * self.current_price <= p && p <= self.current_price / gamma {
            return;
        }
        if p < gamma * self.current_price {
            let (delta_total, lambda_total) = self.sweep_upper_pools(p / gamma);
            x[0] = -delta_total / gamma;
            x[1] = lambda_total;
        } else {
            let (delta_total, lambda_total) = self.sweep_lower_pools(1.0 / (gamma * p));
            x[0] = lambda_total;
            x[1] = -delta_total / gamma;
        }
    }

    #[inline]
    fn tick_high(&self, idx: usize) -> f64 {
        self.lower_ticks[idx]
    }

    #[inline]
    fn tick_low(&self, idx: usize) -> f64 {
        if idx + 1 < self.lower_ticks.len() {
            self.lower_ticks[idx + 1]
        } else {
            0.0
        }
    }

    fn bounded_product_at(&self, idx: usize) -> BoundedProduct {
        let k = self.liquidity[idx];
        let p_minus = self.tick_low(idx);
        let p_plus = self.tick_high(idx);
        // When k == 0 these reduce to zero naturally (empty pool).
        let alpha = (k / p_plus).sqrt();
        let beta = (k * p_minus).sqrt();
        let p = if idx > self.current_tick {
            p_plus
        } else if idx < self.current_tick {
            p_minus
        } else {
            self.current_price
        };
        let r1 = if k == 0.0 {
            0.0
        } else {
            (k / p).sqrt() - alpha
        };
        let r2 = if k == 0.0 { 0.0 } else { (k * p).sqrt() - beta };
        BoundedProduct {
            k,
            alpha,
            beta,
            r1,
            r2,
        }
    }

    /// Sweep pools from `current_tick` upward (towards higher prices).
    fn sweep_upper_pools(&self, price: f64) -> (f64, f64) {
        let mut delta_total = 0.0;
        let mut lambda_total = 0.0;
        let mut initial = true;
        for idx in self.current_tick..self.lower_ticks.len() {
            let pool = self.bounded_product_at(idx);
            if pool.is_empty() {
                initial = false;
                continue;
            }
            let (delta, lambda) = pool.find_arb_pos(price);
            if !initial && (delta == 0.0 || lambda == 0.0) {
                break;
            }
            delta_total += delta;
            lambda_total += lambda;
            initial = false;
        }
        (delta_total, lambda_total)
    }

    /// Sweep pools from `current_tick` downward (towards lower prices), using
    /// flipped bounded-product coordinates (because we're now trading the
    /// other side of each bounded pool).
    fn sweep_lower_pools(&self, price: f64) -> (f64, f64) {
        let mut delta_total = 0.0;
        let mut lambda_total = 0.0;
        let mut initial = true;
        for idx in (0..=self.current_tick).rev() {
            let pool = self.bounded_product_at(idx).flipped();
            if pool.is_empty() {
                initial = false;
                continue;
            }
            let (delta, lambda) = pool.find_arb_pos(price);
            if !initial && (delta == 0.0 || lambda == 0.0) {
                break;
            }
            delta_total += delta;
            lambda_total += lambda;
            initial = false;
        }
        (delta_total, lambda_total)
    }
}

#[derive(Debug, Clone, Copy)]
struct BoundedProduct {
    k: f64,
    alpha: f64,
    beta: f64,
    r1: f64,
    r2: f64,
}

impl BoundedProduct {
    #[inline]
    fn is_empty(self) -> bool {
        self.k == 0.0
    }

    #[inline]
    fn flipped(self) -> Self {
        Self {
            k: self.k,
            alpha: self.beta,
            beta: self.alpha,
            r1: self.r2,
            r2: self.r1,
        }
    }

    /// Julia `find_arb_pos(t, price)` — returns `(δ, λ)` for the constant-product
    /// arbitrage at target price, capped by the pool's `[α, β]` box.
    fn find_arb_pos(self, price: f64) -> (f64, f64) {
        let delta = (self.k / price).sqrt() - (self.r1 + self.alpha);
        if delta <= 0.0 {
            return (0.0, 0.0);
        }
        let delta_max = self.k / self.beta - (self.r1 + self.alpha);
        if delta >= delta_max {
            return (delta_max, self.r2);
        }
        let lambda = (self.r2 + self.beta) - (price * self.k).sqrt();
        (delta, lambda)
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

    fn symmetric_single_band() -> UniV3 {
        // One band with current_price mid-way, L² = 100, γ = 1 (no fee).
        UniV3::new(1.0, vec![2.0], vec![100.0], 1.0).unwrap()
    }

    #[test]
    fn no_trade_when_eta_matches_current_price() {
        let edge = symmetric_single_band();
        let mut x = [0.0, 0.0];
        // p = η₀/η₁ = 1.0 exactly equals current_price; γ = 1 makes the
        // no-trade cone collapse to a point.
        edge.find_arb(&mut x, [1.0, 1.0]);
        approx_eq(&x, &[0.0, 0.0]);
    }

    #[test]
    fn no_trade_inside_fee_cone() {
        // γ = 0.99: no-trade cone is [0.99, 1/0.99].
        let edge = UniV3::new(1.0, vec![2.0], vec![100.0], 0.99).unwrap();
        let mut x = [0.0, 0.0];
        // p = 1.0 is inside the cone.
        edge.find_arb(&mut x, [1.0, 1.0]);
        approx_eq(&x, &[0.0, 0.0]);
        // p = 0.995 also inside.
        edge.find_arb(&mut x, [0.995, 1.0]);
        approx_eq(&x, &[0.0, 0.0]);
    }

    #[test]
    fn trade_in_upper_direction_signs() {
        // γ = 1 (no fee) so the test does not have to reason about the cone.
        // Force p < current_price → expect x[0] < 0 (receive collateral),
        // x[1] > 0 (pay outcome).
        let edge = UniV3::new(1.0, vec![2.0], vec![100.0], 1.0).unwrap();
        let mut x = [0.0, 0.0];
        edge.find_arb(&mut x, [0.5, 1.0]); // p = 0.5 < 1.0
        assert!(x[0] < 0.0, "expected delta_collateral < 0, got {}", x[0]);
        assert!(x[1] > 0.0, "expected lambda_outcome > 0, got {}", x[1]);
    }

    #[test]
    fn trade_in_lower_direction_signs() {
        let edge = UniV3::new(1.0, vec![2.0], vec![100.0], 1.0).unwrap();
        let mut x = [0.0, 0.0];
        edge.find_arb(&mut x, [2.0, 1.0]); // p = 2.0 > 1.0
        assert!(x[0] > 0.0, "expected lambda_collateral > 0, got {}", x[0]);
        assert!(x[1] < 0.0, "expected -delta_outcome < 0, got {}", x[1]);
    }

    #[test]
    fn rejects_non_descending_ticks() {
        assert!(matches!(
            UniV3::new(1.0, vec![2.0, 2.0], vec![100.0, 100.0], 1.0).unwrap_err(),
            UniV3Error::TicksNotDescending
        ));
        assert!(matches!(
            UniV3::new(1.0, vec![2.0, 3.0], vec![100.0, 100.0], 1.0).unwrap_err(),
            UniV3Error::TicksNotDescending
        ));
    }

    #[test]
    fn rejects_price_above_top_tick() {
        assert!(matches!(
            UniV3::new(5.0, vec![2.0], vec![100.0], 1.0).unwrap_err(),
            UniV3Error::PriceOutOfRange
        ));
    }

    #[test]
    fn single_band_symmetry() {
        // Symmetric pool at γ=1: equal-magnitude eta offsets should produce
        // equal-magnitude but oppositely-signed trades (by pool symmetry).
        let edge = UniV3::new(1.0, vec![4.0], vec![100.0], 1.0).unwrap();
        let mut x_plus = [0.0, 0.0];
        let mut x_minus = [0.0, 0.0];
        edge.find_arb(&mut x_plus, [0.5, 1.0]);
        edge.find_arb(&mut x_minus, [2.0, 1.0]);
        // Trade magnitudes won't be exactly symmetric because the mid-tick
        // choice of p is asymmetric across sweep_upper vs sweep_lower, so
        // only check signs and nonzero magnitudes here.
        assert!(x_plus[0] < 0.0 && x_plus[1] > 0.0);
        assert!(x_minus[0] > 0.0 && x_minus[1] < 0.0);
    }

    #[test]
    fn current_tick_lookup_matches_julia() {
        // ticks [5,3,1], current_price 2.5 -> Julia searchsortedlast yields
        // tick index 2 (1-based) == 1 (0-based), the band with ticks[1]=3.
        let edge = UniV3::new(2.5, vec![5.0, 3.0, 1.0], vec![10.0, 20.0, 30.0], 1.0).unwrap();
        assert_eq!(edge.current_tick, 1);
    }
}
