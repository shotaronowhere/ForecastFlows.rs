//! Flow objective: v1 carries the single closed variant `EndowmentLinear`.
//!
//! Julia parity target: `src/objectives.jl` `EndowmentLinear{T}`,
//! `U`, `Ubar`, `∇Ubar!`, `lower_limit`, `upper_limit`,
//! `primal_lower_bounds`, `recovery_targets!`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ObjectiveError {
    #[error("value vector and endowment must have the same length")]
    LengthMismatch,
    #[error("value vector must be finite")]
    NonFiniteValue,
    #[error("endowment must be finite")]
    NonFiniteEndowment,
    #[error("endowment must be nonnegative")]
    NegativeEndowment,
}

/// Linear utility `Uᶜ(y) = cᵀ y` over the endowment polytope `y ≥ -h₀`.
#[derive(Debug, Clone)]
pub struct EndowmentLinear {
    c: Vec<f64>,
    h0: Vec<f64>,
}

impl EndowmentLinear {
    pub fn new(c: Vec<f64>, h0: Vec<f64>) -> Result<Self, ObjectiveError> {
        if c.len() != h0.len() {
            return Err(ObjectiveError::LengthMismatch);
        }
        if !c.iter().all(|x| x.is_finite()) {
            return Err(ObjectiveError::NonFiniteValue);
        }
        if !h0.iter().all(|x| x.is_finite()) {
            return Err(ObjectiveError::NonFiniteEndowment);
        }
        if !h0.iter().all(|&x| x >= 0.0) {
            return Err(ObjectiveError::NegativeEndowment);
        }
        Ok(Self { c, h0 })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.c.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.c.is_empty()
    }

    #[inline]
    pub fn c(&self) -> &[f64] {
        &self.c
    }

    #[inline]
    pub fn h0(&self) -> &[f64] {
        &self.h0
    }

    /// Primal utility `U(y)`: returns `-∞` if any `y[i] < -h₀[i] - atol`.
    /// Matches Julia `U(obj::EndowmentLinear, y)`.
    pub fn primal_utility(&self, y: &[f64]) -> f64 {
        assert_eq!(y.len(), self.c.len(), "y length mismatch");
        let atol = Self::tolerance_scale(&self.h0, y);
        for (yi, h0i) in y.iter().zip(&self.h0) {
            if *yi < -*h0i - atol {
                return f64::NEG_INFINITY;
            }
        }
        dot(&self.c, y)
    }

    /// Dual utility `Ūᶜ(ν)`: `+∞` unless `c ≤ ν` elementwise, else `(ν - c)ᵀ h₀`.
    pub fn dual_utility(&self, nu: &[f64]) -> f64 {
        assert_eq!(nu.len(), self.c.len(), "nu length mismatch");
        for (ci, ni) in self.c.iter().zip(nu) {
            if *ci > *ni {
                return f64::INFINITY;
            }
        }
        let mut acc = 0.0;
        for i in 0..self.c.len() {
            acc += (nu[i] - self.c[i]) * self.h0[i];
        }
        acc
    }

    /// Gradient of the dual utility written into `g`. If `ν[i] < c[i] - atol`,
    /// fills `g` with `+∞` and returns, matching Julia's `∇Ubar!`.
    pub fn dual_gradient_into(&self, g: &mut [f64], nu: &[f64]) {
        assert_eq!(g.len(), self.c.len(), "g length mismatch");
        assert_eq!(nu.len(), self.c.len(), "nu length mismatch");
        let atol = Self::tolerance_scale(&self.c, nu);
        for (i, (ci, ni)) in self.c.iter().zip(nu).enumerate() {
            if *ni < *ci - atol {
                for slot in g.iter_mut() {
                    *slot = f64::INFINITY;
                }
                return;
            }
            g[i] = self.h0[i];
        }
    }

    /// L-BFGS-B lower bound vector `c` (dual variable ≥ c).
    pub fn lower_limit(&self) -> Vec<f64> {
        self.c.clone()
    }

    /// L-BFGS-B upper bound vector (unbounded above).
    pub fn upper_limit(&self) -> Vec<f64> {
        vec![f64::INFINITY; self.c.len()]
    }

    /// Primal lower bounds `-h₀` used by the solver's recovery path.
    pub fn primal_lower_bounds(&self) -> Vec<f64> {
        self.h0.iter().map(|x| -*x).collect()
    }

    /// Recovery targets / fixed flags, matching Julia `recovery_targets!`.
    pub fn recovery_targets(&self, target: &mut [f64], fixed: &mut [bool], nu: &[f64]) {
        assert_eq!(target.len(), self.c.len());
        assert_eq!(fixed.len(), self.c.len());
        assert_eq!(nu.len(), self.c.len());
        let tol = Self::tolerance_scale(&self.c, nu);
        for (i, (ci, ni)) in self.c.iter().zip(nu).enumerate() {
            if *ni > *ci + tol {
                target[i] = -self.h0[i];
                fixed[i] = true;
            } else {
                target[i] = 0.0;
                fixed[i] = false;
            }
        }
    }

    /// `1000 · sqrt(eps(f64)) · max(1, max|a|, max|b|)`.
    fn tolerance_scale(a: &[f64], b: &[f64]) -> f64 {
        let max_a = a.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let max_b = b.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        1000.0 * f64::EPSILON.sqrt() * 1.0_f64.max(max_a).max(max_b)
    }
}

/// Closed enum per narrowed v1 scope; `EndowmentLinear` is the sole variant.
#[derive(Debug, Clone)]
pub enum Objective {
    EndowmentLinear(EndowmentLinear),
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_length_mismatch() {
        let err = EndowmentLinear::new(vec![1.0, 2.0], vec![1.0]).unwrap_err();
        assert!(matches!(err, ObjectiveError::LengthMismatch));
    }

    #[test]
    fn rejects_negative_endowment() {
        let err = EndowmentLinear::new(vec![1.0, 2.0], vec![0.0, -1.0]).unwrap_err();
        assert!(matches!(err, ObjectiveError::NegativeEndowment));
    }

    #[test]
    fn dual_utility_inside_domain() {
        let obj = EndowmentLinear::new(vec![1.0, 0.55, 0.45], vec![1.0, 2.0, 3.0]).unwrap();
        // nu > c componentwise -> finite
        let nu = [1.5, 0.6, 0.5];
        let expected = (1.5 - 1.0) * 1.0 + (0.6 - 0.55) * 2.0 + (0.5 - 0.45) * 3.0;
        assert!((obj.dual_utility(&nu) - expected).abs() < 1e-12);
    }

    #[test]
    fn dual_utility_outside_domain_is_infinite() {
        let obj = EndowmentLinear::new(vec![1.0, 0.55], vec![1.0, 2.0]).unwrap();
        let nu = [0.9, 0.6];
        assert!(obj.dual_utility(&nu).is_infinite());
    }

    #[test]
    fn dual_gradient_returns_h0_inside_domain() {
        let obj = EndowmentLinear::new(vec![1.0, 0.55], vec![1.0, 2.0]).unwrap();
        let mut g = [0.0, 0.0];
        obj.dual_gradient_into(&mut g, &[1.5, 0.6]);
        assert_eq!(g, [1.0, 2.0]);
    }

    #[test]
    fn dual_gradient_fills_inf_outside_domain() {
        let obj = EndowmentLinear::new(vec![1.0, 0.55], vec![1.0, 2.0]).unwrap();
        let mut g = [0.0, 0.0];
        obj.dual_gradient_into(&mut g, &[0.5, 0.6]);
        assert!(g.iter().all(|x| x.is_infinite()));
    }

    #[test]
    fn recovery_targets_flag_active_dual_coordinates() {
        let obj = EndowmentLinear::new(vec![1.0, 0.55], vec![1.0, 2.0]).unwrap();
        let mut target = [0.0, 0.0];
        let mut fixed = [false, false];
        obj.recovery_targets(&mut target, &mut fixed, &[1.5, 0.6]);
        assert_eq!(target, [-1.0, -2.0]);
        assert_eq!(fixed, [true, true]);
    }

    #[test]
    fn primal_utility_out_of_domain_is_neg_infinity() {
        let obj = EndowmentLinear::new(vec![1.0, 0.55], vec![1.0, 2.0]).unwrap();
        // y[0] = -2.0 violates y >= -h0 = -1.0
        assert!(obj.primal_utility(&[-2.0, 0.0]).is_infinite());
        assert!(obj.primal_utility(&[-2.0, 0.0]).is_sign_negative());
    }
}
