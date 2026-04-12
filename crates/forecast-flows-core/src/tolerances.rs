//! Centralized numerical tolerances.
//!
//! Mirrors the `sqrt(eps(T))`-style constants repeated throughout the Julia
//! implementation. Every Rust module that needs a convergence or snapping
//! tolerance should route through this module rather than repeat the literal.

#[inline]
#[must_use]
pub fn sqrt_eps_f64() -> f64 {
    f64::EPSILON.sqrt()
}
