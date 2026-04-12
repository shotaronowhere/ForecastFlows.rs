//! Bounded L-BFGS-B dual solver adapter (Phase 5 scaffold).
//!
//! Julia parity target: `src/solver.jl` — specifically `_solve_lbfgsb_once!`,
//! `dual_objective`, `find_arb!`, `_lbfgsb_bounds`, `_initialize_lbfgsb_state!`,
//! and the doubling + smoothing schedule in the mixed-solve loop.
//!
//! **Current state:** this module only wires `lbfgsb-rs-pure` into the crate
//! and provides a dependency smoke test. The real `BoundedDualProblem`
//! adapter — which combines `Objective::dual_utility` with the edge `find_arb`
//! oracles into a single `(f, ∇f)` callback, applies the lower/upper dual
//! bounds from the objective, runs the doubling loop on `SplitMerge.b`, and
//! emits a certified primal recovery — lands in the next Phase 5 session
//! after a careful line-by-line cross-reference against `src/solver.jl`. The
//! Julia reference touches ~1000 LOC and is sign-sensitive (`dual_objective`
//! returns `U*(ν) − Σ fᵢ(ν|edge)` which is then *maximized*, i.e. the solver
//! minimizes its negation), so we stage the port deliberately rather than
//! rushing.
//!
//! The sanity test below proves:
//! - the `lbfgsb-rs-pure` crate builds under our workspace lints,
//! - the `LBFGSB::new(m).minimize(...)` entry point is reachable, and
//! - a trivially-bounded scalar quadratic converges to the expected optimum.
//!
//! This keeps the next session focused on the adapter logic, not on build
//! plumbing or dependency discovery.

#[cfg(test)]
mod tests {
    use lbfgsb_rs_pure::{LBFGSB, Status};

    #[test]
    fn lbfgsb_minimizes_scalar_quadratic_inside_bounds() {
        // min (x - 0.5)^2 over x ∈ [-1, 1]. Optimum: x* = 0.5, f* = 0.
        let mut x = vec![0.0_f64];
        let lower = vec![-1.0_f64];
        let upper = vec![1.0_f64];
        let mut solver = LBFGSB::new(5);
        let mut f_and_grad = |xv: &[f64]| {
            let d = xv[0] - 0.5;
            (d * d, vec![2.0 * d])
        };
        let sol = solver
            .minimize(&mut x, &lower, &upper, &mut f_and_grad)
            .expect("lbfgsb sanity minimize should succeed");
        assert_eq!(sol.status, Status::Converged);
        assert!(
            (sol.x[0] - 0.5).abs() < 1e-6,
            "expected x ≈ 0.5, got {}",
            sol.x[0]
        );
        assert!(sol.f < 1e-10, "expected f ≈ 0, got {}", sol.f);
    }

    #[test]
    fn lbfgsb_respects_active_upper_bound() {
        // min (x - 2.0)^2 over x ∈ [-1, 1]. Optimum is on upper bound x* = 1.
        let mut x = vec![0.0_f64];
        let lower = vec![-1.0_f64];
        let upper = vec![1.0_f64];
        let mut solver = LBFGSB::new(5);
        let mut f_and_grad = |xv: &[f64]| {
            let d = xv[0] - 2.0;
            (d * d, vec![2.0 * d])
        };
        let sol = solver
            .minimize(&mut x, &lower, &upper, &mut f_and_grad)
            .expect("lbfgsb bounded minimize should succeed");
        assert_eq!(sol.status, Status::Converged);
        assert!(
            (sol.x[0] - 1.0).abs() < 1e-6,
            "expected x pinned to upper bound 1.0, got {}",
            sol.x[0]
        );
    }
}
