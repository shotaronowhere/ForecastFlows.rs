//! Bounded L-BFGS-B dual solver adapter.
//!
//! Julia parity target: `src/solver.jl` — specifically `_solve_lbfgsb_once!`
//! (line 493), `dual_objective` (248), `find_arb!` (223), `_lbfgsb_bounds`
//! (430), and `_initialize_lbfgsb_state!` (462). v1 covers only the
//! `Vis_zero = true` branch (no per-edge objectives), which is the regime
//! used by `compare_prediction_market_families`.
//!
//! Sign convention (matches Julia exactly): Julia's `dual_objective` is the
//! value **minimized** by L-BFGS-B, not the concave dual that gets
//! maximized. Concretely:
//!
//! ```text
//!     g(ν) = Ūᶜ(ν) + Σᵢ ⟨xᵢ*(ν), ν[Aᵢ]⟩
//!   ∇g(ν) = ∇Ūᶜ(ν) + Σᵢ scatter(xᵢ*(ν), Aᵢ)
//! ```
//!
//! where `xᵢ*(ν) = argmaxₓ ⟨ν[Aᵢ], x⟩` is the per-edge closed-form oracle
//! (`Edge::find_arb`). The scatter gradient is exact by the envelope
//! theorem — the explicit `x`-dependence vanishes at the argmax.
//!
//! Bounds come from `EndowmentLinear::{lower_limit, upper_limit}` with the
//! same `max(0, lower_limit)` floor that Julia's `_lbfgsb_bounds` applies,
//! and the initial ν is `clamp(lb + 100·√eps, lb, ub)` mirroring
//! `_default_dual_seed_offset(::EndowmentLinear)` at
//! [`src/solver.jl:67`](../../../ForecastFlows.jl/src/solver.jl).
//!
//! **Not in this session.** No SplitMerge `B` doubling loop, no
//! Moreau-Yosida smoothing schedule, no certification, no primal recovery
//! — those land on top of this adapter in follow-up sessions.

use crate::edge::{Edge, EdgeError};
use crate::objective::{EndowmentLinear, Objective};
use lbfgsb_rs_pure::{LBFGSB, Solution, Status};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SolveError {
    #[error("objective dimension {obj} does not match n_nodes = {n}")]
    ObjectiveDimensionMismatch { obj: usize, n: usize },
    #[error("edge {idx}: {source}")]
    Edge {
        idx: usize,
        #[source]
        source: EdgeError,
    },
    #[error("L-BFGS-B driver failed: {0}")]
    Driver(&'static str),
}

/// Tunable solver options. Defaults mirror Julia `_solve_lbfgsb_once!`.
#[derive(Debug, Clone, Copy)]
pub struct SolverOptions {
    pub memory: usize,
    pub max_iter: usize,
    pub pgtol: f64,
    pub verbose: bool,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            memory: 5,
            max_iter: 10_000,
            pgtol: 1e-5,
            verbose: false,
        }
    }
}

/// Bounded-dual problem over a single `Objective` and a set of closed-enum
/// `Edge`s. The problem is the `Vis_zero = true` reduction at
/// [`src/solver.jl:248`](../../../ForecastFlows.jl/src/solver.jl).
#[derive(Debug, Clone)]
pub struct BoundedDualProblem {
    objective: Objective,
    edges: Vec<Edge>,
    n_nodes: usize,
}

/// Result of a single `BoundedDualProblem::solve` call.
#[derive(Debug, Clone)]
pub struct DualSolution {
    pub nu: Vec<f64>,
    pub netflow: Vec<f64>,
    pub edge_flows: Vec<Vec<f64>>,
    pub dual_value: f64,
    pub iterations: usize,
    pub status: Status,
}

impl BoundedDualProblem {
    pub fn new(objective: Objective, edges: Vec<Edge>, n_nodes: usize) -> Result<Self, SolveError> {
        let Objective::EndowmentLinear(obj_ref) = &objective;
        let obj_len = obj_ref.len();
        if obj_len != n_nodes {
            return Err(SolveError::ObjectiveDimensionMismatch {
                obj: obj_len,
                n: n_nodes,
            });
        }
        for (idx, edge) in edges.iter().enumerate() {
            edge.validate(n_nodes)
                .map_err(|source| SolveError::Edge { idx, source })?;
        }
        Ok(Self {
            objective,
            edges,
            n_nodes,
        })
    }

    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    #[inline]
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Solve the bounded dual with the provided options.
    pub fn solve(&self, opts: SolverOptions) -> Result<DualSolution, SolveError> {
        let n = self.n_nodes;
        let Objective::EndowmentLinear(obj) = &self.objective;
        let (lower, upper) = dual_bounds(obj);
        let mut nu = seed_nu(&lower, &upper, obj);

        // Persistent edge-flow buffers + scratch gradient storage, reused
        // across every callback evaluation to match Julia's in-place
        // `find_arb!` hot path.
        let mut edge_flows: Vec<Vec<f64>> =
            self.edges.iter().map(|e| vec![0.0; e.degree()]).collect();
        let mut obj_grad = vec![0.0_f64; n];

        let edges = &self.edges;
        let mut f_and_grad = |mu: &[f64]| -> (f64, Vec<f64>) {
            // Gather per-edge η, run the closed-form oracle, write xᵢ*.
            for (edge, x) in edges.iter().zip(edge_flows.iter_mut()) {
                let mut eta_buf = [0.0_f64; 16];
                let deg = edge.degree();
                if deg <= eta_buf.len() {
                    for (k, &node) in edge.nodes().iter().enumerate() {
                        eta_buf[k] = mu[node];
                    }
                    edge.find_arb(x, &eta_buf[..deg]);
                } else {
                    let eta: Vec<f64> = edge.nodes().iter().map(|&i| mu[i]).collect();
                    edge.find_arb(x, &eta);
                }
            }

            // f = Ūᶜ(ν) + Σᵢ ⟨xᵢ, ν[Aᵢ]⟩
            let mut f = obj.dual_utility(mu);
            for (edge, x) in edges.iter().zip(edge_flows.iter()) {
                for (k, &node) in edge.nodes().iter().enumerate() {
                    f += x[k] * mu[node];
                }
            }

            // ∇f = h₀ + Σᵢ scatter(xᵢ, Aᵢ). Allocate a fresh Vec because
            // the lbfgsb-rs-pure callback signature takes the gradient by
            // value; we still reuse `obj_grad` as a scratch buffer for
            // the objective gradient before copying into the return vec.
            obj.dual_gradient_into(&mut obj_grad, mu);
            let mut g = obj_grad.clone();
            for (edge, x) in edges.iter().zip(edge_flows.iter()) {
                for (k, &node) in edge.nodes().iter().enumerate() {
                    g[node] += x[k];
                }
            }

            (f, g)
        };

        let mut driver = LBFGSB::new(opts.memory)
            .with_max_iter(opts.max_iter)
            .with_pgtol(opts.pgtol)
            .with_verbose(opts.verbose);
        let Solution {
            x: nu_out,
            f,
            iterations,
            status,
        } = driver
            .minimize(&mut nu, &lower, &upper, &mut f_and_grad)
            .map_err(SolveError::Driver)?;

        // Recompute xᵢ at the final ν so `edge_flows` is authoritative on
        // exit, then assemble the netflow y = Σᵢ scatter(xᵢ, Aᵢ).
        for (edge, x) in self.edges.iter().zip(edge_flows.iter_mut()) {
            let eta: Vec<f64> = edge.nodes().iter().map(|&i| nu_out[i]).collect();
            edge.find_arb(x, &eta);
        }
        let mut netflow = vec![0.0_f64; n];
        for (edge, x) in self.edges.iter().zip(edge_flows.iter()) {
            for (k, &node) in edge.nodes().iter().enumerate() {
                netflow[node] += x[k];
            }
        }

        Ok(DualSolution {
            nu: nu_out,
            netflow,
            edge_flows,
            dual_value: f,
            iterations,
            status,
        })
    }
}

/// Compute `[lower, upper]` dual bounds with the same `max(0, lb)` floor
/// Julia applies in `_lbfgsb_bounds` (line 433).
fn dual_bounds(obj: &EndowmentLinear) -> (Vec<f64>, Vec<f64>) {
    let lb = obj.lower_limit();
    let ub = obj.upper_limit();
    let lower: Vec<f64> = lb.iter().map(|&v| v.max(0.0)).collect();
    (lower, ub)
}

/// Build the initial ν by applying Julia's `_default_dual_seed_offset`
/// (100·√eps for `EndowmentLinear`) and clamping into the bound box.
fn seed_nu(lower: &[f64], upper: &[f64], _obj: &EndowmentLinear) -> Vec<f64> {
    let offset = 100.0 * f64::EPSILON.sqrt();
    lower
        .iter()
        .zip(upper)
        .map(|(&lb, &ub)| {
            let seed = (lb + offset).max(lb);
            if ub.is_finite() { seed.min(ub) } else { seed }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::{Edge, UniV3Edge};
    use crate::objective::{EndowmentLinear, Objective};
    use crate::uni_v3::UniV3;
    use lbfgsb_rs_pure::{LBFGSB, Status};

    // -- lbfgsb-rs-pure dependency smoke tests ----------------------------

    #[test]
    fn lbfgsb_minimizes_scalar_quadratic_inside_bounds() {
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

    // -- BoundedDualProblem smoke tests -----------------------------------

    fn trivial_univ3_problem() -> BoundedDualProblem {
        let obj = EndowmentLinear::new(vec![1.0, 1.0], vec![1.0, 1.0]).unwrap();
        let pool = UniV3::new(1.0, vec![2.0], vec![100.0], 1.0).unwrap();
        let edge = Edge::UniV3(UniV3Edge::new([0, 1], pool).unwrap());
        BoundedDualProblem::new(Objective::EndowmentLinear(obj), vec![edge], 2).unwrap()
    }

    #[test]
    fn bounded_dual_rejects_dimension_mismatch() {
        let obj = EndowmentLinear::new(vec![1.0, 1.0], vec![0.0, 0.0]).unwrap();
        let err = BoundedDualProblem::new(Objective::EndowmentLinear(obj), vec![], 3).unwrap_err();
        assert!(matches!(
            err,
            SolveError::ObjectiveDimensionMismatch { obj: 2, n: 3 }
        ));
    }

    #[test]
    fn bounded_dual_rejects_out_of_range_edge_nodes() {
        let obj = EndowmentLinear::new(vec![1.0, 1.0], vec![0.0, 0.0]).unwrap();
        let pool = UniV3::new(1.0, vec![2.0], vec![100.0], 1.0).unwrap();
        let edge = Edge::UniV3(UniV3Edge::new([0, 5], pool).unwrap());
        let err =
            BoundedDualProblem::new(Objective::EndowmentLinear(obj), vec![edge], 2).unwrap_err();
        assert!(matches!(err, SolveError::Edge { idx: 0, .. }));
    }

    /// At γ = 1 with `current_price = 1` and `ν = c = [1, 1]`, the no-trade
    /// cone collapses to `{1}` and contains the market price
    /// `p = ν₀/ν₁ = 1`, so the edge is idle. With `h₀ = [1, 1]` the dual
    /// gradient at ν = c is `∇Ūᶜ = h₀ = [1, 1]`, which is fully projected
    /// away by the active lower bounds — the problem is already at its
    /// optimum from the seed point.
    #[test]
    fn bounded_dual_trivially_converges_on_no_trade_seed() {
        let problem = trivial_univ3_problem();
        let sol = problem.solve(SolverOptions::default()).unwrap();
        assert_eq!(sol.status, Status::Converged);
        assert_eq!(sol.nu.len(), 2);
        // ν stays near the seed (c + 100·√eps, clamped).
        assert!((sol.nu[0] - 1.0).abs() < 1e-4);
        assert!((sol.nu[1] - 1.0).abs() < 1e-4);
        // Edge idle → netflow ≈ 0.
        assert!(sol.netflow[0].abs() < 1e-10);
        assert!(sol.netflow[1].abs() < 1e-10);
        assert_eq!(sol.edge_flows.len(), 1);
        assert!(sol.edge_flows[0][0].abs() < 1e-10);
        assert!(sol.edge_flows[0][1].abs() < 1e-10);
        // Ūᶜ(ν ≈ c) = (ν − c)·h₀ ≈ 100·√eps · 2 ≈ 3e-6 (seed offset, not
        // a convergence error — the trade contribution is exactly zero).
        assert!(sol.dual_value.abs() < 1e-5);
    }
}
