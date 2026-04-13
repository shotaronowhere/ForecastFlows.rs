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
use crate::split_merge::SplitMerge;
use lbfgsb::{LbfgsbParameter, LbfgsbProblem, LbfgsbState};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Mutex;
use thiserror::Error;

/// The `lbfgsb` 0.1 crate wraps Becker's C port, which is not thread-safe —
/// internal Fortran COMMON blocks make concurrent `state.minimize()` calls
/// race and produce `NeverCertified` failures only when `cargo test` runs
/// multiple solver tests in parallel. Serialize all minimize entries through
/// this global mutex until upstream is patched (or until we swap to a
/// re-entrant port). Holding it across `minimize()` is acceptable because the
/// solve is the dominant cost in every test that touches it.
static LBFGSB_LOCK: Mutex<()> = Mutex::new(());

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
    Driver(String),
}

/// Why the L-BFGS-B driver returned. The wrapped `lbfgsb` crate (a faithful
/// Rust wrapper around Stephen Becker's C port of the Nocedal/Morales
/// reference Fortran) terminates internally on convergence, abnormal line
/// search, or internal error, and surfaces that reason only through private
/// `task`/`csave` buffers. We surface the two states we can distinguish
/// externally:
///
/// - `Terminated`: the driver returned normally (`state.minimize()` returned
///   `Ok`). Caller should use [`SolveCertificate`] to decide whether the
///   final point is actually optimal — a line-search failure inside C still
///   lands here, same as true convergence.
/// - `MaxIter`: the caller's function-evaluation budget (`max_iter`) was
///   exhausted via the internal closure-abort path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStatus {
    Terminated,
    MaxIter,
}

/// Tunable solver options. Defaults mirror Julia `_solve_lbfgsb_once!`.
#[derive(Debug, Clone, Copy)]
pub struct SolverOptions {
    pub memory: usize,
    pub max_iter: usize,
    pub pgtol: f64,
    /// L-BFGS-B function-value stopping ratio `factr·epsmch`. Julia
    /// (`src/solver.jl:499`) defaults to `factr=1e1` — MUCH tighter than the
    /// `lbfgsb` crate's own `1e7` default. Leaving it at the crate default
    /// causes Rust to terminate at ~2e-9 function-value precision while
    /// Julia runs to ~2e-15, which on UniV3 problems with `γ = 1` leaves a
    /// ~1e-7 "near-kink" residual flow in Rust that Julia's tighter solve
    /// converges away.
    pub factr: f64,
    pub verbose: bool,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            memory: 5,
            max_iter: 10_000,
            pgtol: 1e-5,
            factr: 1e1,
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
    /// Number of function-and-gradient evaluations performed during the
    /// solve. L-BFGS-B evaluates the oracle multiple times per outer
    /// iteration (line search probes), so this is larger than Julia's
    /// `isave(30)` iteration count; kept as a diagnostic, not load-bearing.
    pub iterations: usize,
    pub status: SolveStatus,
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
        self.solve_with_seed(opts, None)
    }

    /// Solve the bounded dual with a caller-provided dual seed. `nu0 = None`
    /// recovers the cold-start behavior (Julia `_default_dual_seed_offset`).
    /// When `nu0 = Some(seed)`, the seed is clamped into the `[lb, ub]` box
    /// before being passed to L-BFGS-B, matching Julia's clamp in
    /// `_initialize_lbfgsb_state!`.
    #[allow(clippy::too_many_lines)]
    pub fn solve_with_seed(
        &self,
        opts: SolverOptions,
        nu0: Option<&[f64]>,
    ) -> Result<DualSolution, SolveError> {
        let n = self.n_nodes;
        let Objective::EndowmentLinear(obj) = &self.objective;
        let (lower, upper) = dual_bounds(obj);
        let nu = match nu0 {
            Some(seed) if seed.len() == n => clamp_seed(seed, &lower, &upper),
            _ => seed_nu(&lower, &upper, obj),
        };

        // `LbfgsbProblem::build` takes ownership of the closure, so all
        // captures must be owned (no borrows of `self`). Clone the edge
        // list + objective once into the closure — both types are `Clone`
        // and typically small (one `Edge::UniV3` per market plus an
        // optional trailing `Edge::SplitMerge`). Iteration count lives in
        // an `Rc<RefCell<…>>` so the outer scope can read it back after
        // `state.minimize` returns.
        let iter_count: Rc<RefCell<usize>> = Rc::new(RefCell::new(0usize));
        let iter_count_inner = Rc::clone(&iter_count);
        let edges_owned: Vec<Edge> = self.edges.clone();
        let obj_clone = obj.clone();
        let max_iter = opts.max_iter;
        let n_edges = self.edges.len();
        let mut edge_flows_inner: Vec<Vec<f64>> =
            self.edges.iter().map(|e| vec![0.0; e.degree()]).collect();

        let eval_fn = move |mu: &[f64], g_out: &mut [f64]| -> anyhow::Result<f64> {
            {
                let mut count = iter_count_inner.borrow_mut();
                if *count >= max_iter {
                    return Err(anyhow::anyhow!("max_iter {max_iter} exceeded"));
                }
                *count += 1;
            }

            for (edge, x) in edges_owned.iter().zip(edge_flows_inner.iter_mut()) {
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

            let mut f = obj_clone.dual_utility(mu);
            for (edge, x) in edges_owned.iter().zip(edge_flows_inner.iter()) {
                for (k, &node) in edge.nodes().iter().enumerate() {
                    f += x[k] * mu[node];
                }
            }

            obj_clone.dual_gradient_into(g_out, mu);
            for (edge, x) in edges_owned.iter().zip(edge_flows_inner.iter()) {
                for (k, &node) in edge.nodes().iter().enumerate() {
                    g_out[node] += x[k];
                }
            }

            Ok(f)
        };

        let mut problem = LbfgsbProblem::build(nu, eval_fn);
        let bounds_iter =
            lower
                .iter()
                .zip(upper.iter())
                .map(|(&l, &u)| -> (Option<f64>, Option<f64>) {
                    (Some(l), if u.is_finite() { Some(u) } else { None })
                });
        problem.set_bounds(bounds_iter);

        let param = LbfgsbParameter {
            m: opts.memory,
            factr: opts.factr,
            pgtol: opts.pgtol,
            iprint: if opts.verbose { 99 } else { -1 },
        };
        let mut state = LbfgsbState::new(problem, param);
        let min_result = {
            let _guard = LBFGSB_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.minimize()
        };
        let status = if min_result.is_err() {
            SolveStatus::MaxIter
        } else {
            SolveStatus::Terminated
        };

        let nu_out = state.x().to_vec();
        let f = state.fx();
        let iterations = *iter_count.borrow();

        // Drop the LbfgsbState (and its closure+captures) before recomputing
        // edge_flows at the final ν — the closure's edge_flows was consumed,
        // so we allocate a fresh one for the authoritative post-solve
        // snapshot.
        drop(state);
        let mut edge_flows: Vec<Vec<f64>> = Vec::with_capacity(n_edges);
        for edge in &self.edges {
            edge_flows.push(vec![0.0; edge.degree()]);
        }
        for (edge, x) in self.edges.iter().zip(edge_flows.iter_mut()) {
            let eta: Vec<f64> = edge.nodes().iter().map(|&i| nu_out[i]).collect();
            edge.find_arb(x, &eta);
        }

        cleanup_near_zero_flows(&self.edges, &mut edge_flows, &nu_out);

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

    /// Convenience wrapper that runs `solve` → `recover_primal` (no-op for
    /// direct-only problems) → `certify`. Mirrors the `_finalize_solution!` +
    /// `certify_solution` sequence at the tail of Julia's
    /// `_solve_lbfgsb_once!` (`src/solver.jl:585`).
    ///
    /// `recovery_tol` follows Julia's default of `max(√eps, 10·pgtol)`.
    pub fn solve_and_certify(
        &self,
        opts: SolverOptions,
        tols: CertifyTolerances,
    ) -> Result<(DualSolution, SolveCertificate), SolveError> {
        self.solve_and_certify_with_seed(opts, tols, None)
    }

    /// `solve_and_certify` variant that threads an optional dual seed through
    /// to `solve_with_seed` for workspace-cache warm starts.
    pub fn solve_and_certify_with_seed(
        &self,
        opts: SolverOptions,
        tols: CertifyTolerances,
        nu0: Option<&[f64]>,
    ) -> Result<(DualSolution, SolveCertificate), SolveError> {
        let mut solution = self.solve_with_seed(opts, nu0)?;
        let recovery_tol = f64::EPSILON.sqrt().max(10.0 * opts.pgtol);
        recover_primal(self, &mut solution, recovery_tol);
        let cert = certify(self, &solution, tols);
        Ok((solution, cert))
    }
}

/// Compute `[lower, upper]` dual bounds with the same `max(0, lb)` floor
/// Julia applies in `_lbfgsb_bounds` (line 433).
/// Julia parity: `cleanup_near_zero_flows!` (`src/solver.jl:102`). For every
/// edge whose flow norm is below `flow_tol` AND whose execution value
/// `⟨ν, x⟩` is below `atol`, zero the flow out. Julia's Fortran L-BFGS-B
/// converges ~20x tighter than the C port we link here, so without this
/// cleanup Rust leaves ~1e-7 "near kink" residuals on UniV3 edges sitting at
/// the no-trade cone (γ=1 collapses the cone to a single exact ratio that
/// f64 can't hit). Those residuals would otherwise leak into `extract_trades`
/// as spurious extra trades vs. the Julia fixture.
fn cleanup_near_zero_flows(edges: &[Edge], edge_flows: &mut [Vec<f64>], nu: &[f64]) {
    let atol = f64::EPSILON.sqrt();
    let flow_tol = atol.sqrt().max(10.0 * f64::EPSILON.sqrt());
    for (edge, x) in edges.iter().zip(edge_flows.iter_mut()) {
        let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > flow_tol {
            continue;
        }
        let exec: f64 = edge
            .nodes()
            .iter()
            .zip(x.iter())
            .map(|(&node, &xi)| nu[node] * xi)
            .sum();
        if exec.abs() <= atol {
            for slot in x.iter_mut() {
                *slot = 0.0;
            }
        }
    }
}

fn dual_bounds(obj: &EndowmentLinear) -> (Vec<f64>, Vec<f64>) {
    let lb = obj.lower_limit();
    let ub = obj.upper_limit();
    let lower: Vec<f64> = lb.iter().map(|&v| v.max(0.0)).collect();
    (lower, ub)
}

/// Clamp a caller-provided dual seed into `[lower, upper]` with the same
/// `upper.is_finite()` guard `seed_nu` uses. Mirrors the clamp step in
/// Julia's `_initialize_lbfgsb_state!` so a workspace-cached seed from a
/// prior solve stays inside the current bound box even if the new problem
/// tightens some bounds.
fn clamp_seed(seed: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
    seed.iter()
        .zip(lower)
        .zip(upper)
        .map(|((&s, &lb), &ub)| {
            let c = s.max(lb);
            if ub.is_finite() { c.min(ub) } else { c }
        })
        .collect()
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

// ---------------------------------------------------------------------------
// Moreau-Yosida smoothing helper
// ---------------------------------------------------------------------------

/// Moreau-Yosida smoothing parameter `μ = gap_tol / (5·B²)`, which bounds the
/// smoothing bias `μB²/2` by `gap_tol/10`. Matches Julia
/// `_smoothing_parameter` (`src/prediction_market_api.jl:889`).
///
/// `B` is floored at `1.0` before squaring to guard the small-bound regime,
/// matching `B_safe = max(B, one(T))` in Julia.
pub fn moreau_yosida_mu(split_bound: f64, gap_tol: f64) -> f64 {
    let b_safe = split_bound.max(1.0);
    gap_tol / (5.0 * b_safe * b_safe)
}

// ---------------------------------------------------------------------------
// Certification + primal recovery
// ---------------------------------------------------------------------------

/// Julia parity: `SolveCertificate` (see `src/solver.jl:323` / struct defined
/// in `src/types.jl`). Captures the pass/fail verdict and the three residuals
/// used by `certify_solution`.
#[derive(Debug, Clone)]
pub struct SolveCertificate {
    pub passed: bool,
    pub reason: String,
    pub primal_value: f64,
    pub dual_value: f64,
    pub duality_gap: f64,
    pub target_residual: f64,
    pub bound_residual: f64,
}

/// Tolerance triple for [`certify`]. Defaults mirror Julia's
/// `certify_solution` keyword defaults (`max(sqrt(eps), 1e-6)` / `1e-8`).
#[derive(Debug, Clone, Copy)]
pub struct CertifyTolerances {
    pub gap_tol: f64,
    pub target_tol: f64,
    pub bound_tol: f64,
}

impl Default for CertifyTolerances {
    fn default() -> Self {
        let se = f64::EPSILON.sqrt();
        Self {
            gap_tol: se.max(1e-6),
            target_tol: se.max(1e-6),
            bound_tol: se.max(1e-8),
        }
    }
}

impl CertifyTolerances {
    /// Julia parity: `_solve_certificate_tolerances(T, pgtol)` at
    /// `src/solver.jl:423`. The Julia orchestration derives the certify
    /// tolerances from the caller's `pgtol` at every entry point (both
    /// `_solve_prediction_market_once` and `_solve_prediction_market_mixed`),
    /// so tightening `pgtol` also tightens the certificate instead of leaving
    /// the certificate stuck at its coarse default.
    #[must_use]
    pub fn from_pgtol(pgtol: f64) -> Self {
        let se = f64::EPSILON.sqrt();
        Self {
            gap_tol: (500.0 * se).max(500.0 * pgtol),
            target_tol: (100.0 * se).max(500.0 * pgtol),
            bound_tol: (100.0 * se).max(100.0 * pgtol),
        }
    }
}

/// `max(0, lbᵢ - νᵢ, νᵢ - ubᵢ)` over all dual coords, matching Julia
/// `_dual_bound_residual` (`src/solver.jl:280`).
fn dual_bound_residual(obj: &EndowmentLinear, nu: &[f64]) -> f64 {
    let lb = obj.lower_limit();
    let ub = obj.upper_limit();
    let mut resid = 0.0_f64;
    for i in 0..nu.len() {
        let lbf = lb[i].max(0.0);
        resid = resid.max((lbf - nu[i]).max(0.0));
        if ub[i].is_finite() {
            resid = resid.max((nu[i] - ub[i]).max(0.0));
        }
    }
    resid
}

/// Max split-merge consistency + bound violation across all split edges.
/// Matches Julia `_splitmerge_bound_residual` (`src/solver.jl:304`).
pub fn splitmerge_bound_residual(edges: &[Edge], edge_flows: &[Vec<f64>]) -> f64 {
    let mut resid = 0.0_f64;
    for (edge, x) in edges.iter().zip(edge_flows) {
        let Edge::SplitMerge(sme) = edge else {
            continue;
        };
        let sm = *sme.inner();
        let n = x.len();
        let w = if n == 1 {
            -x[0]
        } else {
            let mut s = 0.0;
            for v in &x[1..] {
                s += *v;
            }
            s / (n - 1) as f64
        };
        resid = resid.max((w.abs() - sm.bound()).max(0.0));
        resid = resid.max((x[0] + w).abs());
        for j in 1..n {
            resid = resid.max((x[j] - w).abs());
        }
    }
    resid
}

/// Primal-recovery oracle for a single split/merge edge. Matches Julia
/// `recover_splitmerge_flow!` (`src/prediction_markets.jl:249`).
#[allow(clippy::too_many_arguments)]
fn recover_splitmerge_flow(
    sm: SplitMerge,
    x: &mut [f64],
    eta: &[f64],
    residual: &[f64],
    fixed: &[bool],
    current_flow: &[f64],
    lower_bounds: &[f64],
    tol: f64,
) -> f64 {
    let gap = sm.gap(eta);
    let gap_tol = tol * eta.len() as f64;
    if gap > gap_tol {
        sm.set_flow(x, sm.bound());
        return sm.bound();
    }
    if gap < -gap_tol {
        sm.set_flow(x, -sm.bound());
        return -sm.bound();
    }

    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..residual.len() {
        if !fixed[i] {
            continue;
        }
        let d = if i == 0 { -1.0 } else { 1.0 };
        num += d * residual[i];
        den += 1.0;
    }
    if den == 0.0 {
        num = -residual[0];
        for r in &residual[1..] {
            num += *r;
        }
        den = residual.len() as f64;
    }

    let b = sm.bound();
    let mut wlo = -b;
    let mut whi = b;
    if lower_bounds[0].is_finite() {
        whi = whi.min(current_flow[0] - lower_bounds[0]);
    }
    for i in 1..current_flow.len() {
        if lower_bounds[i].is_finite() {
            wlo = wlo.max(lower_bounds[i] - current_flow[i]);
        }
    }

    let w = if wlo > whi {
        // Contradictory bounds: prioritize the collateral budget, then
        // satisfy as many outcome lower bounds as possible. Matches Julia's
        // contradictory-bounds branch.
        (num / den).clamp(-b, whi)
    } else {
        (num / den).clamp(wlo, whi)
    };
    sm.set_flow(x, w);
    w
}

/// Julia parity: `recover_primal!` (`src/solver.jl:113`). Rewrites every
/// split-merge edge's flow so the final netflow hits the objective's
/// recovery targets on fixed coordinates. Returns `true` iff the problem
/// actually contains any split-merge edges.
pub fn recover_primal(problem: &BoundedDualProblem, solution: &mut DualSolution, tol: f64) -> bool {
    let has_split = problem
        .edges
        .iter()
        .any(|e| matches!(e, Edge::SplitMerge(_)));
    if !has_split {
        return false;
    }

    let Objective::EndowmentLinear(obj) = &problem.objective;
    let n = problem.n_nodes;
    let mut target = vec![0.0_f64; n];
    let mut fixed = vec![false; n];
    obj.recovery_targets(&mut target, &mut fixed, &solution.nu);
    let lower_bounds = obj.primal_lower_bounds();

    // Seed y with the netflow contribution from every non-split edge.
    let mut y = vec![0.0_f64; n];
    for (edge, x) in problem.edges.iter().zip(&solution.edge_flows) {
        if matches!(edge, Edge::SplitMerge(_)) {
            continue;
        }
        for (k, &node) in edge.nodes().iter().enumerate() {
            y[node] += x[k];
        }
    }

    for (i, edge) in problem.edges.iter().enumerate() {
        let Edge::SplitMerge(sme) = edge else {
            continue;
        };
        let sm = *sme.inner();
        let nodes = sme.nodes().to_vec();
        let deg = nodes.len();
        let mut eta = vec![0.0_f64; deg];
        let mut residual = vec![0.0_f64; deg];
        let mut local_fixed = vec![false; deg];
        let mut local_flow = vec![0.0_f64; deg];
        let mut local_lb = vec![0.0_f64; deg];
        for (k, &node) in nodes.iter().enumerate() {
            eta[k] = solution.nu[node];
            residual[k] = target[node] - y[node];
            local_fixed[k] = fixed[node];
            local_flow[k] = y[node];
            local_lb[k] = lower_bounds[node];
        }
        recover_splitmerge_flow(
            sm,
            &mut solution.edge_flows[i],
            &eta,
            &residual,
            &local_fixed,
            &local_flow,
            &local_lb,
            tol,
        );
        for (k, &node) in nodes.iter().enumerate() {
            y[node] += solution.edge_flows[i][k];
        }
    }

    solution.netflow = y;
    true
}

/// Julia parity: `certify_solution` (`src/solver.jl:323`). Recomputes
/// `ŷ = Σᵢ scatter(xᵢ, Aᵢ)` from the per-edge flows, measures the
/// consistency/target/bound residuals, and emits the verdict.
pub fn certify(
    problem: &BoundedDualProblem,
    solution: &DualSolution,
    tols: CertifyTolerances,
) -> SolveCertificate {
    let Objective::EndowmentLinear(obj) = &problem.objective;
    let n = problem.n_nodes;

    let mut yhat = vec![0.0_f64; n];
    for (edge, x) in problem.edges.iter().zip(&solution.edge_flows) {
        for (k, &node) in edge.nodes().iter().enumerate() {
            yhat[node] += x[k];
        }
    }

    let mut flow_residual = 0.0_f64;
    for i in 0..n {
        flow_residual = flow_residual.max((yhat[i] - solution.netflow[i]).abs());
    }

    let mut target = vec![0.0_f64; n];
    let mut fixed = vec![false; n];
    obj.recovery_targets(&mut target, &mut fixed, &solution.nu);

    let mut target_residual = flow_residual;
    for i in 0..n {
        if fixed[i] {
            target_residual = target_residual.max((yhat[i] - target[i]).abs());
        }
    }

    let bound_residual = dual_bound_residual(obj, &solution.nu).max(splitmerge_bound_residual(
        &problem.edges,
        &solution.edge_flows,
    ));

    let primal_val = obj.primal_utility(&yhat);
    let dual_val = {
        let mut acc = obj.dual_utility(&solution.nu);
        for (edge, x) in problem.edges.iter().zip(&solution.edge_flows) {
            for (k, &node) in edge.nodes().iter().enumerate() {
                acc += x[k] * solution.nu[node];
            }
        }
        acc
    };
    let duality_gap = dual_val - primal_val;

    let primal_finite = primal_val.is_finite();
    let dual_finite = dual_val.is_finite();
    let passed = primal_finite
        && dual_finite
        && target_residual <= tols.target_tol
        && bound_residual <= tols.bound_tol
        && duality_gap.abs() <= tols.gap_tol;

    let mut reasons: Vec<String> = Vec::new();
    if !primal_finite {
        reasons.push("primal objective is not finite".into());
    }
    if !dual_finite {
        reasons.push("dual objective is not finite".into());
    }
    if target_residual > tols.target_tol {
        reasons.push(format!(
            "target residual {target_residual} exceeds tolerance {}",
            tols.target_tol
        ));
    }
    if bound_residual > tols.bound_tol {
        reasons.push(format!(
            "bound residual {bound_residual} exceeds tolerance {}",
            tols.bound_tol
        ));
    }
    if duality_gap.abs() > tols.gap_tol {
        reasons.push(format!(
            "duality gap {duality_gap} exceeds tolerance {}",
            tols.gap_tol
        ));
    }
    if reasons.is_empty() {
        reasons.push("certified".into());
    }

    SolveCertificate {
        passed,
        reason: reasons.join("; "),
        primal_value: primal_val,
        dual_value: dual_val,
        duality_gap,
        target_residual,
        bound_residual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::{Edge, SplitMergeEdge, UniV3Edge};
    use crate::objective::{EndowmentLinear, Objective};
    use crate::split_merge::SplitMerge;
    use crate::uni_v3::UniV3;

    // -- lbfgsb driver smoke tests ----------------------------------------
    //
    // Thin sanity checks over the upstream `lbfgsb` crate so a regression
    // in the underlying C driver shows up here, not buried inside a
    // BoundedDualProblem integration failure. These mirror the earlier
    // `lbfgsb-rs-pure` smoke tests but use the `LbfgsbProblem`/`LbfgsbState`
    // API of Stephen Becker's C port.

    #[test]
    fn lbfgsb_minimizes_scalar_quadratic_inside_bounds() {
        let f_and_grad = |xv: &[f64], g: &mut [f64]| -> anyhow::Result<f64> {
            let d = xv[0] - 0.5;
            g[0] = 2.0 * d;
            Ok(d * d)
        };
        let mut problem = LbfgsbProblem::build(vec![0.0_f64], f_and_grad);
        problem.set_bounds(std::iter::once((Some(-1.0_f64), Some(1.0_f64))));
        let mut state = LbfgsbState::new(problem, LbfgsbParameter::default());
        state.minimize().expect("scalar quadratic minimize");
        let x = state.x();
        assert!((x[0] - 0.5).abs() < 1e-6, "expected x ≈ 0.5, got {}", x[0]);
        assert!(state.fx() < 1e-10, "expected f ≈ 0, got {}", state.fx());
    }

    #[test]
    fn lbfgsb_respects_active_upper_bound() {
        let f_and_grad = |xv: &[f64], g: &mut [f64]| -> anyhow::Result<f64> {
            let d = xv[0] - 2.0;
            g[0] = 2.0 * d;
            Ok(d * d)
        };
        let mut problem = LbfgsbProblem::build(vec![0.0_f64], f_and_grad);
        problem.set_bounds(std::iter::once((Some(-1.0_f64), Some(1.0_f64))));
        let mut state = LbfgsbState::new(problem, LbfgsbParameter::default());
        state.minimize().expect("bounded minimize");
        let x = state.x();
        assert!(
            (x[0] - 1.0).abs() < 1e-6,
            "expected x pinned to upper bound 1.0, got {}",
            x[0]
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
        assert_eq!(sol.status, SolveStatus::Terminated);
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

    // -- splitmerge_bound_residual / recover_primal / certify ------------

    fn splitmerge_problem(mu: f64) -> BoundedDualProblem {
        // 3 nodes: [collateral, outcome1, outcome2]. h0 = 0 so there is no
        // endowment floor, keeping primal recovery degenerate but exercising
        // the recovery-target branch on ν = c.
        let obj = EndowmentLinear::new(vec![1.0, 0.5, 0.5], vec![0.0, 0.0, 0.0]).unwrap();
        let sm = SplitMerge::new(3, 2.0, mu).unwrap();
        let edge = Edge::SplitMerge(SplitMergeEdge::new(vec![0, 1, 2], sm).unwrap());
        BoundedDualProblem::new(Objective::EndowmentLinear(obj), vec![edge], 3).unwrap()
    }

    #[test]
    fn splitmerge_bound_residual_flags_oversized_flow() {
        let problem = splitmerge_problem(0.0);
        // w = 3 > B = 2 so |w| - B = 1.
        let edge_flows = vec![vec![-3.0, 3.0, 3.0]];
        let r = splitmerge_bound_residual(&problem.edges, &edge_flows);
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn splitmerge_bound_residual_zero_for_consistent_flow() {
        let problem = splitmerge_problem(0.0);
        let edge_flows = vec![vec![-1.5, 1.5, 1.5]];
        let r = splitmerge_bound_residual(&problem.edges, &edge_flows);
        assert!(r.abs() < 1e-15);
    }

    #[test]
    fn recover_primal_is_noop_without_split_edges() {
        let problem = trivial_univ3_problem();
        let mut sol = problem.solve(SolverOptions::default()).unwrap();
        let before = sol.edge_flows.clone();
        let recovered = recover_primal(&problem, &mut sol, 1e-8);
        assert!(!recovered);
        assert_eq!(before, sol.edge_flows);
    }

    #[test]
    fn recover_primal_snaps_to_bang_bang_mint_on_large_positive_gap() {
        // Hand-built solution: ν clearly above c on the outcome sides so the
        // split-merge gap is strongly positive, forcing the mint branch.
        let problem = splitmerge_problem(0.0);
        let mut sol = DualSolution {
            nu: vec![1.0, 1.0, 1.0],
            netflow: vec![0.0; 3],
            edge_flows: vec![vec![0.0; 3]],
            dual_value: 0.0,
            iterations: 0,
            status: SolveStatus::Terminated,
        };
        let recovered = recover_primal(&problem, &mut sol, f64::EPSILON.sqrt());
        assert!(recovered);
        // gap = 1 + 1 - 1 = 1 > gap_tol → bang-bang mint at w = B = 2.
        assert!((sol.edge_flows[0][0] + 2.0).abs() < 1e-12);
        assert!((sol.edge_flows[0][1] - 2.0).abs() < 1e-12);
        assert!((sol.edge_flows[0][2] - 2.0).abs() < 1e-12);
        assert!((sol.netflow[0] + 2.0).abs() < 1e-12);
        assert!((sol.netflow[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn moreau_yosida_mu_matches_julia_formula() {
        let gap_tol = 5e-4;
        let b = 4.0;
        // μ = gap_tol / (5 · max(B,1)²) = 5e-4 / 80 = 6.25e-6
        let mu = moreau_yosida_mu(b, gap_tol);
        assert!((mu - 6.25e-6).abs() < 1e-18);
        // B < 1 gets floored.
        let mu_tiny = moreau_yosida_mu(0.1, gap_tol);
        assert!((mu_tiny - gap_tol / 5.0).abs() < 1e-18);
    }

    #[test]
    fn solve_and_certify_direct_only_two_outcome_problem() {
        // 3 nodes: [collateral=0, outcome1=1, outcome2=2]. Two UniV3 edges,
        // one per outcome, both γ = 1 at current_price = 1 so the no-trade
        // cone `{1}` contains the ratio `ν_i/ν_j = 1` at the ν = c seed.
        // Objective: c = [1, 1, 1], h₀ = [1, 1, 1] — identical to the
        // single-UniV3 smoke test except with an extra outcome node.
        let obj = EndowmentLinear::new(vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]).unwrap();
        let pool1 = UniV3::new(1.0, vec![2.0], vec![100.0], 1.0).unwrap();
        let pool2 = UniV3::new(1.0, vec![2.0], vec![100.0], 1.0).unwrap();
        let edges = vec![
            Edge::UniV3(UniV3Edge::new([0, 1], pool1).unwrap()),
            Edge::UniV3(UniV3Edge::new([0, 2], pool2).unwrap()),
        ];
        let problem = BoundedDualProblem::new(Objective::EndowmentLinear(obj), edges, 3).unwrap();
        let loose = CertifyTolerances {
            gap_tol: 1e-3,
            target_tol: 1e-4,
            bound_tol: 1e-4,
        };
        let (sol, cert) = problem
            .solve_and_certify(SolverOptions::default(), loose)
            .unwrap();
        assert_eq!(sol.status, SolveStatus::Terminated);
        assert_eq!(sol.nu.len(), 3);
        assert!(cert.passed, "cert should pass: {}", cert.reason);
        // No split-merge edges → recover_primal is a no-op; netflow stays
        // as the scatter of the per-edge optima.
        assert_eq!(sol.edge_flows.len(), 2);
    }

    #[test]
    fn certify_trivial_problem_passes() {
        let problem = trivial_univ3_problem();
        let sol = problem.solve(SolverOptions::default()).unwrap();
        let cert = certify(&problem, &sol, CertifyTolerances::default());
        // With edge idle, flows are zero → all residuals zero, duality gap
        // equals Ūᶜ(ν) - 0 ≈ 3e-6 which exceeds the default gap_tol of 1e-6.
        // Loosen tolerances to certify.
        let loose = CertifyTolerances {
            gap_tol: 1e-4,
            target_tol: 1e-6,
            bound_tol: 1e-8,
        };
        let cert_loose = certify(&problem, &sol, loose);
        assert!(
            cert_loose.passed,
            "loose cert failed: {}",
            cert_loose.reason
        );
        // Default-tolerance cert should fail on the gap and say so.
        assert!(!cert.passed);
        assert!(cert.reason.contains("duality gap"));
    }
}
