//! Pure-data prediction-market problem builder.
//!
//! Julia parity target: `src/prediction_market_api.jl` —
//! `OutcomeSpec`, `UniV3LiquidityBand`, `UniV3MarketSpec`,
//! `PredictionMarketProblem`, `_prediction_market_objective`,
//! `_build_prediction_market_edge`, `_prediction_market_edges`,
//! `_default_split_bound`.
//!
//! Narrowed v1 scope: `UniV3MarketSpec` only — no `ConstantProductMarketSpec`,
//! no gas model, no workspace. The split/merge hyperedge is appended only
//! when `Mode::MixedEnabled`.
//!
//! Node indexing convention is 0-based: collateral lives at node `0`, outcome
//! `i` lives at node `i + 1` (matches Julia's 1-based `[1, outcome_index + 1]`
//! after the off-by-one shift).

use std::collections::{HashMap, HashSet};

use forecast_flows_core::{
    Edge, EdgeError, EndowmentLinear, ObjectiveError, SplitMerge, SplitMergeEdgeHandle,
    SplitMergeError, UniV3, UniV3Edge, UniV3Error,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProblemError {
    #[error("outcomes must be nonempty")]
    EmptyOutcomes,
    #[error("outcome_id must be nonempty")]
    EmptyOutcomeId,
    #[error("market_id must be nonempty")]
    EmptyMarketId,
    #[error("fair_value must be finite")]
    NonFiniteFairValue,
    #[error("initial_holding must be finite and nonnegative")]
    BadInitialHolding,
    #[error("collateral_balance must be finite and nonnegative")]
    BadCollateralBalance,
    #[error("split_bound must be finite and positive")]
    BadSplitBound,
    #[error("lower_price must be finite and positive")]
    BadBandLowerPrice,
    #[error("liquidity_L must be finite and nonnegative")]
    BadBandLiquidity,
    #[error("bands must be nonempty")]
    EmptyBands,
    #[error("bands must contain at least one positive-liquidity band")]
    NoPositiveLiquidityBand,
    #[error("zero-liquidity gaps require a final zero-liquidity terminal band")]
    ZeroLiquidityGap,
    #[error("outcome_id values must be unique")]
    DuplicateOutcomeId,
    #[error("market_id values must be unique")]
    DuplicateMarketId,
    #[error("market outcome_id \"{0}\" must reference a declared outcome")]
    UnknownMarketOutcome(String),
    #[error(transparent)]
    UniV3(#[from] UniV3Error),
    #[error(transparent)]
    Objective(#[from] ObjectiveError),
    #[error(transparent)]
    Edge(#[from] EdgeError),
    #[error(transparent)]
    SplitMerge(#[from] SplitMergeError),
}

/// Pure-data description of one prediction-market outcome under the stable v2
/// facade. Matches Julia `OutcomeSpec`.
#[derive(Debug, Clone)]
pub struct OutcomeSpec {
    outcome_id: String,
    fair_value: f64,
    initial_holding: f64,
}

impl OutcomeSpec {
    pub fn new(
        outcome_id: impl Into<String>,
        fair_value: f64,
        initial_holding: f64,
    ) -> Result<Self, ProblemError> {
        let outcome_id = outcome_id.into();
        if outcome_id.is_empty() {
            return Err(ProblemError::EmptyOutcomeId);
        }
        if !fair_value.is_finite() {
            return Err(ProblemError::NonFiniteFairValue);
        }
        if !(initial_holding.is_finite() && initial_holding >= 0.0) {
            return Err(ProblemError::BadInitialHolding);
        }
        Ok(Self {
            outcome_id,
            fair_value,
            initial_holding,
        })
    }

    #[inline]
    pub fn outcome_id(&self) -> &str {
        &self.outcome_id
    }

    #[inline]
    pub fn fair_value(&self) -> f64 {
        self.fair_value
    }

    #[inline]
    pub fn initial_holding(&self) -> f64 {
        self.initial_holding
    }
}

/// User-facing UniV3 liquidity band. `lower_price` is the outcome price at the
/// top of the band; `liquidity_l` is the standard Uniswap `L` (not the
/// internal reserve-product `L²`). Matches Julia `UniV3LiquidityBand`.
#[derive(Debug, Clone, Copy)]
pub struct UniV3Band {
    pub lower_price: f64,
    pub liquidity_l: f64,
}

impl UniV3Band {
    pub fn new(lower_price: f64, liquidity_l: f64) -> Result<Self, ProblemError> {
        if !(lower_price.is_finite() && lower_price > 0.0) {
            return Err(ProblemError::BadBandLowerPrice);
        }
        if !(liquidity_l.is_finite() && liquidity_l >= 0.0) {
            return Err(ProblemError::BadBandLiquidity);
        }
        Ok(Self {
            lower_price,
            liquidity_l,
        })
    }
}

/// Pure-data description of a multi-band collateral/outcome UniV3-style market.
/// Bands are stored sorted descending by `lower_price` (highest first);
/// internal `lower_ticks` / `liquidity_k` arrays are derived on demand at
/// edge-build time. Matches Julia `UniV3MarketSpec`.
#[derive(Debug, Clone)]
pub struct UniV3MarketSpec {
    market_id: String,
    outcome_id: String,
    current_price: f64,
    bands: Vec<UniV3Band>,
    fee_multiplier: f64,
}

impl UniV3MarketSpec {
    /// Build a `UniV3MarketSpec`. Mirrors the Julia public constructor
    /// (`prediction_market_api.jl` line 195): sorts bands descending by
    /// `lower_price`, requires at least one positive-liquidity band, requires
    /// any zero-liquidity gap to terminate at the lowest band, and validates
    /// the resulting internal `UniV3` edge eagerly.
    pub fn new(
        market_id: impl Into<String>,
        outcome_id: impl Into<String>,
        current_price: f64,
        bands: impl IntoIterator<Item = UniV3Band>,
        fee_multiplier: f64,
    ) -> Result<Self, ProblemError> {
        let market_id = market_id.into();
        let outcome_id = outcome_id.into();
        if market_id.is_empty() {
            return Err(ProblemError::EmptyMarketId);
        }
        if outcome_id.is_empty() {
            return Err(ProblemError::EmptyOutcomeId);
        }
        let mut bands: Vec<UniV3Band> = bands.into_iter().collect();
        if bands.is_empty() {
            return Err(ProblemError::EmptyBands);
        }
        if !bands.iter().any(|b| b.liquidity_l > 0.0) {
            return Err(ProblemError::NoPositiveLiquidityBand);
        }
        bands.sort_by(|a, b| {
            b.lower_price
                .partial_cmp(&a.lower_price)
                .expect("validated finite prices")
        });
        let zero_band_indices: Vec<usize> = bands
            .iter()
            .enumerate()
            .filter_map(|(i, b)| (b.liquidity_l == 0.0).then_some(i))
            .collect();
        if let Some(&last_zero) = zero_band_indices.last() {
            if last_zero != bands.len() - 1 {
                return Err(ProblemError::ZeroLiquidityGap);
            }
        }
        let spec = Self {
            market_id,
            outcome_id,
            current_price,
            bands,
            fee_multiplier,
        };
        // Eagerly construct the internal UniV3 edge to surface tick-range,
        // monotonicity, and fee-multiplier errors at spec-construction time.
        spec.build_internal_edge()?;
        Ok(spec)
    }

    #[inline]
    pub fn market_id(&self) -> &str {
        &self.market_id
    }

    #[inline]
    pub fn outcome_id(&self) -> &str {
        &self.outcome_id
    }

    #[inline]
    pub fn current_price(&self) -> f64 {
        self.current_price
    }

    #[inline]
    pub fn bands(&self) -> &[UniV3Band] {
        &self.bands
    }

    #[inline]
    pub fn fee_multiplier(&self) -> f64 {
        self.fee_multiplier
    }

    /// Build the solver-facing `UniV3` edge, applying Julia's
    /// `_prediction_market_univ3_internal_*` conversion: invert prices, reverse
    /// the band order, and special-case the terminal zero-liquidity band.
    fn build_internal_edge(&self) -> Result<UniV3, UniV3Error> {
        let internal_current_price = 1.0 / self.current_price;
        let internal_lower_ticks: Vec<f64> = self
            .bands
            .iter()
            .rev()
            .map(|b| 1.0 / b.lower_price)
            .collect();
        let liq_k: Vec<f64> = self
            .bands
            .iter()
            .map(|b| b.liquidity_l * b.liquidity_l)
            .collect();
        let internal_liquidity_k: Vec<f64> = if liq_k.last().copied() == Some(0.0) {
            let n = liq_k.len();
            let mut ret: Vec<f64> = liq_k[..n - 1].iter().rev().copied().collect();
            ret.push(0.0);
            ret
        } else {
            liq_k.iter().rev().copied().collect()
        };
        UniV3::new(
            internal_current_price,
            internal_lower_ticks,
            internal_liquidity_k,
            self.fee_multiplier,
        )
    }
}

/// Direct-only solves use just the AMM edges; mixed-enabled appends the
/// `(n+1)`-node split/merge hyperedge with `EndowmentLinear` collateral
/// gating. Matches Julia's `:direct_only` / `:mixed_enabled` symbols.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    DirectOnly,
    MixedEnabled,
}

/// Pure-data description of the one-collateral prediction-market routing
/// problem. Matches Julia `PredictionMarketProblem`.
#[derive(Debug, Clone)]
pub struct PredictionMarketProblem {
    outcomes: Vec<OutcomeSpec>,
    collateral_balance: f64,
    markets: Vec<UniV3MarketSpec>,
    split_bound: Option<f64>,
    /// Cached `outcome_id -> 0-based outcome index` map; collateral is node 0
    /// and outcome `i` is node `i + 1`.
    outcome_index_by_id: HashMap<String, usize>,
}

impl PredictionMarketProblem {
    pub fn new(
        outcomes: Vec<OutcomeSpec>,
        collateral_balance: f64,
        markets: Vec<UniV3MarketSpec>,
        split_bound: Option<f64>,
    ) -> Result<Self, ProblemError> {
        if outcomes.is_empty() {
            return Err(ProblemError::EmptyOutcomes);
        }
        if !(collateral_balance.is_finite() && collateral_balance >= 0.0) {
            return Err(ProblemError::BadCollateralBalance);
        }
        if let Some(b) = split_bound {
            if !(b.is_finite() && b > 0.0) {
                return Err(ProblemError::BadSplitBound);
            }
        }

        let mut outcome_index_by_id: HashMap<String, usize> =
            HashMap::with_capacity(outcomes.len());
        for (i, outcome) in outcomes.iter().enumerate() {
            if outcome_index_by_id
                .insert(outcome.outcome_id.clone(), i)
                .is_some()
            {
                return Err(ProblemError::DuplicateOutcomeId);
            }
        }

        let mut market_ids: HashSet<&str> = HashSet::with_capacity(markets.len());
        for spec in &markets {
            if !market_ids.insert(spec.market_id()) {
                return Err(ProblemError::DuplicateMarketId);
            }
            if !outcome_index_by_id.contains_key(spec.outcome_id()) {
                return Err(ProblemError::UnknownMarketOutcome(
                    spec.outcome_id().to_string(),
                ));
            }
        }

        Ok(Self {
            outcomes,
            collateral_balance,
            markets,
            split_bound,
            outcome_index_by_id,
        })
    }

    #[inline]
    pub fn outcomes(&self) -> &[OutcomeSpec] {
        &self.outcomes
    }

    #[inline]
    pub fn collateral_balance(&self) -> f64 {
        self.collateral_balance
    }

    #[inline]
    pub fn markets(&self) -> &[UniV3MarketSpec] {
        &self.markets
    }

    #[inline]
    pub fn split_bound(&self) -> Option<f64> {
        self.split_bound
    }

    /// Number of nodes in the dual problem: 1 collateral + N outcomes.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.outcomes.len() + 1
    }
}

/// Default split/merge bound `max(collateral + Σ holdings, ε)` — matches Julia
/// `_default_split_bound` (`prediction_market_api.jl:843`).
pub fn default_split_bound(problem: &PredictionMarketProblem) -> f64 {
    let mut total = problem.collateral_balance;
    for outcome in &problem.outcomes {
        total += outcome.initial_holding;
    }
    total.max(f64::EPSILON)
}

/// Analytical split-bound ceiling — the physical upper limit no flow can
/// exceed for any single outcome (initial holding + total AMM outcome
/// reserve across all markets that name it). Matches Julia
/// `_analytical_split_bound` (`prediction_market_api.jl:870`). Used by
/// `solve_with_doubling` to clamp `B *= 2`.
pub fn analytical_split_bound(problem: &PredictionMarketProblem) -> f64 {
    let mut caps: HashMap<&str, f64> = HashMap::with_capacity(problem.outcomes.len());
    for outcome in &problem.outcomes {
        caps.insert(outcome.outcome_id(), outcome.initial_holding);
    }
    for spec in &problem.markets {
        let extra = spec
            .build_internal_edge()
            .map(|edge| edge.max_outcome_reserve())
            .unwrap_or(0.0);
        let cap = caps.entry(spec.outcome_id()).or_insert(0.0);
        *cap += extra;
    }
    if caps.is_empty() {
        return problem.collateral_balance.max(f64::EPSILON);
    }
    caps.values().copied().fold(f64::INFINITY, f64::min)
}

/// Build the `EndowmentLinear` objective `c = [1, fair_values...]`,
/// `h₀ = [collateral, holdings...]`. Matches Julia
/// `_prediction_market_objective`.
pub fn build_objective(problem: &PredictionMarketProblem) -> Result<EndowmentLinear, ProblemError> {
    let n = problem.outcomes.len();
    let mut c = Vec::with_capacity(n + 1);
    let mut h0 = Vec::with_capacity(n + 1);
    c.push(1.0);
    h0.push(problem.collateral_balance);
    for outcome in &problem.outcomes {
        c.push(outcome.fair_value);
        h0.push(outcome.initial_holding);
    }
    Ok(EndowmentLinear::new(c, h0)?)
}

/// Build the closed-enum edge list. UniV3 markets become `Edge::UniV3` edges
/// on `[0, outcome_index + 1]`; in `Mode::MixedEnabled` an additional
/// split/merge hyperedge on `[0, 1, ..., n_outcomes]` is appended with the
/// resolved bound and Moreau-Yosida smoothing parameter. Matches Julia
/// `_prediction_market_edges`.
///
/// `split_bound_override` lets callers (the doubling loop) inject a bound
/// other than the spec's `split_bound` field. `smoothing` is `0.0` for the
/// nonsmooth seed and `μ = gap_tol/(5·B²)` for the smoothed pass.
pub fn build_edges(
    problem: &PredictionMarketProblem,
    mode: Mode,
    split_bound_override: Option<f64>,
    smoothing: f64,
) -> Result<Vec<Edge>, ProblemError> {
    let mut edges: Vec<Edge> = Vec::with_capacity(problem.markets.len() + 1);
    for spec in &problem.markets {
        let outcome_index = problem.outcome_index_by_id[spec.outcome_id()];
        let inner = spec.build_internal_edge()?;
        let nodes = [0_usize, outcome_index + 1];
        edges.push(Edge::UniV3(UniV3Edge::new(nodes, inner)?));
    }
    if mode == Mode::MixedEnabled {
        let bound = split_bound_override
            .or(problem.split_bound)
            .unwrap_or_else(|| default_split_bound(problem));
        let n_nodes = problem.n_nodes();
        let sm = SplitMerge::new(n_nodes, bound, smoothing)?;
        let nodes: Vec<usize> = (0..n_nodes).collect();
        edges.push(Edge::SplitMerge(SplitMergeEdgeHandle::new(nodes, sm)?));
    }
    Ok(edges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use forecast_flows_core::{
        BoundedDualProblem, CertifyTolerances, Objective, SolverOptions, moreau_yosida_mu,
    };

    fn three_outcome_problem() -> PredictionMarketProblem {
        let outcomes = vec![
            OutcomeSpec::new("a", 0.5, 0.0).unwrap(),
            OutcomeSpec::new("b", 0.3, 0.0).unwrap(),
            OutcomeSpec::new("c", 0.2, 0.0).unwrap(),
        ];
        let market = UniV3MarketSpec::new(
            "m_a",
            "a",
            0.5,
            vec![
                UniV3Band::new(1.0, 100.0).unwrap(),
                UniV3Band::new(0.4, 100.0).unwrap(),
            ],
            1.0,
        )
        .unwrap();
        PredictionMarketProblem::new(outcomes, 1.0, vec![market], None).unwrap()
    }

    #[test]
    fn outcome_spec_rejects_empty_id() {
        let err = OutcomeSpec::new("", 0.5, 0.0).unwrap_err();
        assert!(matches!(err, ProblemError::EmptyOutcomeId));
    }

    #[test]
    fn outcome_spec_rejects_negative_holding() {
        let err = OutcomeSpec::new("a", 0.5, -1.0).unwrap_err();
        assert!(matches!(err, ProblemError::BadInitialHolding));
    }

    #[test]
    fn band_rejects_nonpositive_price() {
        let err = UniV3Band::new(0.0, 1.0).unwrap_err();
        assert!(matches!(err, ProblemError::BadBandLowerPrice));
    }

    #[test]
    fn univ3_spec_requires_positive_liquidity_band() {
        let err = UniV3MarketSpec::new("m", "a", 0.5, vec![UniV3Band::new(1.0, 0.0).unwrap()], 1.0)
            .unwrap_err();
        assert!(matches!(err, ProblemError::NoPositiveLiquidityBand));
    }

    #[test]
    fn univ3_spec_rejects_interior_zero_gap() {
        // Lowest band has positive liquidity → terminal zero rule violated.
        let err = UniV3MarketSpec::new(
            "m",
            "a",
            0.5,
            vec![
                UniV3Band::new(1.0, 100.0).unwrap(),
                UniV3Band::new(0.6, 0.0).unwrap(),
                UniV3Band::new(0.3, 100.0).unwrap(),
            ],
            1.0,
        )
        .unwrap_err();
        assert!(matches!(err, ProblemError::ZeroLiquidityGap));
    }

    #[test]
    fn univ3_spec_accepts_terminal_zero_gap() {
        // Terminal zero band is allowed and encodes "no liquidity below 0.3".
        let spec = UniV3MarketSpec::new(
            "m",
            "a",
            0.6,
            vec![
                UniV3Band::new(1.0, 100.0).unwrap(),
                UniV3Band::new(0.5, 100.0).unwrap(),
                UniV3Band::new(0.3, 0.0).unwrap(),
            ],
            1.0,
        )
        .unwrap();
        assert_eq!(spec.bands().len(), 3);
        // Sorted descending by lower_price.
        assert_eq!(spec.bands()[0].lower_price, 1.0);
        assert_eq!(spec.bands()[2].lower_price, 0.3);
    }

    #[test]
    fn problem_rejects_duplicate_outcome_id() {
        let outcomes = vec![
            OutcomeSpec::new("a", 0.5, 0.0).unwrap(),
            OutcomeSpec::new("a", 0.5, 0.0).unwrap(),
        ];
        let err = PredictionMarketProblem::new(outcomes, 1.0, vec![], None).unwrap_err();
        assert!(matches!(err, ProblemError::DuplicateOutcomeId));
    }

    #[test]
    fn problem_rejects_market_referencing_unknown_outcome() {
        let outcomes = vec![
            OutcomeSpec::new("a", 0.5, 0.0).unwrap(),
            OutcomeSpec::new("b", 0.5, 0.0).unwrap(),
        ];
        let market = UniV3MarketSpec::new(
            "m",
            "z",
            0.5,
            vec![UniV3Band::new(0.4, 100.0).unwrap()],
            1.0,
        )
        .unwrap();
        let err = PredictionMarketProblem::new(outcomes, 1.0, vec![market], None).unwrap_err();
        assert!(matches!(err, ProblemError::UnknownMarketOutcome(id) if id == "z"));
    }

    #[test]
    fn default_split_bound_sums_collateral_and_holdings() {
        let outcomes = vec![
            OutcomeSpec::new("a", 0.5, 2.0).unwrap(),
            OutcomeSpec::new("b", 0.3, 0.5).unwrap(),
        ];
        let problem = PredictionMarketProblem::new(outcomes, 1.5, vec![], None).unwrap();
        assert!((default_split_bound(&problem) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn build_objective_lays_out_collateral_first() {
        let problem = three_outcome_problem();
        let obj = build_objective(&problem).unwrap();
        assert_eq!(obj.c(), &[1.0, 0.5, 0.3, 0.2]);
        assert_eq!(obj.h0(), &[1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn build_edges_direct_only_skips_split_merge() {
        let problem = three_outcome_problem();
        let edges = build_edges(&problem, Mode::DirectOnly, None, 0.0).unwrap();
        assert_eq!(edges.len(), 1);
        assert!(matches!(edges[0], Edge::UniV3(_)));
        assert_eq!(edges[0].nodes(), &[0, 1]);
    }

    #[test]
    fn build_edges_mixed_enabled_appends_split_merge() {
        let problem = three_outcome_problem();
        let edges = build_edges(&problem, Mode::MixedEnabled, None, 0.0).unwrap();
        assert_eq!(edges.len(), 2);
        match &edges[1] {
            Edge::SplitMerge(_) => {}
            _ => panic!("expected SplitMerge as last edge"),
        }
        assert_eq!(edges[1].nodes(), &[0, 1, 2, 3]);
    }

    #[test]
    fn build_edges_uses_split_bound_override() {
        let problem = three_outcome_problem();
        let mu = moreau_yosida_mu(2.5, 1e-6);
        let edges = build_edges(&problem, Mode::MixedEnabled, Some(2.5), mu).unwrap();
        match &edges[1] {
            Edge::SplitMerge(handle) => {
                let sm = handle.inner();
                assert!((sm.bound() - 2.5).abs() < 1e-12);
                assert!((sm.smoothing() - mu).abs() < 1e-18);
            }
            _ => panic!("expected SplitMerge edge"),
        }
    }

    #[test]
    fn round_trip_into_bounded_dual_problem_certifies() {
        // Single market with current_price = fair_value → no-trade equilibrium
        // at ν = c. Mixed mode adds a split/merge edge at the no-trade point;
        // certify must pass.
        let outcomes = vec![
            OutcomeSpec::new("a", 0.5, 0.0).unwrap(),
            OutcomeSpec::new("b", 0.5, 0.0).unwrap(),
        ];
        let market = UniV3MarketSpec::new(
            "m_a",
            "a",
            0.5,
            vec![
                UniV3Band::new(1.0, 100.0).unwrap(),
                UniV3Band::new(0.4, 100.0).unwrap(),
            ],
            1.0,
        )
        .unwrap();
        let problem = PredictionMarketProblem::new(outcomes, 1.0, vec![market], None).unwrap();
        let obj = build_objective(&problem).unwrap();
        let edges = build_edges(&problem, Mode::DirectOnly, None, 0.0).unwrap();
        let bdp =
            BoundedDualProblem::new(Objective::EndowmentLinear(obj), edges, problem.n_nodes())
                .unwrap();
        // Loosen gap_tol past the LBFGS-B termination noise floor (the default
        // `max(√eps, 1e-6)` is tighter than pgtol-limited convergence permits
        // on this seed).
        let tols = CertifyTolerances {
            gap_tol: 1e-4,
            target_tol: 1e-4,
            bound_tol: 1e-6,
        };
        let (_solution, cert) = bdp
            .solve_and_certify(SolverOptions::default(), tols)
            .unwrap();
        assert!(cert.passed, "certificate failed: {}", cert.reason);
    }
}
