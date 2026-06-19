use nalgebra::DVector;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TradingError {
    #[error("expected {expected} instruments, got {actual}")]
    InstrumentCount { expected: usize, actual: usize },
    #[error("expected horizon of at least {expected}, got {actual}")]
    HorizonLength { expected: usize, actual: usize },
    #[error("expected vector length {expected}, got {actual}")]
    VectorLength { expected: usize, actual: usize },
    #[error("configuration value `{name}` must be finite and non-negative")]
    NonNegativeConfig { name: &'static str },
    #[error("cvar beta must be in [0, 1), got {0}")]
    InvalidCvarBeta(f64),
}

pub type Result<T> = std::result::Result<T, TradingError>;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub vanna: f64,
    pub volga: f64,
}

impl Greeks {
    pub fn add_scaled(&mut self, other: Greeks, scale: f64) {
        self.delta += other.delta * scale;
        self.gamma += other.gamma * scale;
        self.vega += other.vega * scale;
        self.theta += other.theta * scale;
        self.vanna += other.vanna * scale;
        self.volga += other.volga * scale;
    }

    pub fn weighted_square_error(self, target: Greeks, weights: Greeks) -> f64 {
        weights.delta * (self.delta - target.delta).powi(2)
            + weights.gamma * (self.gamma - target.gamma).powi(2)
            + weights.vega * (self.vega - target.vega).powi(2)
            + weights.theta * (self.theta - target.theta).powi(2)
            + weights.vanna * (self.vanna - target.vanna).powi(2)
            + weights.volga * (self.volga - target.volga).powi(2)
    }
}

#[derive(Clone, Debug)]
pub struct PortfolioState {
    pub positions: DVector<f64>,
    pub greeks: Greeks,
    pub wealth: f64,
    pub peak_wealth: f64,
}

impl PortfolioState {
    pub fn new(positions: DVector<f64>, greeks: Greeks, wealth: f64, peak_wealth: f64) -> Self {
        Self {
            positions,
            greeks,
            wealth,
            peak_wealth: peak_wealth.max(wealth),
        }
    }

    pub fn drawdown_ratio(&self) -> f64 {
        if self.peak_wealth <= 0.0 {
            0.0
        } else {
            ((self.peak_wealth - self.wealth) / self.peak_wealth).max(0.0)
        }
    }
}

#[derive(Clone, Debug)]
pub struct InstrumentSnapshot {
    pub bid: f64,
    pub ask: f64,
    pub fair_value: f64,
    pub greeks: Greeks,
    pub transaction_cost_per_unit: f64,
    pub margin_per_unit: f64,
    pub max_buy: f64,
    pub max_sell: f64,
    pub lot_size: f64,
}

impl InstrumentSnapshot {
    pub fn mid(&self) -> f64 {
        0.5 * (self.bid + self.ask)
    }

    pub fn trade_edge(&self, quantity: f64) -> f64 {
        if quantity >= 0.0 {
            (self.fair_value - self.ask) * quantity
        } else {
            (self.bid - self.fair_value) * quantity.abs()
        }
    }

    pub fn transaction_cost(&self, quantity: f64) -> f64 {
        self.transaction_cost_per_unit * quantity.abs()
    }
}

#[derive(Clone, Debug)]
pub struct StageForecast {
    pub instruments: Vec<InstrumentSnapshot>,
    pub target_greeks: Greeks,
    pub loss_scenarios: Vec<f64>,
    pub gp_covariance_trace: f64,
}

#[derive(Clone, Debug)]
pub struct RiskLimits {
    pub max_abs_position: DVector<f64>,
    pub max_abs_trade: DVector<f64>,
    pub max_margin_ratio: f64,
    pub max_drawdown_ratio: f64,
    pub max_leverage: f64,
}

#[derive(Clone, Debug)]
pub struct ConstraintTightening {
    pub position_buffer: f64,
    pub margin_buffer: f64,
    pub drawdown_buffer: f64,
    pub leverage_buffer: f64,
}

impl Default for ConstraintTightening {
    fn default() -> Self {
        Self {
            position_buffer: 0.0,
            margin_buffer: 0.0,
            drawdown_buffer: 0.0,
            leverage_buffer: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ObjectiveWeights {
    pub greek: Greeks,
    pub cvar: f64,
    pub drawdown: f64,
    pub margin: f64,
    pub model_uncertainty: f64,
    pub trade_size: f64,
}

impl Default for ObjectiveWeights {
    fn default() -> Self {
        Self {
            greek: Greeks {
                delta: 10.0,
                gamma: 1.0,
                vega: 1.0,
                theta: 0.0,
                vanna: 0.25,
                volga: 0.25,
            },
            cvar: 1.0,
            drawdown: 10.0,
            margin: 5.0,
            model_uncertainty: 1.0,
            trade_size: 1e-3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TradingMpcConfig {
    pub horizon: usize,
    pub discount: f64,
    pub cvar_beta: f64,
    pub coordinate_passes: usize,
    pub step_lots: f64,
    pub min_improvement: f64,
    pub weights: ObjectiveWeights,
    pub tightening: ConstraintTightening,
}

impl Default for TradingMpcConfig {
    fn default() -> Self {
        Self {
            horizon: 3,
            discount: 0.98,
            cvar_beta: 0.95,
            coordinate_passes: 4,
            step_lots: 1.0,
            min_improvement: 1e-9,
            weights: ObjectiveWeights::default(),
            tightening: ConstraintTightening::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct StageBreakdown {
    pub edge: f64,
    pub transaction_cost: f64,
    pub greek_penalty: f64,
    pub cvar_penalty: f64,
    pub drawdown_penalty: f64,
    pub margin_penalty: f64,
    pub uncertainty_penalty: f64,
    pub trade_penalty: f64,
    pub total: f64,
}

#[derive(Clone, Debug)]
pub struct TradePlan {
    pub first_trade: DVector<f64>,
    pub controls: Vec<DVector<f64>>,
    pub objective: f64,
    pub stages: Vec<StageBreakdown>,
}

pub struct OptionTradingMpc {
    config: TradingMpcConfig,
    limits: RiskLimits,
}

impl OptionTradingMpc {
    pub fn new(config: TradingMpcConfig, limits: RiskLimits) -> Result<Self> {
        if !(0.0..1.0).contains(&config.cvar_beta) {
            return Err(TradingError::InvalidCvarBeta(config.cvar_beta));
        }
        validate_non_negative("discount", config.discount)?;
        validate_non_negative("step_lots", config.step_lots)?;
        validate_non_negative("min_improvement", config.min_improvement)?;

        Ok(Self { config, limits })
    }

    pub fn horizon(&self) -> usize {
        self.config.horizon
    }

    pub fn optimize_next_trade(
        &self,
        state: &PortfolioState,
        forecasts: &[StageForecast],
    ) -> Result<TradePlan> {
        self.validate_inputs(state, forecasts)?;

        let instrument_count = state.positions.len();
        let horizon = self.config.horizon;
        let mut controls = vec![DVector::zeros(instrument_count); horizon];
        let (mut best_objective, mut best_stages) =
            self.evaluate_plan(state, forecasts, &controls)?;

        for _ in 0..self.config.coordinate_passes {
            let mut improved = false;

            for stage in 0..horizon {
                for instrument in 0..instrument_count {
                    let lot = forecasts[stage].instruments[instrument].lot_size.max(1.0);
                    let step = lot * self.config.step_lots;

                    for direction in [1.0, -1.0] {
                        let mut candidate = controls.clone();
                        candidate[stage][instrument] += direction * step;

                        if !self.plan_is_feasible(state, forecasts, &candidate)? {
                            continue;
                        }

                        let (candidate_objective, candidate_stages) =
                            self.evaluate_plan(state, forecasts, &candidate)?;
                        if candidate_objective + self.config.min_improvement < best_objective {
                            controls = candidate;
                            best_objective = candidate_objective;
                            best_stages = candidate_stages;
                            improved = true;
                        }
                    }
                }
            }

            if !improved {
                break;
            }
        }

        Ok(TradePlan {
            first_trade: controls[0].clone(),
            controls,
            objective: best_objective,
            stages: best_stages,
        })
    }

    pub fn evaluate_plan(
        &self,
        state: &PortfolioState,
        forecasts: &[StageForecast],
        controls: &[DVector<f64>],
    ) -> Result<(f64, Vec<StageBreakdown>)> {
        self.validate_inputs(state, forecasts)?;
        if controls.len() < self.config.horizon {
            return Err(TradingError::HorizonLength {
                expected: self.config.horizon,
                actual: controls.len(),
            });
        }

        let mut simulated = state.clone();
        let mut objective = 0.0;
        let mut discount = 1.0;
        let mut stages = Vec::with_capacity(self.config.horizon);

        for stage in 0..self.config.horizon {
            let forecast = &forecasts[stage];
            let trade = &controls[stage];
            if trade.len() != simulated.positions.len() {
                return Err(TradingError::VectorLength {
                    expected: simulated.positions.len(),
                    actual: trade.len(),
                });
            }

            let breakdown = self.stage_loss(&simulated, forecast, trade);
            objective += discount * breakdown.total;
            discount *= self.config.discount;
            self.apply_trade(&mut simulated, forecast, trade, &breakdown);
            stages.push(breakdown);
        }

        Ok((objective, stages))
    }

    fn stage_loss(
        &self,
        state: &PortfolioState,
        forecast: &StageForecast,
        trade: &DVector<f64>,
    ) -> StageBreakdown {
        let mut edge = 0.0;
        let mut transaction_cost = 0.0;
        let mut predicted_greeks = state.greeks;
        let mut trade_penalty = 0.0;

        for (instrument, quantity) in forecast.instruments.iter().zip(trade.iter()) {
            edge += instrument.trade_edge(*quantity);
            transaction_cost += instrument.transaction_cost(*quantity);
            predicted_greeks.add_scaled(instrument.greeks, *quantity);
            trade_penalty += quantity.powi(2);
        }

        let greek_penalty = predicted_greeks
            .weighted_square_error(forecast.target_greeks, self.config.weights.greek);
        let cvar_penalty =
            self.config.weights.cvar * cvar(&forecast.loss_scenarios, self.config.cvar_beta);
        let drawdown_penalty = self.config.weights.drawdown * state.drawdown_ratio().powi(2);
        let margin_ratio = self.margin_ratio(state, forecast, trade);
        let margin_penalty = self.config.weights.margin * margin_ratio.powi(2);
        let uncertainty_penalty =
            self.config.weights.model_uncertainty * forecast.gp_covariance_trace.max(0.0);
        let trade_penalty = self.config.weights.trade_size * trade_penalty;

        let total = -edge
            + transaction_cost
            + greek_penalty
            + cvar_penalty
            + drawdown_penalty
            + margin_penalty
            + uncertainty_penalty
            + trade_penalty;

        StageBreakdown {
            edge,
            transaction_cost,
            greek_penalty,
            cvar_penalty,
            drawdown_penalty,
            margin_penalty,
            uncertainty_penalty,
            trade_penalty,
            total,
        }
    }

    fn apply_trade(
        &self,
        state: &mut PortfolioState,
        forecast: &StageForecast,
        trade: &DVector<f64>,
        breakdown: &StageBreakdown,
    ) {
        for (idx, quantity) in trade.iter().enumerate() {
            state.positions[idx] += quantity;
            state
                .greeks
                .add_scaled(forecast.instruments[idx].greeks, *quantity);
        }

        state.wealth += breakdown.edge - breakdown.transaction_cost;
        state.peak_wealth = state.peak_wealth.max(state.wealth);
    }

    fn plan_is_feasible(
        &self,
        state: &PortfolioState,
        forecasts: &[StageForecast],
        controls: &[DVector<f64>],
    ) -> Result<bool> {
        let mut simulated = state.clone();

        for stage in 0..self.config.horizon {
            let forecast = &forecasts[stage];
            let trade = &controls[stage];

            if !self.stage_is_feasible(&simulated, forecast, trade) {
                return Ok(false);
            }

            let breakdown = self.stage_loss(&simulated, forecast, trade);
            self.apply_trade(&mut simulated, forecast, trade, &breakdown);
        }

        Ok(true)
    }

    fn stage_is_feasible(
        &self,
        state: &PortfolioState,
        forecast: &StageForecast,
        trade: &DVector<f64>,
    ) -> bool {
        for idx in 0..trade.len() {
            let quantity = trade[idx];
            let instrument = &forecast.instruments[idx];

            if quantity > instrument.max_buy || -quantity > instrument.max_sell {
                return false;
            }

            if quantity.abs() + self.config.tightening.position_buffer
                > self.limits.max_abs_trade[idx]
            {
                return false;
            }

            let next_position = state.positions[idx] + quantity;
            if next_position.abs() + self.config.tightening.position_buffer
                > self.limits.max_abs_position[idx]
            {
                return false;
            }
        }

        let margin =
            self.margin_ratio(state, forecast, trade) + self.config.tightening.margin_buffer;
        if margin > self.limits.max_margin_ratio {
            return false;
        }

        let drawdown = state.drawdown_ratio() + self.config.tightening.drawdown_buffer;
        if drawdown > self.limits.max_drawdown_ratio {
            return false;
        }

        let leverage =
            self.leverage(state, forecast, trade) + self.config.tightening.leverage_buffer;
        leverage <= self.limits.max_leverage
    }

    fn margin_ratio(
        &self,
        state: &PortfolioState,
        forecast: &StageForecast,
        trade: &DVector<f64>,
    ) -> f64 {
        if state.wealth <= 0.0 {
            return f64::INFINITY;
        }

        let margin: f64 = forecast
            .instruments
            .iter()
            .zip(state.positions.iter().zip(trade.iter()))
            .map(|(instrument, (position, quantity))| {
                instrument.margin_per_unit * (position + quantity).abs()
            })
            .sum();

        margin / state.wealth
    }

    fn leverage(
        &self,
        state: &PortfolioState,
        forecast: &StageForecast,
        trade: &DVector<f64>,
    ) -> f64 {
        if state.wealth <= 0.0 {
            return f64::INFINITY;
        }

        let gross_notional: f64 = forecast
            .instruments
            .iter()
            .zip(state.positions.iter().zip(trade.iter()))
            .map(|(instrument, (position, quantity))| {
                instrument.mid() * (position + quantity).abs()
            })
            .sum();

        gross_notional / state.wealth
    }

    fn validate_inputs(&self, state: &PortfolioState, forecasts: &[StageForecast]) -> Result<()> {
        let instrument_count = state.positions.len();

        if forecasts.len() < self.config.horizon {
            return Err(TradingError::HorizonLength {
                expected: self.config.horizon,
                actual: forecasts.len(),
            });
        }

        validate_vector_length(instrument_count, self.limits.max_abs_position.len())?;
        validate_vector_length(instrument_count, self.limits.max_abs_trade.len())?;

        for forecast in forecasts.iter().take(self.config.horizon) {
            if forecast.instruments.len() != instrument_count {
                return Err(TradingError::InstrumentCount {
                    expected: instrument_count,
                    actual: forecast.instruments.len(),
                });
            }
        }

        Ok(())
    }
}

pub fn cvar(losses: &[f64], beta: f64) -> f64 {
    if losses.is_empty() {
        return 0.0;
    }

    let mut sorted = losses.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let tail_count = ((1.0 - beta) * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted.iter().take(tail_count).sum::<f64>() / tail_count as f64
}

fn validate_vector_length(expected: usize, actual: usize) -> Result<()> {
    if expected != actual {
        Err(TradingError::VectorLength { expected, actual })
    } else {
        Ok(())
    }
}

fn validate_non_negative(name: &'static str, value: f64) -> Result<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(TradingError::NonNegativeConfig { name })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cvar_averages_worst_tail_losses() {
        let losses = [1.0, 5.0, 2.0, 10.0];
        assert_eq!(cvar(&losses, 0.5), 7.5);
    }

    #[test]
    fn optimizer_buys_positive_edge_when_risk_allows() {
        let state = PortfolioState::new(
            DVector::from_vec(vec![0.0]),
            Greeks::default(),
            10_000.0,
            10_000.0,
        );
        let forecast = StageForecast {
            instruments: vec![InstrumentSnapshot {
                bid: 9.8,
                ask: 10.0,
                fair_value: 11.0,
                greeks: Greeks {
                    delta: 0.2,
                    gamma: 0.01,
                    vega: 0.5,
                    theta: -0.02,
                    vanna: 0.0,
                    volga: 0.0,
                },
                transaction_cost_per_unit: 0.05,
                margin_per_unit: 20.0,
                max_buy: 10.0,
                max_sell: 10.0,
                lot_size: 1.0,
            }],
            target_greeks: Greeks::default(),
            loss_scenarios: vec![0.1, 0.2, 0.4],
            gp_covariance_trace: 0.01,
        };
        let limits = RiskLimits {
            max_abs_position: DVector::from_vec(vec![10.0]),
            max_abs_trade: DVector::from_vec(vec![10.0]),
            max_margin_ratio: 1.0,
            max_drawdown_ratio: 0.2,
            max_leverage: 10.0,
        };
        let config = TradingMpcConfig {
            horizon: 1,
            weights: ObjectiveWeights {
                greek: Greeks::default(),
                cvar: 0.0,
                drawdown: 0.0,
                margin: 0.0,
                model_uncertainty: 0.0,
                trade_size: 0.0,
            },
            ..TradingMpcConfig::default()
        };

        let mpc = OptionTradingMpc::new(config, limits).unwrap();
        let plan = mpc.optimize_next_trade(&state, &[forecast]).unwrap();

        assert!(plan.first_trade[0] > 0.0);
        assert!(plan.stages[0].edge > 0.0);
    }
}
