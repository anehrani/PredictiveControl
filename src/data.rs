use crate::options::{OptionTradingMpc, PortfolioState, StageForecast, TradePlan, TradingError};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub trait OptionTradingDataSource {
    type Error;

    fn current_portfolio(&self) -> std::result::Result<PortfolioState, Self::Error>;

    fn forecasts(&self, horizon: usize) -> std::result::Result<Vec<StageForecast>, Self::Error>;
}

#[derive(Debug)]
pub enum PlanFromSourceError<E> {
    Source(E),
    Trading(TradingError),
}

impl<E: Display> Display for PlanFromSourceError<E> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Source(error) => write!(formatter, "data source error: {error}"),
            Self::Trading(error) => write!(formatter, "trading optimizer error: {error}"),
        }
    }
}

impl<E> Error for PlanFromSourceError<E>
where
    E: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Source(error) => Some(error),
            Self::Trading(error) => Some(error),
        }
    }
}

pub fn plan_next_trade_from_source<S>(
    optimizer: &OptionTradingMpc,
    source: &S,
) -> std::result::Result<TradePlan, PlanFromSourceError<S::Error>>
where
    S: OptionTradingDataSource,
{
    let state = source
        .current_portfolio()
        .map_err(PlanFromSourceError::Source)?;
    let forecasts = source
        .forecasts(optimizer.horizon())
        .map_err(PlanFromSourceError::Source)?;

    optimizer
        .optimize_next_trade(&state, &forecasts)
        .map_err(PlanFromSourceError::Trading)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::{
        Greeks, InstrumentSnapshot, ObjectiveWeights, PortfolioState, RiskLimits, StageForecast,
        TradingMpcConfig,
    };
    use nalgebra::DVector;

    struct StaticSource {
        state: PortfolioState,
        forecast: StageForecast,
    }

    impl OptionTradingDataSource for StaticSource {
        type Error = std::convert::Infallible;

        fn current_portfolio(&self) -> std::result::Result<PortfolioState, Self::Error> {
            Ok(self.state.clone())
        }

        fn forecasts(
            &self,
            horizon: usize,
        ) -> std::result::Result<Vec<StageForecast>, Self::Error> {
            Ok(vec![self.forecast.clone(); horizon])
        }
    }

    #[test]
    fn plans_from_data_source() {
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
                greeks: Greeks::default(),
                transaction_cost_per_unit: 0.01,
                margin_per_unit: 10.0,
                max_buy: 5.0,
                max_sell: 5.0,
                lot_size: 1.0,
            }],
            target_greeks: Greeks::default(),
            loss_scenarios: vec![0.0],
            gp_covariance_trace: 0.0,
        };
        let optimizer = OptionTradingMpc::new(
            TradingMpcConfig {
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
            },
            RiskLimits {
                max_abs_position: DVector::from_vec(vec![5.0]),
                max_abs_trade: DVector::from_vec(vec![5.0]),
                max_margin_ratio: 1.0,
                max_drawdown_ratio: 1.0,
                max_leverage: 1.0,
            },
        )
        .unwrap();

        let source = StaticSource { state, forecast };
        let plan = plan_next_trade_from_source(&optimizer, &source).unwrap();

        assert!(plan.first_trade[0] > 0.0);
    }
}
