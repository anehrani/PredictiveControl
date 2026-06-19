use nalgebra::DVector;
use predictive_control::{
    Greeks, InstrumentSnapshot, ObjectiveWeights, OptionTradingDataSource, OptionTradingMpc,
    PortfolioState, RiskLimits, StageForecast, TradingMpcConfig, plan_next_trade_from_source,
};

struct StaticDataSource {
    state: PortfolioState,
    forecast: StageForecast,
}

impl OptionTradingDataSource for StaticDataSource {
    type Error = std::convert::Infallible;

    fn current_portfolio(&self) -> std::result::Result<PortfolioState, Self::Error> {
        Ok(self.state.clone())
    }

    fn forecasts(&self, horizon: usize) -> std::result::Result<Vec<StageForecast>, Self::Error> {
        Ok(vec![self.forecast.clone(); horizon])
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let state = PortfolioState::new(
        DVector::from_vec(vec![0.0, 0.0, 0.0]),
        Greeks::default(),
        100_000.0,
        100_000.0,
    );

    let instruments = vec![
        InstrumentSnapshot {
            bid: 2.35,
            ask: 2.45,
            fair_value: 3.50,
            greeks: Greeks {
                delta: 0.34,
                gamma: 0.04,
                vega: 0.18,
                theta: -0.03,
                vanna: 0.01,
                volga: 0.02,
            },
            transaction_cost_per_unit: 0.02,
            margin_per_unit: 550.0,
            max_buy: 20.0,
            max_sell: 20.0,
            lot_size: 1.0,
        },
        InstrumentSnapshot {
            bid: 4_999.75,
            ask: 5_000.25,
            fair_value: 5_000.00,
            greeks: Greeks {
                delta: 1.0,
                ..Greeks::default()
            },
            transaction_cost_per_unit: 0.25,
            margin_per_unit: 250.0,
            max_buy: 5.0,
            max_sell: 5.0,
            lot_size: 1.0,
        },
        InstrumentSnapshot {
            bid: 1.90,
            ask: 2.00,
            fair_value: 1.30,
            greeks: Greeks {
                delta: -0.28,
                gamma: 0.03,
                vega: 0.15,
                theta: -0.02,
                vanna: -0.01,
                volga: 0.02,
            },
            transaction_cost_per_unit: 0.02,
            margin_per_unit: 600.0,
            max_buy: 20.0,
            max_sell: 20.0,
            lot_size: 1.0,
        },
    ];

    let forecast = StageForecast {
        instruments,
        target_greeks: Greeks {
            delta: 0.0,
            gamma: 0.2,
            vega: -0.5,
            theta: 0.1,
            vanna: 0.0,
            volga: 0.0,
        },
        loss_scenarios: vec![150.0, 220.0, 80.0, 450.0, 120.0],
        gp_covariance_trace: 0.15,
    };

    let limits = RiskLimits {
        max_abs_position: DVector::from_vec(vec![20.0, 5.0, 20.0]),
        max_abs_trade: DVector::from_vec(vec![5.0, 2.0, 5.0]),
        max_margin_ratio: 0.25,
        max_drawdown_ratio: 0.10,
        max_leverage: 2.0,
    };

    let config = TradingMpcConfig {
        horizon: 3,
        coordinate_passes: 8,
        weights: ObjectiveWeights {
            greek: Greeks {
                delta: 0.05,
                gamma: 0.01,
                vega: 0.01,
                theta: 0.0,
                vanna: 0.0,
                volga: 0.0,
            },
            cvar: 0.0,
            drawdown: 100.0,
            margin: 0.1,
            model_uncertainty: 0.0,
            trade_size: 0.001,
        },
        ..TradingMpcConfig::default()
    };

    let mpc = OptionTradingMpc::new(config, limits)?;
    let data_source = StaticDataSource { state, forecast };
    let plan = plan_next_trade_from_source(&mpc, &data_source)?;

    println!("objective: {:.4}", plan.objective);
    println!("execute now: {}", plan.first_trade.transpose());
    println!("stage 0 edge: {:.4}", plan.stages[0].edge);
    println!("stage 0 loss: {:.4}", plan.stages[0].total);

    Ok(())
}
