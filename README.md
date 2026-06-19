# Predictive control for Options

This crate models option trading as a receding-horizon risk allocator:

1. Forecast option/future/underlying snapshots over a horizon.
2. Optimize future trade vectors `U_0, ..., U_{N-1}`.
3. Execute only `U_0`.
4. Recompute the plan at the next market step.

The optimizer minimizes a risk-adjusted stage loss:

```text
-expected_edge
+ transaction_cost
+ Greek tracking penalty
+ CVaR tail-loss penalty
+ drawdown penalty
+ margin/leverage penalty
+ GP/model-uncertainty penalty
+ trade-size penalty
```

The public entry point is `predictive_control::options::OptionTradingMpc`.
It returns buy/sell quantities, not option price predictions.

## Library boundary

This repo should remain exchange-agnostic. A separate exchange-data repo can
depend on this crate and implement:

```rust
use predictive_control::{
    OptionTradingDataSource, PortfolioState, StageForecast,
};

struct ExchangeDataSource;

impl OptionTradingDataSource for ExchangeDataSource {
    type Error = ExchangeError;

    fn current_portfolio(&self) -> Result<PortfolioState, Self::Error> {
        // Convert exchange positions/account data into PortfolioState.
    }

    fn forecasts(&self, horizon: usize) -> Result<Vec<StageForecast>, Self::Error> {
        // Convert exchange option-chain data, fair values, Greeks,
        // scenario losses, and model uncertainty into StageForecast values.
    }
}
```

Then the trading app calls:

```rust
let plan = predictive_control::plan_next_trade_from_source(&mpc, &exchange_data)?;
let trade_to_execute_now = plan.first_trade;
```

Run the option allocator example with:

```sh
cargo run --example options_trade_allocator
```
