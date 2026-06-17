# grampc-s-rs

Rust-native building blocks inspired by [GRAMPC-S](https://github.com/grampc/grampc-s).

This project is intended to be a pure Rust implementation. It should not depend on
the upstream C++ codebase, CMake build files, MATLAB scripts, Python bindings, or
foreign-language solver wrappers. New functionality should be implemented with Rust
code and Rust crates.

This is not a direct binding to the external GRAMPC solver. It ports the solver-independent
stochastic MPC pieces first:

- moment-based distributions and sampling, including the upstream univariate distribution families,
- Monte Carlo, unscented, Stirling first/second-order, and composed Gaussian quadrature transformations,
- Gaussian, Chebyshev, and symmetric chance-constraint tightening,
- Gaussian-process residual models with squared-exponential, Matern, periodic, locally periodic,
  sum, and product kernels,
- univariate polynomials with Hermite and Legendre generators,
- lightweight problem and RK4 simulator traits,
- a double-integrator example based on the upstream example shape.

Run the tests:

```sh
cargo test
```

Run the example:

```sh
cargo run --example double_integrator
```

## Porting Notes

The upstream GRAMPC-S project also contains GRAMPC solver integration, Python/MATLAB bindings,
additional distributions, polynomial-chaos expansion, and more GP kernels. This crate is structured
so those layers can be added incrementally in Rust without changing the public foundations.

Rust-only porting priorities:

1. Add multivariate polynomials and polynomial-chaos expansion.
2. Add stochastic problem-description strategies: Taylor, sigma-point, Monte Carlo, and resampling.
3. Add Rust-native deterministic optimal-control/SMPC solver integration.
4. Port the upstream examples as Rust examples.
