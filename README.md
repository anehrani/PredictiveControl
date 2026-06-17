# grampc-s-rs

Rust-native building blocks inspired by [GRAMPC-S](https://github.com/grampc/grampc-s).

This project is intended to be a pure Rust implementation. It should not depend on
the upstream C++ codebase, CMake build files, MATLAB scripts, Python bindings, or
foreign-language solver wrappers. New functionality should be implemented with Rust
code and Rust crates.

This is not a direct binding to the external GRAMPC solver. It ports the solver-independent
stochastic MPC pieces first:

- moment-based distributions and sampling, including the upstream univariate distribution families,
- Monte Carlo, unscented, Stirling first/second-order, composed Gaussian quadrature, and
  polynomial-chaos transformations,
- Gaussian, Chebyshev, and symmetric chance-constraint tightening,
- Gaussian-process residual models with squared-exponential, Matern, periodic, locally periodic,
  sum, and product kernels,
- univariate/multivariate polynomials with Hermite and Legendre generators,
- lightweight problem, stochastic dynamics moment propagation, RK4 simulation, shooting solver
  with quadratic-penalty and augmented-Lagrangian inequality handling, and stochastic MPC
  moment-state orchestration,
- a double-integrator example based on the upstream example shape.

Run the tests:

```sh
cargo test
```

Run the example:

```sh
cargo run --example double_integrator
cargo run --example double_integrator_solver
cargo run --example stochastic_double_integrator_mpc
```

## Porting Notes

The upstream GRAMPC-S project also contains GRAMPC solver integration, Python/MATLAB bindings,
additional distributions, polynomial-chaos expansion, and more GP kernels. This crate is structured
so those layers can be added incrementally in Rust without changing the public foundations.

Remaining Rust-only porting priorities:

1. Improve the Rust-native solver toward GRAMPC-style line search, scaling, and multiplier updates.
2. Port the remaining upstream examples as Rust examples.
3. Add end-to-end numerical validation against upstream examples and documented formulas.
4. Expand documentation for the Rust-native APIs and solver limitations.
