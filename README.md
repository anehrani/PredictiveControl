# grampc-s-rs

Rust-native building blocks inspired by [GRAMPC-S](https://github.com/grampc/grampc-s).

This is not a direct binding to the external GRAMPC solver. It ports the solver-independent
stochastic MPC pieces first:

- moment-based distributions and sampling,
- unscented, Stirling first-order, and composed Gaussian quadrature transformations,
- Gaussian, Chebyshev, and symmetric chance-constraint tightening,
- squared-exponential Gaussian-process residual models,
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

The upstream C++ project also contains GRAMPC solver integration, Python/MATLAB bindings,
additional distributions, polynomial-chaos expansion, and more GP kernels. This crate is structured
so those layers can be added incrementally without changing the public foundations.
