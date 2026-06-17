//! Rust-native stochastic MPC building blocks inspired by GRAMPC-S.
//!
//! The original GRAMPC-S project couples stochastic approximations to the
//! GRAMPC nonlinear MPC solver. This crate ports the solver-independent pieces
//! first: distributions, sigma/quad point transformations, chance constraints,
//! Gaussian-process residual models, and lightweight problem/simulation traits.

pub mod constraints;
pub mod distribution;
pub mod error;
pub mod gaussian_process;
pub mod problem;
pub mod simulator;
pub mod transformation;

pub use constraints::*;
pub use distribution::*;
pub use error::{Error, Result};
pub use gaussian_process::*;
pub use problem::*;
pub use simulator::*;
pub use transformation::*;

pub type Vector = nalgebra::DVector<f64>;
pub type Matrix = nalgebra::DMatrix<f64>;
