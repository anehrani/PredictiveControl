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
pub mod polynomial;
pub mod problem;
pub mod simulator;
pub mod smpc;
pub mod solver;
pub mod stochastic;
pub mod transformation;

pub use constraints::*;
pub use distribution::*;
pub use error::{Error, Result};
pub use gaussian_process::*;
pub use polynomial::*;
pub use problem::*;
pub use simulator::*;
pub use smpc::*;
pub use solver::*;
pub use stochastic::*;
pub use transformation::*;

pub type Vector = nalgebra::DVector<f64>;
pub type Matrix = nalgebra::DMatrix<f64>;
