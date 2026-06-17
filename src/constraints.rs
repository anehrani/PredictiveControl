use nalgebra::DVector;
use statrs::distribution::{ContinuousCDF, Normal};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChanceConstraintMethod {
    Gaussian,
    Chebyshev,
    Symmetric,
}

#[derive(Debug, Clone)]
pub struct ChanceConstraintApproximation {
    method: ChanceConstraintMethod,
    probabilities: DVector<f64>,
    tightening_coefficients: DVector<f64>,
}

impl ChanceConstraintApproximation {
    pub fn gaussian(probabilities: DVector<f64>) -> Result<Self> {
        Self::new(ChanceConstraintMethod::Gaussian, probabilities)
    }

    pub fn chebyshev(probabilities: DVector<f64>) -> Result<Self> {
        Self::new(ChanceConstraintMethod::Chebyshev, probabilities)
    }

    pub fn symmetric(probabilities: DVector<f64>) -> Result<Self> {
        Self::new(ChanceConstraintMethod::Symmetric, probabilities)
    }

    pub fn new(method: ChanceConstraintMethod, probabilities: DVector<f64>) -> Result<Self> {
        validate_probabilities(&probabilities)?;
        let tightening_coefficients = compute_coefficients(method, &probabilities)?;
        Ok(Self {
            method,
            probabilities,
            tightening_coefficients,
        })
    }

    pub fn method(&self) -> ChanceConstraintMethod {
        self.method
    }

    pub fn probabilities(&self) -> &DVector<f64> {
        &self.probabilities
    }

    pub fn tightening_coefficients(&self) -> &DVector<f64> {
        &self.tightening_coefficients
    }

    pub fn set_probabilities(&mut self, probabilities: DVector<f64>) -> Result<()> {
        validate_probabilities(&probabilities)?;
        self.tightening_coefficients = compute_coefficients(self.method, &probabilities)?;
        self.probabilities = probabilities;
        Ok(())
    }

    pub fn tighten_upper_bound(&self, mean: f64, std_dev: f64, index: usize) -> f64 {
        mean + self.tightening_coefficients[index] * std_dev
    }
}

fn validate_probabilities(probabilities: &DVector<f64>) -> Result<()> {
    if probabilities.is_empty() {
        return Err(Error::Empty("probabilities"));
    }
    for probability in probabilities.iter().copied() {
        if !(0.0..1.0).contains(&probability) {
            return Err(Error::InvalidProbability(probability));
        }
    }
    Ok(())
}

fn compute_coefficients(
    method: ChanceConstraintMethod,
    probabilities: &DVector<f64>,
) -> Result<DVector<f64>> {
    let normal = Normal::new(0.0, 1.0).expect("valid standard normal");
    Ok(probabilities.map(|probability| match method {
        ChanceConstraintMethod::Gaussian => normal.inverse_cdf(probability),
        ChanceConstraintMethod::Chebyshev => (probability / (1.0 - probability)).sqrt(),
        ChanceConstraintMethod::Symmetric => (1.0 / (2.0 * (1.0 - probability))).sqrt(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_known_tightening_coefficients() {
        let cheb = ChanceConstraintApproximation::chebyshev(DVector::from_vec(vec![0.8])).unwrap();
        assert!((cheb.tightening_coefficients()[0] - 2.0).abs() < 1e-12);

        let sym = ChanceConstraintApproximation::symmetric(DVector::from_vec(vec![0.5])).unwrap();
        assert!((sym.tightening_coefficients()[0] - 1.0).abs() < 1e-12);

        let gauss =
            ChanceConstraintApproximation::gaussian(DVector::from_vec(vec![0.975])).unwrap();
        assert!((gauss.tightening_coefficients()[0] - 1.959963984540054).abs() < 1e-12);
    }
}
