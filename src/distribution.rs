use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution as RandDistribution, Exp, Normal};

use crate::error::{Error, Result, dim_error};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolynomialFamily {
    Hermite,
    Legendre,
    MomentOnly,
}

pub trait Distribution {
    fn mean(&self) -> &DVector<f64>;
    fn covariance(&self) -> &DMatrix<f64>;
    fn cov_cholesky(&self) -> &DMatrix<f64>;
    fn polynomial_family(&self) -> &[PolynomialFamily];

    fn dimension(&self) -> usize {
        self.mean().len()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> DVector<f64>;
}

#[derive(Debug, Clone)]
pub struct MomentDistribution {
    mean: DVector<f64>,
    covariance: DMatrix<f64>,
    cov_cholesky: DMatrix<f64>,
    polynomial_family: Vec<PolynomialFamily>,
}

impl MomentDistribution {
    pub fn new(
        mean: DVector<f64>,
        covariance: DMatrix<f64>,
        polynomial_family: Vec<PolynomialFamily>,
    ) -> Result<Self> {
        validate_moments(&mean, &covariance)?;
        if polynomial_family.len() != mean.len() {
            return Err(dim_error(
                "polynomial family",
                mean.len().to_string(),
                polynomial_family.len().to_string(),
            ));
        }
        let cov_cholesky = cholesky_factor(&covariance)?;
        Ok(Self {
            mean,
            covariance,
            cov_cholesky,
            polynomial_family,
        })
    }

    pub fn with_cholesky(
        mean: DVector<f64>,
        covariance: DMatrix<f64>,
        cov_cholesky: DMatrix<f64>,
        polynomial_family: Vec<PolynomialFamily>,
    ) -> Result<Self> {
        validate_moments(&mean, &covariance)?;
        if cov_cholesky.shape() != covariance.shape() {
            return Err(dim_error(
                "covariance cholesky",
                format!("{}x{}", covariance.nrows(), covariance.ncols()),
                format!("{}x{}", cov_cholesky.nrows(), cov_cholesky.ncols()),
            ));
        }
        if polynomial_family.len() != mean.len() {
            return Err(dim_error(
                "polynomial family",
                mean.len().to_string(),
                polynomial_family.len().to_string(),
            ));
        }
        Ok(Self {
            mean,
            covariance,
            cov_cholesky,
            polynomial_family,
        })
    }
}

impl Distribution for MomentDistribution {
    fn mean(&self) -> &DVector<f64> {
        &self.mean
    }

    fn covariance(&self) -> &DMatrix<f64> {
        &self.covariance
    }

    fn cov_cholesky(&self) -> &DMatrix<f64> {
        &self.cov_cholesky
    }

    fn polynomial_family(&self) -> &[PolynomialFamily] {
        &self.polynomial_family
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> DVector<f64> {
        let normal = Normal::new(0.0, 1.0).expect("valid standard normal");
        let z = DVector::from_fn(self.dimension(), |_, _| normal.sample(rng));
        &self.mean + &self.cov_cholesky * z
    }
}

#[derive(Debug, Clone)]
pub struct Gaussian {
    moments: MomentDistribution,
}

impl Gaussian {
    pub fn new(mean: DVector<f64>, covariance: DMatrix<f64>) -> Result<Self> {
        let polynomial_family = vec![PolynomialFamily::Hermite; mean.len()];
        Ok(Self {
            moments: MomentDistribution::new(mean, covariance, polynomial_family)?,
        })
    }

    pub fn univariate(mean: f64, variance: f64) -> Result<Self> {
        if variance <= 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "variance",
                value: variance,
            });
        }
        Self::new(
            DVector::from_element(1, mean),
            DMatrix::from_element(1, 1, variance),
        )
    }
}

impl Distribution for Gaussian {
    fn mean(&self) -> &DVector<f64> {
        self.moments.mean()
    }

    fn covariance(&self) -> &DMatrix<f64> {
        self.moments.covariance()
    }

    fn cov_cholesky(&self) -> &DMatrix<f64> {
        self.moments.cov_cholesky()
    }

    fn polynomial_family(&self) -> &[PolynomialFamily] {
        self.moments.polynomial_family()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> DVector<f64> {
        self.moments.sample(rng)
    }
}

#[derive(Debug, Clone)]
pub struct Uniform {
    lower: f64,
    upper: f64,
    moments: MomentDistribution,
}

impl Uniform {
    pub fn univariate(lower: f64, upper: f64) -> Result<Self> {
        if lower >= upper {
            return Err(dim_error(
                "uniform bounds",
                "lower < upper",
                format!("{lower} >= {upper}"),
            ));
        }
        let mean = 0.5 * (lower + upper);
        let variance = (upper - lower).powi(2) / 12.0;
        Ok(Self {
            lower,
            upper,
            moments: MomentDistribution::new(
                DVector::from_element(1, mean),
                DMatrix::from_element(1, 1, variance),
                vec![PolynomialFamily::Legendre],
            )?,
        })
    }
}

impl Distribution for Uniform {
    fn mean(&self) -> &DVector<f64> {
        self.moments.mean()
    }

    fn covariance(&self) -> &DMatrix<f64> {
        self.moments.covariance()
    }

    fn cov_cholesky(&self) -> &DMatrix<f64> {
        self.moments.cov_cholesky()
    }

    fn polynomial_family(&self) -> &[PolynomialFamily] {
        self.moments.polynomial_family()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> DVector<f64> {
        DVector::from_element(1, rng.random_range(self.lower..=self.upper))
    }
}

#[derive(Debug, Clone)]
pub struct Exponential {
    lambda: f64,
    moments: MomentDistribution,
}

impl Exponential {
    pub fn univariate(lambda: f64) -> Result<Self> {
        if lambda <= 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "lambda",
                value: lambda,
            });
        }
        let mean = 1.0 / lambda;
        let variance = 1.0 / lambda.powi(2);
        Ok(Self {
            lambda,
            moments: MomentDistribution::new(
                DVector::from_element(1, mean),
                DMatrix::from_element(1, 1, variance),
                vec![PolynomialFamily::MomentOnly],
            )?,
        })
    }
}

impl Distribution for Exponential {
    fn mean(&self) -> &DVector<f64> {
        self.moments.mean()
    }

    fn covariance(&self) -> &DMatrix<f64> {
        self.moments.covariance()
    }

    fn cov_cholesky(&self) -> &DMatrix<f64> {
        self.moments.cov_cholesky()
    }

    fn polynomial_family(&self) -> &[PolynomialFamily] {
        self.moments.polynomial_family()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> DVector<f64> {
        let exp = Exp::new(self.lambda).expect("positive lambda");
        DVector::from_element(1, exp.sample(rng))
    }
}

#[derive(Debug, Clone)]
pub struct MultivariateUncorrelated<D> {
    components: Vec<D>,
    moments: MomentDistribution,
}

impl<D: Distribution> MultivariateUncorrelated<D> {
    pub fn new(components: Vec<D>) -> Result<Self> {
        if components.is_empty() {
            return Err(Error::Empty("components"));
        }
        let dimension: usize = components.iter().map(Distribution::dimension).sum();
        let mut mean = DVector::zeros(dimension);
        let mut covariance = DMatrix::zeros(dimension, dimension);
        let mut families = Vec::with_capacity(dimension);
        let mut offset = 0;
        for component in &components {
            let dim = component.dimension();
            mean.rows_mut(offset, dim).copy_from(component.mean());
            covariance
                .view_mut((offset, offset), (dim, dim))
                .copy_from(component.covariance());
            families.extend_from_slice(component.polynomial_family());
            offset += dim;
        }
        let moments = MomentDistribution::new(mean, covariance, families)?;
        Ok(Self {
            components,
            moments,
        })
    }
}

impl<D: Distribution> Distribution for MultivariateUncorrelated<D> {
    fn mean(&self) -> &DVector<f64> {
        self.moments.mean()
    }

    fn covariance(&self) -> &DMatrix<f64> {
        self.moments.covariance()
    }

    fn cov_cholesky(&self) -> &DMatrix<f64> {
        self.moments.cov_cholesky()
    }

    fn polynomial_family(&self) -> &[PolynomialFamily] {
        self.moments.polynomial_family()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> DVector<f64> {
        let mut out = DVector::zeros(self.dimension());
        let mut offset = 0;
        for component in &self.components {
            let sample = component.sample(rng);
            let dim = sample.len();
            out.rows_mut(offset, dim).copy_from(&sample);
            offset += dim;
        }
        out
    }
}

fn validate_moments(mean: &DVector<f64>, covariance: &DMatrix<f64>) -> Result<()> {
    if covariance.nrows() != mean.len() || covariance.ncols() != mean.len() {
        return Err(dim_error(
            "moments",
            format!("{}x{}", mean.len(), mean.len()),
            format!("{}x{}", covariance.nrows(), covariance.ncols()),
        ));
    }
    Ok(())
}

fn cholesky_factor(covariance: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    covariance
        .clone()
        .cholesky()
        .map(|factor| factor.l())
        .ok_or(Error::NotPositiveDefinite("covariance"))
}
