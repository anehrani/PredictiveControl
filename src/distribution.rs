use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{
    Beta as RandBeta, ChiSquared as RandChiSquared, Distribution as RandDistribution, Exp,
    FisherF as RandFisherF, Gamma as RandGamma, Gumbel as RandGumbel, LogNormal as RandLogNormal,
    Normal, StudentT as RandStudentT, Weibull as RandWeibull,
};
use statrs::function::gamma::gamma;

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
pub struct PiecewiseConstant {
    interval_limits: Vec<f64>,
    cumulative_weights: Vec<f64>,
    total_weight: f64,
    moments: MomentDistribution,
}

impl PiecewiseConstant {
    pub fn univariate(interval_limits: Vec<f64>, interval_density: Vec<f64>) -> Result<Self> {
        if interval_limits.len() < 2 {
            return Err(Error::Empty("interval_limits"));
        }
        if interval_density.len() + 1 != interval_limits.len() {
            return Err(dim_error(
                "piecewise constant density",
                format!("{} densities", interval_limits.len() - 1),
                format!("{} densities", interval_density.len()),
            ));
        }

        let mut cumulative_weights = Vec::with_capacity(interval_density.len());
        let mut total_weight = 0.0;
        let mut weighted_mean = 0.0;
        let mut weighted_second_moment = 0.0;

        for (interval_index, density) in interval_density.iter().copied().enumerate() {
            if density < 0.0 {
                return Err(Error::NonPositiveParameter {
                    name: "interval_density",
                    value: density,
                });
            }
            let lower = interval_limits[interval_index];
            let upper = interval_limits[interval_index + 1];
            if lower >= upper {
                return Err(dim_error(
                    "piecewise constant interval",
                    "strictly increasing limits",
                    format!("{lower} >= {upper}"),
                ));
            }
            let width = upper - lower;
            let weight = density * width;
            let interval_mean = 0.5 * (lower + upper);
            let interval_second_moment = width.powi(2) / 12.0 + interval_mean.powi(2);
            total_weight += weight;
            cumulative_weights.push(total_weight);
            weighted_mean += weight * interval_mean;
            weighted_second_moment += weight * interval_second_moment;
        }

        if total_weight <= 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "total_interval_weight",
                value: total_weight,
            });
        }

        let mean = weighted_mean / total_weight;
        let variance = weighted_second_moment / total_weight - mean.powi(2);
        Ok(Self {
            interval_limits,
            cumulative_weights,
            total_weight,
            moments: univariate_moments(mean, variance, PolynomialFamily::MomentOnly)?,
        })
    }
}

impl Distribution for PiecewiseConstant {
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
        let target = rng.random_range(0.0..self.total_weight);
        let interval_index = self
            .cumulative_weights
            .iter()
            .position(|weight| target <= *weight)
            .unwrap_or(self.cumulative_weights.len() - 1);
        DVector::from_element(
            1,
            rng.random_range(
                self.interval_limits[interval_index]..=self.interval_limits[interval_index + 1],
            ),
        )
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
pub struct Gamma {
    shape: f64,
    scale: f64,
    moments: MomentDistribution,
}

impl Gamma {
    pub fn univariate(shape: f64, scale: f64) -> Result<Self> {
        require_positive("shape", shape)?;
        require_positive("scale", scale)?;
        let mean = shape * scale;
        let variance = shape * scale.powi(2);
        Ok(Self {
            shape,
            scale,
            moments: univariate_moments(mean, variance, PolynomialFamily::MomentOnly)?,
        })
    }
}

impl Distribution for Gamma {
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
        let gamma = RandGamma::new(self.shape, self.scale).expect("positive gamma parameters");
        DVector::from_element(1, gamma.sample(rng))
    }
}

#[derive(Debug, Clone)]
pub struct ChiSquared {
    degrees_of_freedom: f64,
    moments: MomentDistribution,
}

impl ChiSquared {
    pub fn univariate(degrees_of_freedom: f64) -> Result<Self> {
        require_positive("degrees_of_freedom", degrees_of_freedom)?;
        Ok(Self {
            degrees_of_freedom,
            moments: univariate_moments(
                degrees_of_freedom,
                2.0 * degrees_of_freedom,
                PolynomialFamily::MomentOnly,
            )?,
        })
    }
}

impl Distribution for ChiSquared {
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
        let chi =
            RandChiSquared::new(self.degrees_of_freedom).expect("positive degrees of freedom");
        DVector::from_element(1, chi.sample(rng))
    }
}

#[derive(Debug, Clone)]
pub struct LogNormal {
    location: f64,
    scale: f64,
    moments: MomentDistribution,
}

impl LogNormal {
    pub fn univariate(location: f64, scale: f64) -> Result<Self> {
        require_positive("scale", scale)?;
        let variance_factor = scale.powi(2).exp() - 1.0;
        let mean = (location + 0.5 * scale.powi(2)).exp();
        let variance = variance_factor * (2.0 * location + scale.powi(2)).exp();
        Ok(Self {
            location,
            scale,
            moments: univariate_moments(mean, variance, PolynomialFamily::MomentOnly)?,
        })
    }
}

impl Distribution for LogNormal {
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
        let log_normal = RandLogNormal::new(self.location, self.scale).expect("positive scale");
        DVector::from_element(1, log_normal.sample(rng))
    }
}

#[derive(Debug, Clone)]
pub struct Weibull {
    scale: f64,
    shape: f64,
    moments: MomentDistribution,
}

impl Weibull {
    pub fn univariate(scale: f64, shape: f64) -> Result<Self> {
        require_positive("scale", scale)?;
        require_positive("shape", shape)?;
        let mean = scale * gamma(1.0 + 1.0 / shape);
        let variance =
            scale.powi(2) * (gamma(1.0 + 2.0 / shape) - gamma(1.0 + 1.0 / shape).powi(2));
        Ok(Self {
            scale,
            shape,
            moments: univariate_moments(mean, variance, PolynomialFamily::MomentOnly)?,
        })
    }
}

impl Distribution for Weibull {
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
        let weibull =
            RandWeibull::new(self.scale, self.shape).expect("positive weibull parameters");
        DVector::from_element(1, weibull.sample(rng))
    }
}

#[derive(Debug, Clone)]
pub struct Beta {
    alpha: f64,
    beta: f64,
    moments: MomentDistribution,
}

impl Beta {
    pub fn univariate(alpha: f64, beta: f64) -> Result<Self> {
        require_positive("alpha", alpha)?;
        require_positive("beta", beta)?;
        let sum = alpha + beta;
        let mean = alpha / sum;
        let variance = alpha * beta / (sum.powi(2) * (sum + 1.0));
        Ok(Self {
            alpha,
            beta,
            moments: univariate_moments(mean, variance, PolynomialFamily::MomentOnly)?,
        })
    }
}

impl Distribution for Beta {
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
        let beta = RandBeta::new(self.alpha, self.beta).expect("positive beta parameters");
        DVector::from_element(1, beta.sample(rng))
    }
}

#[derive(Debug, Clone)]
pub struct StudentT {
    degrees_of_freedom: f64,
    moments: MomentDistribution,
}

impl StudentT {
    pub fn univariate(degrees_of_freedom: f64) -> Result<Self> {
        if degrees_of_freedom <= 2.0 {
            return Err(Error::NonPositiveParameter {
                name: "degrees_of_freedom - 2",
                value: degrees_of_freedom - 2.0,
            });
        }
        Ok(Self {
            degrees_of_freedom,
            moments: univariate_moments(
                0.0,
                degrees_of_freedom / (degrees_of_freedom - 2.0),
                PolynomialFamily::MomentOnly,
            )?,
        })
    }
}

impl Distribution for StudentT {
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
        let student_t =
            RandStudentT::new(self.degrees_of_freedom).expect("finite student-t moments");
        DVector::from_element(1, student_t.sample(rng))
    }
}

#[derive(Debug, Clone)]
pub struct FisherF {
    numerator_degrees_of_freedom: f64,
    denominator_degrees_of_freedom: f64,
    moments: MomentDistribution,
}

impl FisherF {
    pub fn univariate(
        numerator_degrees_of_freedom: f64,
        denominator_degrees_of_freedom: f64,
    ) -> Result<Self> {
        require_positive("numerator_degrees_of_freedom", numerator_degrees_of_freedom)?;
        if denominator_degrees_of_freedom <= 4.0 {
            return Err(Error::NonPositiveParameter {
                name: "denominator_degrees_of_freedom - 4",
                value: denominator_degrees_of_freedom - 4.0,
            });
        }
        let d1 = numerator_degrees_of_freedom;
        let d2 = denominator_degrees_of_freedom;
        let mean = d2 / (d2 - 2.0);
        let variance = 2.0 * d2.powi(2) * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0).powi(2) * (d2 - 4.0));
        Ok(Self {
            numerator_degrees_of_freedom,
            denominator_degrees_of_freedom,
            moments: univariate_moments(mean, variance, PolynomialFamily::MomentOnly)?,
        })
    }
}

impl Distribution for FisherF {
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
        let fisher_f = RandFisherF::new(
            self.numerator_degrees_of_freedom,
            self.denominator_degrees_of_freedom,
        )
        .expect("finite fisher-f moments");
        DVector::from_element(1, fisher_f.sample(rng))
    }
}

#[derive(Debug, Clone)]
pub struct ExtremeValue {
    location: f64,
    scale: f64,
    moments: MomentDistribution,
}

impl ExtremeValue {
    pub fn univariate(location: f64, scale: f64) -> Result<Self> {
        require_positive("scale", scale)?;
        let mean = location + 0.577_215_664_901_532_9 * scale;
        let variance = std::f64::consts::PI.powi(2) * scale.powi(2) / 6.0;
        Ok(Self {
            location,
            scale,
            moments: univariate_moments(mean, variance, PolynomialFamily::MomentOnly)?,
        })
    }
}

impl Distribution for ExtremeValue {
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
        let gumbel = RandGumbel::new(self.location, self.scale).expect("positive scale");
        DVector::from_element(1, gumbel.sample(rng))
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

fn require_positive(name: &'static str, value: f64) -> Result<()> {
    if value <= 0.0 {
        Err(Error::NonPositiveParameter { name, value })
    } else {
        Ok(())
    }
}

fn univariate_moments(
    mean: f64,
    variance: f64,
    family: PolynomialFamily,
) -> Result<MomentDistribution> {
    MomentDistribution::new(
        DVector::from_element(1, mean),
        DMatrix::from_element(1, 1, variance),
        vec![family],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn additional_univariate_distributions_have_expected_moments() {
        assert_scalar_moments(&Gamma::univariate(2.0, 3.0).unwrap());
        assert_scalar_moments(&ChiSquared::univariate(5.0).unwrap());
        assert_scalar_moments(&LogNormal::univariate(0.2, 0.4).unwrap());
        assert_scalar_moments(&Weibull::univariate(2.0, 3.0).unwrap());
        assert_scalar_moments(&Beta::univariate(2.0, 5.0).unwrap());
        assert_scalar_moments(
            &PiecewiseConstant::univariate(vec![0.0, 1.0, 3.0], vec![1.0, 0.5]).unwrap(),
        );
        assert_scalar_moments(&StudentT::univariate(5.0).unwrap());
        assert_scalar_moments(&FisherF::univariate(6.0, 10.0).unwrap());
        assert_scalar_moments(&ExtremeValue::univariate(1.0, 2.0).unwrap());
    }

    #[test]
    fn finite_moment_distributions_sample_scalars() {
        let mut rng = StdRng::seed_from_u64(7);
        assert_scalar_sample(&Gamma::univariate(2.0, 3.0).unwrap(), &mut rng);
        assert_scalar_sample(&ChiSquared::univariate(5.0).unwrap(), &mut rng);
        assert_scalar_sample(&LogNormal::univariate(0.2, 0.4).unwrap(), &mut rng);
        assert_scalar_sample(&Weibull::univariate(2.0, 3.0).unwrap(), &mut rng);
        assert_scalar_sample(&Beta::univariate(2.0, 5.0).unwrap(), &mut rng);
        assert_scalar_sample(
            &PiecewiseConstant::univariate(vec![0.0, 1.0, 3.0], vec![1.0, 0.5]).unwrap(),
            &mut rng,
        );
        assert_scalar_sample(&StudentT::univariate(5.0).unwrap(), &mut rng);
        assert_scalar_sample(&FisherF::univariate(6.0, 10.0).unwrap(), &mut rng);
        assert_scalar_sample(&ExtremeValue::univariate(1.0, 2.0).unwrap(), &mut rng);
    }

    #[test]
    fn student_t_and_fisher_f_reject_infinite_variance_parameters() {
        assert!(StudentT::univariate(2.0).is_err());
        assert!(FisherF::univariate(4.0, 4.0).is_err());
    }

    #[test]
    fn piecewise_constant_normalizes_interval_weights() {
        let dist = PiecewiseConstant::univariate(vec![0.0, 1.0, 3.0], vec![1.0, 0.5]).unwrap();
        assert!((dist.mean()[0] - 1.25).abs() < 1e-12);
        assert!((dist.covariance()[(0, 0)] - 37.0 / 48.0).abs() < 1e-12);

        let mut rng = StdRng::seed_from_u64(21);
        for _ in 0..32 {
            let sample = dist.sample(&mut rng);
            assert!((0.0..=3.0).contains(&sample[0]));
        }
    }

    fn assert_scalar_moments<D: Distribution>(dist: &D) {
        assert_eq!(dist.dimension(), 1);
        assert!(dist.mean()[0].is_finite());
        assert!(dist.covariance()[(0, 0)] > 0.0);
        assert_eq!(dist.polynomial_family(), &[PolynomialFamily::MomentOnly]);
    }

    fn assert_scalar_sample<D: Distribution, R: Rng + ?Sized>(dist: &D, rng: &mut R) {
        let sample = dist.sample(rng);
        assert_eq!(sample.len(), 1);
        assert!(sample[0].is_finite());
    }
}
