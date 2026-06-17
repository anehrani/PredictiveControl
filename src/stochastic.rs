use nalgebra::{DMatrix, DVector};
use rand::Rng;

use crate::distribution::Distribution;
use crate::error::{Error, Result, dim_error};
use crate::gaussian_process::{GaussianProcess, StationaryKernel};
use crate::problem::{Dynamics, GrampcLikeParams};
use crate::transformation::{MonteCarloTransformation, PointTransformation};

#[derive(Debug, Clone)]
pub struct DynamicsMoments {
    pub mean: DVector<f64>,
    pub covariance: DMatrix<f64>,
}

impl DynamicsMoments {
    pub fn new(mean: DVector<f64>, covariance: DMatrix<f64>) -> Result<Self> {
        if covariance.nrows() != mean.len() || covariance.ncols() != mean.len() {
            return Err(dim_error(
                "dynamics moments",
                format!("{}x{}", mean.len(), mean.len()),
                format!("{}x{}", covariance.nrows(), covariance.ncols()),
            ));
        }
        Ok(Self { mean, covariance })
    }
}

pub trait ScalarResidualModel {
    fn residual_mean(&self, state: &DVector<f64>, control: &DVector<f64>) -> Result<f64>;
    fn residual_variance(&self, state: &DVector<f64>, control: &DVector<f64>) -> Result<f64>;
}

impl<K: StationaryKernel> ScalarResidualModel for GaussianProcess<K> {
    fn residual_mean(&self, state: &DVector<f64>, control: &DVector<f64>) -> Result<f64> {
        self.mean(state, control)
    }

    fn residual_variance(&self, state: &DVector<f64>, control: &DVector<f64>) -> Result<f64> {
        self.variance(state, control)
    }
}

#[derive(Debug, Clone)]
pub struct SigmaPointDynamics<T> {
    transformation: T,
}

impl<T: PointTransformation> SigmaPointDynamics<T> {
    pub fn new(transformation: T) -> Self {
        Self { transformation }
    }

    pub fn transformation(&self) -> &T {
        &self.transformation
    }

    pub fn approximate<P, D>(
        &self,
        problem: &P,
        t: f64,
        state_distribution: &D,
        u: &DVector<f64>,
        p: &DVector<f64>,
        params: &GrampcLikeParams,
    ) -> Result<DynamicsMoments>
    where
        P: Dynamics,
        D: Distribution,
    {
        let state_dim = problem.dimensions().states;
        require_state_transform(
            &self.transformation,
            state_distribution.dimension(),
            state_dim,
        )?;
        let state_points = self
            .transformation
            .points_from_distribution(state_distribution)?;
        let dynamics_points = evaluate_dynamics_points(problem, t, &state_points, u, p, params)?;
        DynamicsMoments::new(
            self.transformation.mean(&dynamics_points)?,
            self.transformation
                .covariance(&dynamics_points, &dynamics_points)?,
        )
    }
}

#[derive(Debug, Clone)]
pub struct ResamplingDynamics<T> {
    transformation: T,
    process_noise: DMatrix<f64>,
}

impl<T: PointTransformation> ResamplingDynamics<T> {
    pub fn new(transformation: T, state_dim: usize) -> Self {
        Self {
            transformation,
            process_noise: DMatrix::zeros(state_dim, state_dim),
        }
    }

    pub fn with_process_noise(transformation: T, process_noise: DMatrix<f64>) -> Result<Self> {
        if process_noise.nrows() != process_noise.ncols() {
            return Err(dim_error(
                "resampling process noise",
                "square matrix",
                format!("{}x{}", process_noise.nrows(), process_noise.ncols()),
            ));
        }
        Ok(Self {
            transformation,
            process_noise,
        })
    }

    pub fn transformation(&self) -> &T {
        &self.transformation
    }

    pub fn process_noise(&self) -> &DMatrix<f64> {
        &self.process_noise
    }

    pub fn derivative<P, D>(
        &self,
        problem: &P,
        t: f64,
        state_distribution: &D,
        u: &DVector<f64>,
        p: &DVector<f64>,
        params: &GrampcLikeParams,
    ) -> Result<DynamicsMoments>
    where
        P: Dynamics,
        D: Distribution,
    {
        let state_dim = problem.dimensions().states;
        require_state_transform(
            &self.transformation,
            state_distribution.dimension(),
            state_dim,
        )?;
        require_square_dim("resampling process noise", &self.process_noise, state_dim)?;
        let state_points = self
            .transformation
            .points_from_distribution(state_distribution)?;
        let dynamics_points = evaluate_dynamics_points(problem, t, &state_points, u, p, params)?;
        let mean_derivative = self.transformation.mean(&dynamics_points)?;
        let cross_covariance = self
            .transformation
            .covariance(&state_points, &dynamics_points)?;
        let covariance_derivative =
            &cross_covariance + cross_covariance.transpose() + &self.process_noise;
        DynamicsMoments::new(mean_derivative, covariance_derivative)
    }
}

#[derive(Debug, Clone)]
pub struct ResamplingGpDynamics<T, G> {
    transformation: T,
    residual_models: Vec<G>,
    residual_indices: Vec<usize>,
}

impl<T: PointTransformation, G: ScalarResidualModel> ResamplingGpDynamics<T, G> {
    pub fn new(
        transformation: T,
        residual_models: Vec<G>,
        residual_indices: Vec<usize>,
    ) -> Result<Self> {
        if residual_models.len() != residual_indices.len() {
            return Err(dim_error(
                "resampling gp residuals",
                format!("{} residual indices", residual_models.len()),
                format!("{} residual indices", residual_indices.len()),
            ));
        }
        Ok(Self {
            transformation,
            residual_models,
            residual_indices,
        })
    }

    pub fn derivative<P, D>(
        &self,
        problem: &P,
        t: f64,
        state_distribution: &D,
        u: &DVector<f64>,
        p: &DVector<f64>,
        params: &GrampcLikeParams,
    ) -> Result<DynamicsMoments>
    where
        P: Dynamics,
        D: Distribution,
    {
        let state_dim = problem.dimensions().states;
        require_state_transform(
            &self.transformation,
            state_distribution.dimension(),
            state_dim,
        )?;
        for &index in &self.residual_indices {
            if index >= state_dim {
                return Err(dim_error(
                    "resampling gp residual index",
                    format!("0..{state_dim}"),
                    index.to_string(),
                ));
            }
        }

        let state_points = self
            .transformation
            .points_from_distribution(state_distribution)?;
        let mut dynamics_points =
            evaluate_dynamics_points(problem, t, &state_points, u, p, params)?;
        for point_idx in 0..state_points.ncols() {
            let state_point = state_points.column(point_idx).into_owned();
            for (residual_model, &state_index) in self
                .residual_models
                .iter()
                .zip(self.residual_indices.iter())
            {
                dynamics_points[(state_index, point_idx)] +=
                    residual_model.residual_mean(&state_point, u)?;
            }
        }

        let mean_derivative = self.transformation.mean(&dynamics_points)?;
        let cross_covariance = self
            .transformation
            .covariance(&state_points, &dynamics_points)?;
        let mut covariance_derivative = &cross_covariance + cross_covariance.transpose();
        for (residual_model, &state_index) in self
            .residual_models
            .iter()
            .zip(self.residual_indices.iter())
        {
            covariance_derivative[(state_index, state_index)] +=
                residual_model.residual_variance(state_distribution.mean(), u)?;
        }
        DynamicsMoments::new(mean_derivative, covariance_derivative)
    }
}

#[derive(Debug, Clone)]
pub struct MonteCarloDynamics {
    number_of_samples: usize,
}

impl MonteCarloDynamics {
    pub fn new(number_of_samples: usize) -> Result<Self> {
        if number_of_samples == 0 {
            return Err(Error::Empty("monte carlo dynamics samples"));
        }
        Ok(Self { number_of_samples })
    }

    pub fn number_of_samples(&self) -> usize {
        self.number_of_samples
    }

    pub fn approximate<P, D, R>(
        &self,
        problem: &P,
        t: f64,
        state_distribution: &D,
        u: &DVector<f64>,
        p: &DVector<f64>,
        params: &GrampcLikeParams,
        rng: &mut R,
    ) -> Result<DynamicsMoments>
    where
        P: Dynamics,
        D: Distribution,
        R: Rng + ?Sized,
    {
        let state_dim = problem.dimensions().states;
        if state_distribution.dimension() != state_dim {
            return Err(dim_error(
                "monte carlo dynamics state",
                state_dim.to_string(),
                state_distribution.dimension().to_string(),
            ));
        }
        let transform =
            MonteCarloTransformation::new(state_dim, state_dim, self.number_of_samples)?;
        let state_points = transform.points_from_distribution(state_distribution, rng)?;
        let dynamics_points = evaluate_dynamics_points(problem, t, &state_points, u, p, params)?;
        DynamicsMoments::new(
            transform.mean(&dynamics_points)?,
            transform.covariance(&dynamics_points, &dynamics_points)?,
        )
    }
}

#[derive(Debug, Clone)]
pub struct TaylorDynamics {
    finite_difference_step: f64,
}

impl TaylorDynamics {
    pub fn new(finite_difference_step: f64) -> Result<Self> {
        if finite_difference_step <= 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "finite_difference_step",
                value: finite_difference_step,
            });
        }
        Ok(Self {
            finite_difference_step,
        })
    }

    pub fn approximate<P, D>(
        &self,
        problem: &P,
        t: f64,
        state_distribution: &D,
        u: &DVector<f64>,
        p: &DVector<f64>,
        params: &GrampcLikeParams,
    ) -> Result<DynamicsMoments>
    where
        P: Dynamics,
        D: Distribution,
    {
        let state_dim = problem.dimensions().states;
        if state_distribution.dimension() != state_dim {
            return Err(dim_error(
                "taylor dynamics state",
                state_dim.to_string(),
                state_distribution.dimension().to_string(),
            ));
        }
        let mean = problem.dynamics(t, state_distribution.mean(), u, p, params);
        if mean.len() != state_dim {
            return Err(dim_error(
                "taylor dynamics output",
                state_dim.to_string(),
                mean.len().to_string(),
            ));
        }
        let jacobian = finite_difference_state_jacobian(
            problem,
            t,
            state_distribution.mean(),
            u,
            p,
            params,
            self.finite_difference_step,
        )?;
        let covariance = &jacobian * state_distribution.covariance() * jacobian.transpose();
        DynamicsMoments::new(mean, covariance)
    }
}

fn require_state_transform<T: PointTransformation>(
    transform: &T,
    distribution_dim: usize,
    state_dim: usize,
) -> Result<()> {
    if distribution_dim != state_dim {
        return Err(dim_error(
            "sigma point dynamics state",
            state_dim.to_string(),
            distribution_dim.to_string(),
        ));
    }
    if transform.input_dimension() != state_dim || transform.output_dimension() != state_dim {
        return Err(dim_error(
            "sigma point dynamics transform",
            format!("{state_dim} input/output dimensions"),
            format!(
                "{}/{} input/output dimensions",
                transform.input_dimension(),
                transform.output_dimension()
            ),
        ));
    }
    Ok(())
}

fn require_square_dim(context: &'static str, matrix: &DMatrix<f64>, dim: usize) -> Result<()> {
    if matrix.nrows() != dim || matrix.ncols() != dim {
        return Err(dim_error(
            context,
            format!("{dim}x{dim}"),
            format!("{}x{}", matrix.nrows(), matrix.ncols()),
        ));
    }
    Ok(())
}

fn evaluate_dynamics_points<P: Dynamics>(
    problem: &P,
    t: f64,
    state_points: &DMatrix<f64>,
    u: &DVector<f64>,
    p: &DVector<f64>,
    params: &GrampcLikeParams,
) -> Result<DMatrix<f64>> {
    let state_dim = problem.dimensions().states;
    let mut dynamics_points = DMatrix::zeros(state_dim, state_points.ncols());
    for point_idx in 0..state_points.ncols() {
        let value = problem.dynamics(
            t,
            &state_points.column(point_idx).into_owned(),
            u,
            p,
            params,
        );
        if value.len() != state_dim {
            return Err(dim_error(
                "dynamics output",
                state_dim.to_string(),
                value.len().to_string(),
            ));
        }
        dynamics_points.set_column(point_idx, &value);
    }
    Ok(dynamics_points)
}

fn finite_difference_state_jacobian<P: Dynamics>(
    problem: &P,
    t: f64,
    x: &DVector<f64>,
    u: &DVector<f64>,
    p: &DVector<f64>,
    params: &GrampcLikeParams,
    step: f64,
) -> Result<DMatrix<f64>> {
    let state_dim = problem.dimensions().states;
    let mut jacobian = DMatrix::zeros(state_dim, state_dim);
    for state_idx in 0..state_dim {
        let mut plus = x.clone();
        let mut minus = x.clone();
        plus[state_idx] += step;
        minus[state_idx] -= step;
        let f_plus = problem.dynamics(t, &plus, u, p, params);
        let f_minus = problem.dynamics(t, &minus, u, p, params);
        if f_plus.len() != state_dim || f_minus.len() != state_dim {
            return Err(dim_error(
                "finite difference dynamics output",
                state_dim.to_string(),
                format!("{}/{}", f_plus.len(), f_minus.len()),
            ));
        }
        jacobian.set_column(state_idx, &((f_plus - f_minus) / (2.0 * step)));
    }
    Ok(jacobian)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::Gaussian;
    use crate::problem::{DoubleIntegrator, GrampcLikeParams};
    use crate::transformation::UnscentedTransformation;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[derive(Debug, Clone)]
    struct ConstantResidual {
        mean: f64,
        variance: f64,
    }

    impl ScalarResidualModel for ConstantResidual {
        fn residual_mean(&self, _state: &DVector<f64>, _control: &DVector<f64>) -> Result<f64> {
            Ok(self.mean)
        }

        fn residual_variance(&self, _state: &DVector<f64>, _control: &DVector<f64>) -> Result<f64> {
            Ok(self.variance)
        }
    }

    fn fixture() -> (
        DoubleIntegrator,
        Gaussian,
        DVector<f64>,
        DVector<f64>,
        GrampcLikeParams,
    ) {
        (
            DoubleIntegrator::new([1.0; 6], 0.0),
            Gaussian::new(
                DVector::from_vec(vec![1.0, 2.0]),
                DMatrix::from_diagonal(&DVector::from_vec(vec![0.25, 0.04])),
            )
            .unwrap(),
            DVector::from_vec(vec![3.0]),
            DVector::zeros(0),
            GrampcLikeParams::new(
                DVector::from_vec(vec![0.0, 0.0]),
                DVector::from_vec(vec![0.0]),
            ),
        )
    }

    #[test]
    fn sigma_point_dynamics_propagates_linear_moments() {
        let (problem, state, u, p, params) = fixture();
        let transform = UnscentedTransformation::new(2, 2, 1.0, 2.0, 0.0).unwrap();
        let approximation = SigmaPointDynamics::new(transform)
            .approximate(&problem, 0.0, &state, &u, &p, &params)
            .unwrap();

        assert!((approximation.mean - DVector::from_vec(vec![2.0, 3.0])).amax() < 1e-12);
        assert!((approximation.covariance[(0, 0)] - 0.04).abs() < 1e-12);
        assert!(approximation.covariance[(1, 1)].abs() < 1e-12);
    }

    #[test]
    fn resampling_dynamics_computes_mean_and_covariance_derivatives() {
        let (problem, state, u, p, params) = fixture();
        let transform = UnscentedTransformation::new(2, 2, 1.0, 2.0, 0.0).unwrap();
        let approximation = ResamplingDynamics::with_process_noise(
            transform,
            DMatrix::from_diagonal(&DVector::from_vec(vec![0.01, 0.02])),
        )
        .unwrap()
        .derivative(&problem, 0.0, &state, &u, &p, &params)
        .unwrap();

        assert!((approximation.mean - DVector::from_vec(vec![2.0, 3.0])).amax() < 1e-12);
        assert!((approximation.covariance[(0, 0)] - 0.01).abs() < 1e-12);
        assert!((approximation.covariance[(0, 1)] - 0.04).abs() < 1e-12);
        assert!((approximation.covariance[(1, 0)] - 0.04).abs() < 1e-12);
        assert!((approximation.covariance[(1, 1)] - 0.02).abs() < 1e-12);
    }

    #[test]
    fn resampling_gp_dynamics_adds_residual_mean_and_variance() {
        let (problem, state, u, p, params) = fixture();
        let transform = UnscentedTransformation::new(2, 2, 1.0, 2.0, 0.0).unwrap();
        let approximation = ResamplingGpDynamics::new(
            transform,
            vec![ConstantResidual {
                mean: 0.5,
                variance: 0.25,
            }],
            vec![1],
        )
        .unwrap()
        .derivative(&problem, 0.0, &state, &u, &p, &params)
        .unwrap();

        assert!((approximation.mean - DVector::from_vec(vec![2.0, 3.5])).amax() < 1e-12);
        assert!((approximation.covariance[(0, 1)] - 0.04).abs() < 1e-12);
        assert!((approximation.covariance[(1, 1)] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn taylor_dynamics_propagates_linear_moments() {
        let (problem, state, u, p, params) = fixture();
        let approximation = TaylorDynamics::new(1e-6)
            .unwrap()
            .approximate(&problem, 0.0, &state, &u, &p, &params)
            .unwrap();

        assert!((approximation.mean - DVector::from_vec(vec![2.0, 3.0])).amax() < 1e-12);
        assert!((approximation.covariance[(0, 0)] - 0.04).abs() < 1e-10);
        assert!(approximation.covariance[(1, 1)].abs() < 1e-10);
    }

    #[test]
    fn monte_carlo_dynamics_samples_finite_moments() {
        let (problem, state, u, p, params) = fixture();
        let mut rng = StdRng::seed_from_u64(42);
        let approximation = MonteCarloDynamics::new(256)
            .unwrap()
            .approximate(&problem, 0.0, &state, &u, &p, &params, &mut rng)
            .unwrap();

        assert_eq!(approximation.mean.len(), 2);
        assert_eq!(approximation.covariance.shape(), (2, 2));
        assert!(approximation.mean.iter().all(|value| value.is_finite()));
        assert!(
            approximation
                .covariance
                .iter()
                .all(|value| value.is_finite())
        );
    }
}
