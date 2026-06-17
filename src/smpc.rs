use nalgebra::{DMatrix, DVector, linalg::SymmetricEigen};

use crate::constraints::ChanceConstraintApproximation;
use crate::error::{Error, Result, dim_error};
use crate::problem::{Dynamics, GrampcLikeParams, OptimalControlProblem, ProblemDimensions};
use crate::transformation::PointTransformation;

#[derive(Debug, Clone)]
pub struct MomentState {
    pub mean: DVector<f64>,
    pub covariance: DMatrix<f64>,
}

impl MomentState {
    pub fn new(mean: DVector<f64>, covariance: DMatrix<f64>) -> Result<Self> {
        if covariance.nrows() != mean.len() || covariance.ncols() != mean.len() {
            return Err(dim_error(
                "moment state covariance",
                format!("{}x{}", mean.len(), mean.len()),
                format!("{}x{}", covariance.nrows(), covariance.ncols()),
            ));
        }
        Ok(Self { mean, covariance })
    }

    pub fn pack(&self) -> DVector<f64> {
        pack_moment_state(&self.mean, &self.covariance)
    }

    pub fn unpack(state: &DVector<f64>, state_dim: usize) -> Result<Self> {
        unpack_moment_state(state, state_dim)
    }
}

#[derive(Debug, Clone)]
pub struct StochasticMpcProblem<P, TD, TC> {
    base_problem: P,
    dynamics_transform: TD,
    constraint_transform: TC,
    chance_constraints: Option<ChanceConstraintApproximation>,
    covariance_trace_weight: f64,
    covariance_jitter: f64,
}

impl<P, TD, TC> StochasticMpcProblem<P, TD, TC>
where
    P: OptimalControlProblem,
    TD: PointTransformation,
    TC: PointTransformation,
{
    pub fn new(base_problem: P, dynamics_transform: TD, constraint_transform: TC) -> Result<Self> {
        validate_transforms(&base_problem, &dynamics_transform, &constraint_transform)?;
        Ok(Self {
            base_problem,
            dynamics_transform,
            constraint_transform,
            chance_constraints: None,
            covariance_trace_weight: 0.0,
            covariance_jitter: 0.0,
        })
    }

    pub fn with_chance_constraints(
        mut self,
        chance_constraints: ChanceConstraintApproximation,
    ) -> Result<Self> {
        let inequalities = self.base_problem.dimensions().inequalities;
        if chance_constraints.tightening_coefficients().len() != inequalities {
            return Err(dim_error(
                "chance constraint coefficients",
                inequalities.to_string(),
                chance_constraints
                    .tightening_coefficients()
                    .len()
                    .to_string(),
            ));
        }
        self.chance_constraints = Some(chance_constraints);
        Ok(self)
    }

    pub fn with_covariance_trace_weight(mut self, covariance_trace_weight: f64) -> Result<Self> {
        if covariance_trace_weight < 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "covariance_trace_weight",
                value: covariance_trace_weight,
            });
        }
        self.covariance_trace_weight = covariance_trace_weight;
        Ok(self)
    }

    pub fn with_covariance_jitter(mut self, covariance_jitter: f64) -> Result<Self> {
        if covariance_jitter < 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "covariance_jitter",
                value: covariance_jitter,
            });
        }
        self.covariance_jitter = covariance_jitter;
        Ok(self)
    }

    pub fn base_problem(&self) -> &P {
        &self.base_problem
    }

    pub fn initial_state(
        &self,
        mean: DVector<f64>,
        covariance: DMatrix<f64>,
    ) -> Result<DVector<f64>> {
        let state_dim = self.base_problem.dimensions().states;
        if mean.len() != state_dim {
            return Err(dim_error(
                "stochastic mpc initial mean",
                state_dim.to_string(),
                mean.len().to_string(),
            ));
        }
        MomentState::new(mean, covariance).map(|state| state.pack())
    }

    pub fn unpack_state(&self, state: &DVector<f64>) -> Result<MomentState> {
        unpack_moment_state(state, self.base_problem.dimensions().states)
    }
}

impl<P, TD, TC> Dynamics for StochasticMpcProblem<P, TD, TC>
where
    P: OptimalControlProblem,
    TD: PointTransformation,
    TC: PointTransformation,
{
    fn dimensions(&self) -> ProblemDimensions {
        let base = self.base_problem.dimensions();
        ProblemDimensions::new(
            base.states + base.states * base.states,
            base.controls,
            base.parameters,
            base.equalities,
            base.inequalities,
        )
    }

    fn dynamics(
        &self,
        t: f64,
        x: &DVector<f64>,
        u: &DVector<f64>,
        p: &DVector<f64>,
        params: &GrampcLikeParams,
    ) -> DVector<f64> {
        let state_dim = self.base_problem.dimensions().states;
        let moment_state = unpack_moment_state(x, state_dim).expect("validated stochastic state");
        let cov_sqrt = covariance_square_root(&moment_state.covariance, self.covariance_jitter);
        let state_points = self
            .dynamics_transform
            .points_from_moments(&moment_state.mean, &cov_sqrt)
            .expect("validated dynamics transform");
        let dynamics_points = evaluate_base_dynamics(
            &self.base_problem,
            t,
            &state_points,
            u,
            p,
            params,
            state_dim,
        );
        let mean_derivative = self
            .dynamics_transform
            .mean(&dynamics_points)
            .expect("validated dynamics points");
        let cross_covariance = self
            .dynamics_transform
            .covariance(&state_points, &dynamics_points)
            .expect("validated dynamics covariance");
        let covariance_derivative = &cross_covariance + cross_covariance.transpose();
        pack_moment_state(&mean_derivative, &covariance_derivative)
    }
}

impl<P, TD, TC> OptimalControlProblem for StochasticMpcProblem<P, TD, TC>
where
    P: OptimalControlProblem,
    TD: PointTransformation,
    TC: PointTransformation,
{
    fn stage_cost(
        &self,
        t: f64,
        x: &DVector<f64>,
        u: &DVector<f64>,
        p: &DVector<f64>,
        params: &GrampcLikeParams,
    ) -> f64 {
        let state_dim = self.base_problem.dimensions().states;
        let moment_state = unpack_moment_state(x, state_dim).expect("validated stochastic state");
        self.base_problem
            .stage_cost(t, &moment_state.mean, u, p, params)
            + self.covariance_trace_weight * moment_state.covariance.trace()
    }

    fn terminal_cost(
        &self,
        t: f64,
        x: &DVector<f64>,
        p: &DVector<f64>,
        params: &GrampcLikeParams,
    ) -> f64 {
        let state_dim = self.base_problem.dimensions().states;
        let moment_state = unpack_moment_state(x, state_dim).expect("validated stochastic state");
        self.base_problem
            .terminal_cost(t, &moment_state.mean, p, params)
            + self.covariance_trace_weight * moment_state.covariance.trace()
    }

    fn inequality_constraints(
        &self,
        t: f64,
        x: &DVector<f64>,
        u: &DVector<f64>,
        p: &DVector<f64>,
        params: &GrampcLikeParams,
    ) -> DVector<f64> {
        let base_dims = self.base_problem.dimensions();
        if base_dims.inequalities == 0 {
            return DVector::zeros(0);
        }

        let moment_state =
            unpack_moment_state(x, base_dims.states).expect("validated stochastic state");
        if self.chance_constraints.is_none() {
            return self
                .base_problem
                .inequality_constraints(t, &moment_state.mean, u, p, params);
        }

        let cov_sqrt = covariance_square_root(&moment_state.covariance, self.covariance_jitter);
        let state_points = self
            .constraint_transform
            .points_from_moments(&moment_state.mean, &cov_sqrt)
            .expect("validated constraint transform");
        let mut constraint_points = DMatrix::zeros(
            base_dims.inequalities,
            self.constraint_transform.number_of_points(),
        );
        for point_idx in 0..state_points.ncols() {
            let constraints = self.base_problem.inequality_constraints(
                t,
                &state_points.column(point_idx).into_owned(),
                u,
                p,
                params,
            );
            constraint_points.set_column(point_idx, &constraints);
        }

        let constraint_mean = self
            .constraint_transform
            .mean(&constraint_points)
            .expect("validated constraint points");
        let chance = self
            .chance_constraints
            .as_ref()
            .expect("checked chance constraints");
        DVector::from_iterator(
            base_dims.inequalities,
            (0..base_dims.inequalities).map(|idx| {
                let row = constraint_points.row(idx).transpose();
                let std_dev = self
                    .constraint_transform
                    .variance(&row)
                    .expect("validated constraint variance")
                    .max(0.0)
                    .sqrt();
                chance.tighten_upper_bound(constraint_mean[idx], std_dev, idx)
            }),
        )
    }
}

fn validate_transforms<P, TD, TC>(
    problem: &P,
    dynamics_transform: &TD,
    constraint_transform: &TC,
) -> Result<()>
where
    P: OptimalControlProblem,
    TD: PointTransformation,
    TC: PointTransformation,
{
    let dims = problem.dimensions();
    if dynamics_transform.input_dimension() != dims.states
        || dynamics_transform.output_dimension() != dims.states
    {
        return Err(dim_error(
            "stochastic dynamics transform",
            format!("{}/{} input/output", dims.states, dims.states),
            format!(
                "{}/{} input/output",
                dynamics_transform.input_dimension(),
                dynamics_transform.output_dimension()
            ),
        ));
    }
    if constraint_transform.input_dimension() != dims.states
        || constraint_transform.output_dimension() != dims.inequalities
    {
        return Err(dim_error(
            "stochastic constraint transform",
            format!("{}/{} input/output", dims.states, dims.inequalities),
            format!(
                "{}/{} input/output",
                constraint_transform.input_dimension(),
                constraint_transform.output_dimension()
            ),
        ));
    }
    Ok(())
}

fn evaluate_base_dynamics<P: Dynamics>(
    problem: &P,
    t: f64,
    state_points: &DMatrix<f64>,
    u: &DVector<f64>,
    p: &DVector<f64>,
    params: &GrampcLikeParams,
    state_dim: usize,
) -> DMatrix<f64> {
    let mut dynamics_points = DMatrix::zeros(state_dim, state_points.ncols());
    for point_idx in 0..state_points.ncols() {
        let value = problem.dynamics(
            t,
            &state_points.column(point_idx).into_owned(),
            u,
            p,
            params,
        );
        dynamics_points.set_column(point_idx, &value);
    }
    dynamics_points
}

fn pack_moment_state(mean: &DVector<f64>, covariance: &DMatrix<f64>) -> DVector<f64> {
    let mut state = DVector::zeros(mean.len() + covariance.len());
    state.rows_mut(0, mean.len()).copy_from(mean);
    for (idx, value) in covariance.as_slice().iter().copied().enumerate() {
        state[mean.len() + idx] = value;
    }
    state
}

fn unpack_moment_state(state: &DVector<f64>, state_dim: usize) -> Result<MomentState> {
    let expected = state_dim + state_dim * state_dim;
    if state.len() != expected {
        return Err(dim_error(
            "moment state",
            expected.to_string(),
            state.len().to_string(),
        ));
    }
    let mean = state.rows(0, state_dim).into_owned();
    let covariance = DMatrix::from_column_slice(
        state_dim,
        state_dim,
        &state.as_slice()[state_dim..state_dim + state_dim * state_dim],
    );
    MomentState::new(mean, symmetrize(&covariance))
}

fn covariance_square_root(covariance: &DMatrix<f64>, jitter: f64) -> DMatrix<f64> {
    let symmetric = symmetrize(covariance);
    let eig = SymmetricEigen::new(symmetric);
    let sqrt_diag = DMatrix::from_diagonal(
        &eig.eigenvalues
            .map(|value| (value + jitter).max(0.0).sqrt()),
    );
    &eig.eigenvectors * sqrt_diag
}

fn symmetrize(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    0.5 * (matrix + matrix.transpose())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::ChanceConstraintApproximation;
    use crate::problem::{DoubleIntegrator, GrampcLikeParams};
    use crate::transformation::UnscentedTransformation;

    fn stochastic_problem()
    -> StochasticMpcProblem<DoubleIntegrator, UnscentedTransformation, UnscentedTransformation>
    {
        StochasticMpcProblem::new(
            DoubleIntegrator::new([0.1, 1.0, 0.5, 10.0, 5.0, 0.0], 1.5),
            UnscentedTransformation::new(2, 2, 1.0, 2.0, 0.0).unwrap(),
            UnscentedTransformation::new(2, 1, 1.0, 2.0, 0.0).unwrap(),
        )
        .unwrap()
        .with_chance_constraints(
            ChanceConstraintApproximation::gaussian(DVector::from_vec(vec![0.95])).unwrap(),
        )
        .unwrap()
        .with_covariance_trace_weight(0.2)
        .unwrap()
    }

    #[test]
    fn moment_state_round_trips() {
        let state = MomentState::new(
            DVector::from_vec(vec![1.0, 2.0]),
            DMatrix::from_row_slice(2, 2, &[1.0, 0.2, 0.2, 0.5]),
        )
        .unwrap();
        let packed = state.pack();
        let unpacked = MomentState::unpack(&packed, 2).unwrap();
        assert!((unpacked.mean - state.mean).amax() < 1e-12);
        assert!((unpacked.covariance - state.covariance).amax() < 1e-12);
    }

    #[test]
    fn stochastic_problem_propagates_moment_derivatives() {
        let problem = stochastic_problem();
        let params = GrampcLikeParams::new(
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![0.0]),
        );
        let x = problem
            .initial_state(
                DVector::from_vec(vec![1.0, 2.0]),
                DMatrix::from_diagonal(&DVector::from_vec(vec![0.25, 0.04])),
            )
            .unwrap();
        let dx = problem.dynamics(
            0.0,
            &x,
            &DVector::from_vec(vec![3.0]),
            &DVector::zeros(0),
            &params,
        );
        let derivative = problem.unpack_state(&dx).unwrap();
        assert!((derivative.mean - DVector::from_vec(vec![2.0, 3.0])).amax() < 1e-12);
        assert!(derivative.covariance[(0, 0)].abs() < 1e-12);
        assert!((derivative.covariance[(0, 1)] - 0.04).abs() < 1e-12);
        assert!((derivative.covariance[(1, 0)] - 0.04).abs() < 1e-12);
    }

    #[test]
    fn stochastic_problem_tightens_chance_constraints() {
        let problem = stochastic_problem();
        let params = GrampcLikeParams::new(
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![0.0]),
        );
        let x = problem
            .initial_state(
                DVector::from_vec(vec![0.0, 1.8]),
                DMatrix::from_diagonal(&DVector::from_vec(vec![0.01, 0.01])),
            )
            .unwrap();
        let tightened = problem.inequality_constraints(
            0.0,
            &x,
            &DVector::from_vec(vec![0.0]),
            &DVector::zeros(0),
            &params,
        );
        assert!((tightened[0] - (-0.3 + 0.164_485_362_695_147_24)).abs() < 1e-12);
    }
}
