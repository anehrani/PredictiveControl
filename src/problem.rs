use nalgebra::DVector;

use crate::error::{Error, Result, dim_error};

pub struct ProblemDimensions {
    pub states: usize,
    pub controls: usize,
    pub parameters: usize,
    pub equalities: usize,
    pub inequalities: usize,
}

impl ProblemDimensions {
    pub fn new(
        states: usize,
        controls: usize,
        parameters: usize,
        equalities: usize,
        inequalities: usize,
    ) -> Self {
        Self {
            states,
            controls,
            parameters,
            equalities,
            inequalities,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GrampcLikeParams {
    pub x_des: DVector<f64>,
    pub u_des: DVector<f64>,
}

impl GrampcLikeParams {
    pub fn new(x_des: DVector<f64>, u_des: DVector<f64>) -> Self {
        Self { x_des, u_des }
    }
}

pub trait Dynamics {
    fn dimensions(&self) -> ProblemDimensions;

    fn dynamics(
        &self,
        t: f64,
        x: &DVector<f64>,
        u: &DVector<f64>,
        p: &DVector<f64>,
        params: &GrampcLikeParams,
    ) -> DVector<f64>;
}

pub trait OptimalControlProblem: Dynamics {
    fn stage_cost(
        &self,
        _t: f64,
        _x: &DVector<f64>,
        _u: &DVector<f64>,
        _p: &DVector<f64>,
        _params: &GrampcLikeParams,
    ) -> f64 {
        0.0
    }

    fn terminal_cost(
        &self,
        _t: f64,
        _x: &DVector<f64>,
        _p: &DVector<f64>,
        _params: &GrampcLikeParams,
    ) -> f64 {
        0.0
    }

    fn inequality_constraints(
        &self,
        _t: f64,
        _x: &DVector<f64>,
        _u: &DVector<f64>,
        _p: &DVector<f64>,
        _params: &GrampcLikeParams,
    ) -> DVector<f64> {
        DVector::zeros(self.dimensions().inequalities)
    }
}

pub struct DoubleIntegrator {
    pub cost_weights: [f64; 6],
    pub constraint_offset: f64,
}

impl DoubleIntegrator {
    pub fn new(cost_weights: [f64; 6], constraint_offset: f64) -> Self {
        Self {
            cost_weights,
            constraint_offset,
        }
    }
}

impl Dynamics for DoubleIntegrator {
    fn dimensions(&self) -> ProblemDimensions {
        ProblemDimensions::new(2, 1, 0, 0, 1)
    }

    fn dynamics(
        &self,
        _t: f64,
        x: &DVector<f64>,
        u: &DVector<f64>,
        _p: &DVector<f64>,
        _params: &GrampcLikeParams,
    ) -> DVector<f64> {
        DVector::from_vec(vec![x[1], u[0]])
    }
}

impl OptimalControlProblem for DoubleIntegrator {
    fn stage_cost(
        &self,
        _t: f64,
        x: &DVector<f64>,
        u: &DVector<f64>,
        _p: &DVector<f64>,
        params: &GrampcLikeParams,
    ) -> f64 {
        self.cost_weights[0] * (u[0] - params.u_des[0]).powi(2)
            + self.cost_weights[1] * (x[0] - params.x_des[0]).powi(2)
            + self.cost_weights[2] * (x[1] - params.x_des[1]).powi(2)
    }

    fn terminal_cost(
        &self,
        t: f64,
        x: &DVector<f64>,
        _p: &DVector<f64>,
        params: &GrampcLikeParams,
    ) -> f64 {
        self.cost_weights[3] * (x[0] - params.x_des[0]).powi(2)
            + self.cost_weights[4] * (x[1] - params.x_des[1]).powi(2)
            + self.cost_weights[5] * t
    }

    fn inequality_constraints(
        &self,
        _t: f64,
        x: &DVector<f64>,
        _u: &DVector<f64>,
        _p: &DVector<f64>,
        _params: &GrampcLikeParams,
    ) -> DVector<f64> {
        DVector::from_element(1, -x[1] + self.constraint_offset)
    }
}

pub struct MassSpringDamper {
    number_of_masses: usize,
    mass: f64,
    stage_state_weights: DVector<f64>,
    stage_control_weights: DVector<f64>,
    terminal_state_weights: DVector<f64>,
}

impl MassSpringDamper {
    pub fn new(number_of_masses: usize, mass: f64) -> Result<Self> {
        if number_of_masses < 2 {
            return Err(dim_error(
                "mass spring damper masses",
                "at least 2",
                number_of_masses.to_string(),
            ));
        }
        if mass <= 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "mass",
                value: mass,
            });
        }
        let states = 2 * number_of_masses;
        Ok(Self {
            number_of_masses,
            mass,
            stage_state_weights: DVector::from_element(states, 1.0),
            stage_control_weights: DVector::from_element(2, 0.01),
            terminal_state_weights: DVector::from_element(states, 1.0),
        })
    }

    pub fn with_weights(
        number_of_masses: usize,
        mass: f64,
        stage_state_weights: DVector<f64>,
        stage_control_weights: DVector<f64>,
        terminal_state_weights: DVector<f64>,
    ) -> Result<Self> {
        let states = 2 * number_of_masses;
        if number_of_masses < 2 {
            return Err(dim_error(
                "mass spring damper masses",
                "at least 2",
                number_of_masses.to_string(),
            ));
        }
        if mass <= 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "mass",
                value: mass,
            });
        }
        if stage_state_weights.len() != states {
            return Err(dim_error(
                "mass spring damper stage state weights",
                states.to_string(),
                stage_state_weights.len().to_string(),
            ));
        }
        if stage_control_weights.len() != 2 {
            return Err(dim_error(
                "mass spring damper stage control weights",
                "2",
                stage_control_weights.len().to_string(),
            ));
        }
        if terminal_state_weights.len() != states {
            return Err(dim_error(
                "mass spring damper terminal weights",
                states.to_string(),
                terminal_state_weights.len().to_string(),
            ));
        }
        Ok(Self {
            number_of_masses,
            mass,
            stage_state_weights,
            stage_control_weights,
            terminal_state_weights,
        })
    }

    pub fn number_of_masses(&self) -> usize {
        self.number_of_masses
    }
}

impl Dynamics for MassSpringDamper {
    fn dimensions(&self) -> ProblemDimensions {
        ProblemDimensions::new(2 * self.number_of_masses, 2, 2, 0, 0)
    }

    fn dynamics(
        &self,
        _t: f64,
        x: &DVector<f64>,
        u: &DVector<f64>,
        p: &DVector<f64>,
        _params: &GrampcLikeParams,
    ) -> DVector<f64> {
        let n = self.number_of_masses;
        let spring = p[0];
        let damping = p[1];
        let inv_mass = 1.0 / self.mass;
        let mut out = DVector::zeros(2 * n);

        for k in 0..n {
            out[k] = x[n + k];
        }

        out[n] = inv_mass
            * (-2.0 * spring * x[0] + spring * x[1] - 2.0 * damping * x[n]
                + damping * x[n + 1]
                + u[0]);

        for k in 1..=(n - 2) {
            out[n + k] = inv_mass
                * (spring * x[k - 1] - 2.0 * spring * x[k]
                    + spring * x[k + 1]
                    + damping * x[n + k - 1]
                    - 2.0 * damping * x[n + k]
                    + damping * x[n + k + 1]);
        }

        out[2 * n - 1] = inv_mass
            * (spring * x[n - 2] - 2.0 * spring * x[n - 1] + damping * x[2 * n - 2]
                - 2.0 * damping * x[2 * n - 1]
                - u[1]);

        out
    }
}

impl OptimalControlProblem for MassSpringDamper {
    fn stage_cost(
        &self,
        _t: f64,
        x: &DVector<f64>,
        u: &DVector<f64>,
        _p: &DVector<f64>,
        params: &GrampcLikeParams,
    ) -> f64 {
        let state_cost: f64 = x
            .iter()
            .zip(params.x_des.iter())
            .zip(self.stage_state_weights.iter())
            .map(|((x_i, x_des_i), weight)| weight * (x_i - x_des_i).powi(2))
            .sum();
        let control_cost: f64 = u
            .iter()
            .zip(params.u_des.iter())
            .zip(self.stage_control_weights.iter())
            .map(|((u_i, u_des_i), weight)| weight * (u_i - u_des_i).powi(2))
            .sum();
        0.5 * (state_cost + control_cost)
    }

    fn terminal_cost(
        &self,
        _t: f64,
        x: &DVector<f64>,
        _p: &DVector<f64>,
        params: &GrampcLikeParams,
    ) -> f64 {
        0.5 * x
            .iter()
            .zip(params.x_des.iter())
            .zip(self.terminal_state_weights.iter())
            .map(|((x_i, x_des_i), weight)| weight * (x_i - x_des_i).powi(2))
            .sum::<f64>()
    }
}

pub struct Reactor {
    system_parameters: [f64; 3],
    cost_weights: [f64; 5],
    concentration_limit: f64,
}

impl Reactor {
    pub fn new(
        system_parameters: [f64; 3],
        cost_weights: [f64; 5],
        concentration_limit: f64,
    ) -> Result<Self> {
        for (index, value) in system_parameters.iter().copied().enumerate() {
            if value <= 0.0 {
                return Err(Error::NonPositiveParameter {
                    name: match index {
                        0 => "reactor reaction parameter a",
                        1 => "reactor reaction parameter b",
                        _ => "reactor reaction parameter c",
                    },
                    value,
                });
            }
        }
        if concentration_limit <= 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "reactor concentration limit",
                value: concentration_limit,
            });
        }
        Ok(Self {
            system_parameters,
            cost_weights,
            concentration_limit,
        })
    }

    pub fn benchmark() -> Self {
        Self {
            system_parameters: [50.0, 100.0, 100.0],
            cost_weights: [0.0, 0.0, 1.0, 1.0, 2.0],
            concentration_limit: 0.14,
        }
    }

    pub fn system_parameters(&self) -> [f64; 3] {
        self.system_parameters
    }

    pub fn cost_weights(&self) -> [f64; 5] {
        self.cost_weights
    }

    pub fn concentration_limit(&self) -> f64 {
        self.concentration_limit
    }
}

impl Dynamics for Reactor {
    fn dimensions(&self) -> ProblemDimensions {
        ProblemDimensions::new(2, 1, 0, 0, 1)
    }

    fn dynamics(
        &self,
        _t: f64,
        x: &DVector<f64>,
        u: &DVector<f64>,
        _p: &DVector<f64>,
        _params: &GrampcLikeParams,
    ) -> DVector<f64> {
        let [reaction_a, reaction_b, reaction_c] = self.system_parameters;
        DVector::from_vec(vec![
            -reaction_a * x[0] - reaction_c * x[0].powi(2) + (1.0 - x[0]) * u[0],
            reaction_a * x[0] - reaction_b * x[1] - x[1] * u[0],
        ])
    }
}

impl OptimalControlProblem for Reactor {
    fn stage_cost(
        &self,
        _t: f64,
        x: &DVector<f64>,
        u: &DVector<f64>,
        _p: &DVector<f64>,
        params: &GrampcLikeParams,
    ) -> f64 {
        self.cost_weights[2] * (x[0] - params.x_des[0]).powi(2)
            + self.cost_weights[3] * (x[1] - params.x_des[1]).powi(2)
            + self.cost_weights[4] * (u[0] - params.u_des[0]).powi(2)
    }

    fn terminal_cost(
        &self,
        _t: f64,
        x: &DVector<f64>,
        _p: &DVector<f64>,
        params: &GrampcLikeParams,
    ) -> f64 {
        self.cost_weights[0] * (x[0] - params.x_des[0]).powi(2)
            + self.cost_weights[1] * (x[1] - params.x_des[1]).powi(2)
    }

    fn inequality_constraints(
        &self,
        _t: f64,
        x: &DVector<f64>,
        _u: &DVector<f64>,
        _p: &DVector<f64>,
        _params: &GrampcLikeParams,
    ) -> DVector<f64> {
        DVector::from_element(1, x[1] - self.concentration_limit)
    }
}

pub struct InvertedPendulum {
    system_parameters: [f64; 5],
    cost_weights: [f64; 9],
    velocity_limit: f64,
}

impl InvertedPendulum {
    pub fn new(
        system_parameters: [f64; 5],
        cost_weights: [f64; 9],
        velocity_limit: f64,
    ) -> Result<Self> {
        for (index, value) in system_parameters.iter().copied().enumerate() {
            if value <= 0.0 {
                return Err(Error::NonPositiveParameter {
                    name: match index {
                        0 => "pendulum friction",
                        1 => "pendulum mass coupling",
                        2 => "pendulum length",
                        3 => "gravity",
                        _ => "cart inertia",
                    },
                    value,
                });
            }
        }
        if velocity_limit <= 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "pendulum velocity limit",
                value: velocity_limit,
            });
        }
        Ok(Self {
            system_parameters,
            cost_weights,
            velocity_limit,
        })
    }

    pub fn benchmark() -> Self {
        Self {
            system_parameters: [2e-3, 0.2, 0.15, 9.81, 1.3e-3],
            cost_weights: [100.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1e-9],
            velocity_limit: 0.8,
        }
    }

    pub fn system_parameters(&self) -> [f64; 5] {
        self.system_parameters
    }

    pub fn cost_weights(&self) -> [f64; 9] {
        self.cost_weights
    }

    pub fn velocity_limit(&self) -> f64 {
        self.velocity_limit
    }

    fn denominator(&self) -> f64 {
        let [_, coupling, length, _, inertia] = self.system_parameters;
        inertia + coupling.powi(2) * length
    }
}

impl Dynamics for InvertedPendulum {
    fn dimensions(&self) -> ProblemDimensions {
        ProblemDimensions::new(4, 1, 0, 0, 2)
    }

    fn dynamics(
        &self,
        _t: f64,
        x: &DVector<f64>,
        u: &DVector<f64>,
        _p: &DVector<f64>,
        _params: &GrampcLikeParams,
    ) -> DVector<f64> {
        let [friction, coupling, length, gravity, _inertia] = self.system_parameters;
        let denominator = self.denominator();
        DVector::from_vec(vec![
            x[1],
            u[0],
            x[3],
            -(friction * x[3]
                + coupling * (length * u[0] * x[2].cos() + length * gravity * x[2].sin()))
                / denominator,
        ])
    }
}

impl OptimalControlProblem for InvertedPendulum {
    fn stage_cost(
        &self,
        _t: f64,
        x: &DVector<f64>,
        u: &DVector<f64>,
        _p: &DVector<f64>,
        params: &GrampcLikeParams,
    ) -> f64 {
        self.cost_weights[8] * (u[0] - params.u_des[0]).powi(2)
            + (0..4)
                .map(|i| self.cost_weights[i] * (x[i] - params.x_des[i]).powi(2))
                .sum::<f64>()
    }

    fn terminal_cost(
        &self,
        _t: f64,
        x: &DVector<f64>,
        _p: &DVector<f64>,
        params: &GrampcLikeParams,
    ) -> f64 {
        (0..4)
            .map(|i| self.cost_weights[4 + i] * (x[i] - params.x_des[i]).powi(2))
            .sum()
    }

    fn inequality_constraints(
        &self,
        _t: f64,
        x: &DVector<f64>,
        _u: &DVector<f64>,
        _p: &DVector<f64>,
        _params: &GrampcLikeParams,
    ) -> DVector<f64> {
        DVector::from_vec(vec![
            x[1] - self.velocity_limit,
            -x[1] - self.velocity_limit,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mass_spring_damper_matches_upstream_boundary_equations() {
        let problem = MassSpringDamper::new(3, 1.0).unwrap();
        let x = DVector::from_vec(vec![1.0, 2.0, 3.0, 0.1, 0.2, 0.3]);
        let u = DVector::from_vec(vec![0.5, -0.25]);
        let p = DVector::from_vec(vec![2.0, 0.4]);
        let params = GrampcLikeParams::new(DVector::zeros(6), DVector::zeros(2));

        let dx = problem.dynamics(0.0, &x, &u, &p, &params);

        assert_eq!(dx.rows(0, 3), x.rows(3, 3));
        assert!(
            (dx[3] - (-2.0 * 2.0 * 1.0 + 2.0 * 2.0 - 2.0 * 0.4 * 0.1 + 0.4 * 0.2 + 0.5)).abs()
                < 1e-12
        );
        assert!(
            (dx[4]
                - (2.0 * 1.0 - 2.0 * 2.0 * 2.0 + 2.0 * 3.0 + 0.4 * 0.1 - 2.0 * 0.4 * 0.2
                    + 0.4 * 0.3))
                .abs()
                < 1e-12
        );
        assert!(
            (dx[5] - (2.0 * 2.0 - 2.0 * 2.0 * 3.0 + 0.4 * 0.2 - 2.0 * 0.4 * 0.3 + 0.25)).abs()
                < 1e-12
        );
    }

    #[test]
    fn reactor_matches_upstream_dynamics_cost_and_constraint() {
        let problem = Reactor::new([50.0, 100.0, 100.0], [3.0, 4.0, 5.0, 6.0, 7.0], 0.14).unwrap();
        let x = DVector::from_vec(vec![0.8, 0.01]);
        let u = DVector::from_vec(vec![19.6]);
        let p = DVector::zeros(0);
        let params = GrampcLikeParams::new(
            DVector::from_vec(vec![0.215, 0.09]),
            DVector::from_vec(vec![19.0]),
        );

        let dx = problem.dynamics(0.0, &x, &u, &p, &params);

        assert!((dx[0] - (-50.0 * 0.8 - 100.0 * 0.8 * 0.8 + (1.0 - 0.8) * 19.6)).abs() < 1e-12);
        assert!((dx[1] - (50.0 * 0.8 - 100.0 * 0.01 - 0.01 * 19.6)).abs() < 1e-12);
        assert!(
            (problem.stage_cost(0.0, &x, &u, &p, &params)
                - (5.0 * (0.8_f64 - 0.215).powi(2)
                    + 6.0 * (0.01_f64 - 0.09).powi(2)
                    + 7.0 * (19.6_f64 - 19.0).powi(2)))
            .abs()
                < 1e-12
        );
        assert!(
            (problem.terminal_cost(0.0, &x, &p, &params)
                - (3.0 * (0.8_f64 - 0.215).powi(2) + 4.0 * (0.01_f64 - 0.09).powi(2)))
            .abs()
                < 1e-12
        );
        assert!(
            (problem.inequality_constraints(0.0, &x, &u, &p, &params)[0] - (0.01 - 0.14)).abs()
                < 1e-12
        );
    }

    #[test]
    fn inverted_pendulum_matches_upstream_dynamics_cost_and_constraints() {
        let problem = InvertedPendulum::new(
            [2e-3, 0.2, 0.15, 9.81, 1.3e-3],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            0.8,
        )
        .unwrap();
        let x = DVector::from_vec(vec![0.5, 0.1, std::f64::consts::PI, -0.2]);
        let u = DVector::from_vec(vec![0.3]);
        let p = DVector::zeros(0);
        let params = GrampcLikeParams::new(
            DVector::from_vec(vec![0.0, 0.0, std::f64::consts::PI, 0.0]),
            DVector::from_vec(vec![0.0]),
        );

        let dx = problem.dynamics(0.0, &x, &u, &p, &params);
        let denominator = 1.3e-3 + 0.2_f64.powi(2) * 0.15;
        let expected_angle_acceleration = -(2e-3 * -0.2
            + 0.2
                * (0.15 * 0.3 * std::f64::consts::PI.cos()
                    + 0.15 * 9.81 * std::f64::consts::PI.sin()))
            / denominator;

        assert!((dx[0] - 0.1).abs() < 1e-12);
        assert!((dx[1] - 0.3).abs() < 1e-12);
        assert!((dx[2] + 0.2).abs() < 1e-12);
        assert!((dx[3] - expected_angle_acceleration).abs() < 1e-12);
        assert!(
            (problem.stage_cost(0.0, &x, &u, &p, &params)
                - (1.0 * 0.5_f64.powi(2)
                    + 2.0 * 0.1_f64.powi(2)
                    + 4.0 * (-0.2_f64).powi(2)
                    + 9.0 * 0.3_f64.powi(2)))
            .abs()
                < 1e-12
        );
        assert!(
            (problem.terminal_cost(0.0, &x, &p, &params)
                - (5.0 * 0.5_f64.powi(2) + 6.0 * 0.1_f64.powi(2) + 8.0 * (-0.2_f64).powi(2)))
            .abs()
                < 1e-12
        );
        let constraints = problem.inequality_constraints(0.0, &x, &u, &p, &params);
        assert!((constraints[0] - (0.1 - 0.8)).abs() < 1e-12);
        assert!((constraints[1] - (-0.1 - 0.8)).abs() < 1e-12);
    }
}
