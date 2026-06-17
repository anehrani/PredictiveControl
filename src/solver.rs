use nalgebra::DVector;

use crate::error::{Error, Result, dim_error};
use crate::problem::{GrampcLikeParams, OptimalControlProblem};
use crate::simulator::{SimulationStep, rk4_step};

#[derive(Debug, Clone)]
pub struct ControlBounds {
    lower: DVector<f64>,
    upper: DVector<f64>,
}

impl ControlBounds {
    pub fn new(lower: DVector<f64>, upper: DVector<f64>) -> Result<Self> {
        if lower.len() != upper.len() {
            return Err(dim_error(
                "control bounds",
                lower.len().to_string(),
                upper.len().to_string(),
            ));
        }
        for (lo, hi) in lower.iter().zip(upper.iter()) {
            if lo > hi {
                return Err(dim_error(
                    "control bounds",
                    "lower <= upper",
                    format!("{lo} > {hi}"),
                ));
            }
        }
        Ok(Self { lower, upper })
    }

    pub fn project(&self, control: &mut DVector<f64>) -> Result<()> {
        if control.len() != self.lower.len() {
            return Err(dim_error(
                "control projection",
                self.lower.len().to_string(),
                control.len().to_string(),
            ));
        }
        for i in 0..control.len() {
            control[i] = control[i].clamp(self.lower[i], self.upper[i]);
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ShootingSolver {
    pub dt: f64,
    pub max_iterations: usize,
    pub initial_step_size: f64,
    pub finite_difference_step: f64,
    pub gradient_tolerance: f64,
    pub cost_tolerance: f64,
    pub inequality_penalty: f64,
    pub augmented_lagrangian: Option<AugmentedLagrangianConfig>,
    pub bounds: Option<ControlBounds>,
}

#[derive(Debug, Clone)]
pub struct AugmentedLagrangianConfig {
    pub max_outer_iterations: usize,
    pub penalty_update_factor: f64,
    pub violation_tolerance: f64,
}

impl AugmentedLagrangianConfig {
    pub fn new(
        max_outer_iterations: usize,
        penalty_update_factor: f64,
        violation_tolerance: f64,
    ) -> Result<Self> {
        if max_outer_iterations == 0 {
            return Err(Error::Empty("augmented lagrangian outer iterations"));
        }
        if penalty_update_factor <= 1.0 {
            return Err(Error::NonPositiveParameter {
                name: "penalty_update_factor - 1",
                value: penalty_update_factor - 1.0,
            });
        }
        if violation_tolerance <= 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "violation_tolerance",
                value: violation_tolerance,
            });
        }
        Ok(Self {
            max_outer_iterations,
            penalty_update_factor,
            violation_tolerance,
        })
    }
}

impl Default for AugmentedLagrangianConfig {
    fn default() -> Self {
        Self {
            max_outer_iterations: 4,
            penalty_update_factor: 5.0,
            violation_tolerance: 1e-5,
        }
    }
}

impl ShootingSolver {
    pub fn new(dt: f64) -> Result<Self> {
        if dt <= 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "dt",
                value: dt,
            });
        }
        Ok(Self {
            dt,
            max_iterations: 64,
            initial_step_size: 0.25,
            finite_difference_step: 1e-5,
            gradient_tolerance: 1e-6,
            cost_tolerance: 1e-9,
            inequality_penalty: 1_000.0,
            augmented_lagrangian: None,
            bounds: None,
        })
    }

    pub fn with_bounds(mut self, bounds: ControlBounds) -> Self {
        self.bounds = Some(bounds);
        self
    }

    pub fn with_augmented_lagrangian(mut self, config: AugmentedLagrangianConfig) -> Self {
        self.augmented_lagrangian = Some(config);
        self
    }

    pub fn solve<P: OptimalControlProblem>(
        &self,
        problem: &P,
        x0: DVector<f64>,
        controls: Vec<DVector<f64>>,
        p: DVector<f64>,
        params: &GrampcLikeParams,
    ) -> Result<ShootingSolution> {
        if let Some(config) = &self.augmented_lagrangian {
            return self.solve_augmented_lagrangian(problem, x0, controls, p, params, config);
        }
        self.solve_inner(
            problem,
            x0,
            controls,
            p,
            params,
            None,
            self.inequality_penalty,
            0,
        )
    }

    fn solve_inner<P: OptimalControlProblem>(
        &self,
        problem: &P,
        x0: DVector<f64>,
        mut controls: Vec<DVector<f64>>,
        p: DVector<f64>,
        params: &GrampcLikeParams,
        multipliers: Option<&[DVector<f64>]>,
        penalty: f64,
        outer_iterations: usize,
    ) -> Result<ShootingSolution> {
        validate_problem_inputs(problem, &x0, &controls, &p)?;
        self.validate()?;
        project_controls(&mut controls, self.bounds.as_ref())?;

        let mut best =
            self.rollout_internal(problem, &x0, &controls, &p, params, multipliers, penalty)?;
        let mut previous_cost = best.cost;
        let mut iterations = 0;

        for iteration in 0..self.max_iterations {
            iterations = iteration + 1;
            let gradient =
                self.gradient(problem, &x0, &controls, &p, params, multipliers, penalty)?;
            let gradient_norm = gradient_norm(&gradient);
            if gradient_norm < self.gradient_tolerance {
                break;
            }

            let mut accepted = false;
            let mut step_size = self.initial_step_size;
            while step_size > 1e-12 {
                let mut candidate_controls = controls.clone();
                for (control, grad) in candidate_controls.iter_mut().zip(gradient.iter()) {
                    *control -= step_size * grad;
                }
                project_controls(&mut candidate_controls, self.bounds.as_ref())?;

                let candidate = self.rollout_internal(
                    problem,
                    &x0,
                    &candidate_controls,
                    &p,
                    params,
                    multipliers,
                    penalty,
                )?;
                if candidate.cost < best.cost {
                    controls = candidate_controls;
                    best = candidate;
                    accepted = true;
                    break;
                }
                step_size *= 0.5;
            }

            if !accepted || (previous_cost - best.cost).abs() < self.cost_tolerance {
                break;
            }
            previous_cost = best.cost;
        }

        Ok(ShootingSolution {
            controls,
            trajectory: best.trajectory,
            cost: best.cost,
            iterations,
            outer_iterations,
        })
    }

    fn solve_augmented_lagrangian<P: OptimalControlProblem>(
        &self,
        problem: &P,
        x0: DVector<f64>,
        mut controls: Vec<DVector<f64>>,
        p: DVector<f64>,
        params: &GrampcLikeParams,
        config: &AugmentedLagrangianConfig,
    ) -> Result<ShootingSolution> {
        validate_problem_inputs(problem, &x0, &controls, &p)?;
        self.validate()?;
        project_controls(&mut controls, self.bounds.as_ref())?;

        let inequalities = problem.dimensions().inequalities;
        if inequalities == 0 {
            return self.solve_inner(
                problem,
                x0,
                controls,
                p,
                params,
                None,
                self.inequality_penalty,
                0,
            );
        }

        let mut multipliers = vec![DVector::zeros(inequalities); controls.len()];
        let mut penalty = self.inequality_penalty;
        let mut total_iterations = 0;
        let mut outer_iterations = 0;

        for outer in 0..config.max_outer_iterations {
            outer_iterations = outer + 1;
            let solution = self.solve_inner(
                problem,
                x0.clone(),
                controls,
                p.clone(),
                params,
                Some(&multipliers),
                penalty,
                outer_iterations,
            )?;
            total_iterations += solution.iterations;
            controls = solution.controls;

            let violations = self.stage_constraints(problem, &x0, &controls, &p, params)?;
            let max_violation = max_positive_violation(&violations);
            update_multipliers(&mut multipliers, &violations, penalty);
            if max_violation <= config.violation_tolerance {
                break;
            }
            penalty *= config.penalty_update_factor;
        }

        let rollout = self.rollout(problem, &x0, &controls, &p, params)?;
        Ok(ShootingSolution {
            controls,
            trajectory: rollout.trajectory,
            cost: rollout.cost,
            iterations: total_iterations,
            outer_iterations,
        })
    }

    pub fn rollout<P: OptimalControlProblem>(
        &self,
        problem: &P,
        x0: &DVector<f64>,
        controls: &[DVector<f64>],
        p: &DVector<f64>,
        params: &GrampcLikeParams,
    ) -> Result<Rollout> {
        self.rollout_internal(
            problem,
            x0,
            controls,
            p,
            params,
            None,
            self.inequality_penalty,
        )
    }

    fn rollout_internal<P: OptimalControlProblem>(
        &self,
        problem: &P,
        x0: &DVector<f64>,
        controls: &[DVector<f64>],
        p: &DVector<f64>,
        params: &GrampcLikeParams,
        multipliers: Option<&[DVector<f64>]>,
        penalty: f64,
    ) -> Result<Rollout> {
        validate_problem_inputs(problem, x0, controls, p)?;
        self.validate()?;
        validate_multipliers(
            multipliers,
            controls.len(),
            problem.dimensions().inequalities,
        )?;
        let mut t = 0.0;
        let mut x = x0.clone();
        let mut cost = 0.0;
        let mut trajectory = Vec::with_capacity(controls.len() + 1);
        trajectory.push(SimulationStep { t, x: x.clone() });

        for control in controls {
            cost += self.dt * problem.stage_cost(t, &x, control, p, params);
            let constraints = problem.inequality_constraints(t, &x, control, p, params);
            let step_multiplier = multipliers.map(|values| &values[trajectory.len() - 1]);
            cost += self.dt * inequality_merit(&constraints, step_multiplier, penalty);
            x = rk4_step(problem, t, &x, control, p, params, self.dt);
            t += self.dt;
            trajectory.push(SimulationStep { t, x: x.clone() });
        }

        cost += problem.terminal_cost(t, &x, p, params);
        Ok(Rollout { trajectory, cost })
    }

    fn stage_constraints<P: OptimalControlProblem>(
        &self,
        problem: &P,
        x0: &DVector<f64>,
        controls: &[DVector<f64>],
        p: &DVector<f64>,
        params: &GrampcLikeParams,
    ) -> Result<Vec<DVector<f64>>> {
        validate_problem_inputs(problem, x0, controls, p)?;
        self.validate()?;
        let mut t = 0.0;
        let mut x = x0.clone();
        let mut constraints = Vec::with_capacity(controls.len());
        for control in controls {
            constraints.push(problem.inequality_constraints(t, &x, control, p, params));
            x = rk4_step(problem, t, &x, control, p, params, self.dt);
            t += self.dt;
        }
        Ok(constraints)
    }

    fn gradient<P: OptimalControlProblem>(
        &self,
        problem: &P,
        x0: &DVector<f64>,
        controls: &[DVector<f64>],
        p: &DVector<f64>,
        params: &GrampcLikeParams,
        multipliers: Option<&[DVector<f64>]>,
        penalty: f64,
    ) -> Result<Vec<DVector<f64>>> {
        let mut gradient = controls.to_vec();
        for step_idx in 0..controls.len() {
            for control_idx in 0..controls[step_idx].len() {
                let mut plus_controls = controls.to_vec();
                let mut minus_controls = controls.to_vec();
                plus_controls[step_idx][control_idx] += self.finite_difference_step;
                minus_controls[step_idx][control_idx] -= self.finite_difference_step;
                project_controls(&mut plus_controls, self.bounds.as_ref())?;
                project_controls(&mut minus_controls, self.bounds.as_ref())?;

                let plus_cost = self
                    .rollout_internal(problem, x0, &plus_controls, p, params, multipliers, penalty)?
                    .cost;
                let minus_cost = self
                    .rollout_internal(
                        problem,
                        x0,
                        &minus_controls,
                        p,
                        params,
                        multipliers,
                        penalty,
                    )?
                    .cost;
                gradient[step_idx][control_idx] =
                    (plus_cost - minus_cost) / (2.0 * self.finite_difference_step);
            }
        }
        Ok(gradient)
    }

    fn validate(&self) -> Result<()> {
        if self.dt <= 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "dt",
                value: self.dt,
            });
        }
        if self.initial_step_size <= 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "initial_step_size",
                value: self.initial_step_size,
            });
        }
        if self.finite_difference_step <= 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "finite_difference_step",
                value: self.finite_difference_step,
            });
        }
        if self.inequality_penalty <= 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "inequality_penalty",
                value: self.inequality_penalty,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ShootingSolution {
    pub controls: Vec<DVector<f64>>,
    pub trajectory: Vec<SimulationStep>,
    pub cost: f64,
    pub iterations: usize,
    pub outer_iterations: usize,
}

#[derive(Debug, Clone)]
pub struct Rollout {
    pub trajectory: Vec<SimulationStep>,
    pub cost: f64,
}

fn validate_problem_inputs<P: OptimalControlProblem>(
    problem: &P,
    x0: &DVector<f64>,
    controls: &[DVector<f64>],
    p: &DVector<f64>,
) -> Result<()> {
    let dims = problem.dimensions();
    if x0.len() != dims.states {
        return Err(dim_error(
            "shooting initial state",
            dims.states.to_string(),
            x0.len().to_string(),
        ));
    }
    if p.len() != dims.parameters {
        return Err(dim_error(
            "shooting parameters",
            dims.parameters.to_string(),
            p.len().to_string(),
        ));
    }
    if controls.is_empty() {
        return Err(Error::Empty("controls"));
    }
    for control in controls {
        if control.len() != dims.controls {
            return Err(dim_error(
                "shooting control",
                dims.controls.to_string(),
                control.len().to_string(),
            ));
        }
    }
    Ok(())
}

fn validate_multipliers(
    multipliers: Option<&[DVector<f64>]>,
    steps: usize,
    inequalities: usize,
) -> Result<()> {
    if let Some(multipliers) = multipliers {
        if multipliers.len() != steps {
            return Err(dim_error(
                "augmented lagrangian multipliers",
                steps.to_string(),
                multipliers.len().to_string(),
            ));
        }
        for multiplier in multipliers {
            if multiplier.len() != inequalities {
                return Err(dim_error(
                    "augmented lagrangian multiplier",
                    inequalities.to_string(),
                    multiplier.len().to_string(),
                ));
            }
        }
    }
    Ok(())
}

fn inequality_merit(
    values: &DVector<f64>,
    multipliers: Option<&DVector<f64>>,
    penalty: f64,
) -> f64 {
    match multipliers {
        Some(multipliers) => values
            .iter()
            .zip(multipliers.iter())
            .map(|(constraint, multiplier)| {
                let shifted = (multiplier + penalty * constraint).max(0.0);
                (shifted.powi(2) - multiplier.powi(2)) / (2.0 * penalty)
            })
            .sum(),
        None => values
            .iter()
            .map(|value| penalty * value.max(0.0).powi(2))
            .sum(),
    }
}

fn update_multipliers(
    multipliers: &mut [DVector<f64>],
    constraints: &[DVector<f64>],
    penalty: f64,
) {
    for (multiplier, constraint) in multipliers.iter_mut().zip(constraints.iter()) {
        for i in 0..multiplier.len() {
            multiplier[i] = (multiplier[i] + penalty * constraint[i]).max(0.0);
        }
    }
}

fn max_positive_violation(constraints: &[DVector<f64>]) -> f64 {
    constraints
        .iter()
        .flat_map(|values| values.iter())
        .fold(0.0_f64, |max_value, value| max_value.max(value.max(0.0)))
}

fn project_controls(controls: &mut [DVector<f64>], bounds: Option<&ControlBounds>) -> Result<()> {
    if let Some(bounds) = bounds {
        for control in controls {
            bounds.project(control)?;
        }
    }
    Ok(())
}

fn gradient_norm(gradient: &[DVector<f64>]) -> f64 {
    gradient
        .iter()
        .map(|control_grad| control_grad.norm_squared())
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{DoubleIntegrator, Dynamics, GrampcLikeParams, ProblemDimensions};

    struct StaticControlLimit;

    impl Dynamics for StaticControlLimit {
        fn dimensions(&self) -> ProblemDimensions {
            ProblemDimensions::new(1, 1, 0, 0, 1)
        }

        fn dynamics(
            &self,
            _t: f64,
            _x: &DVector<f64>,
            _u: &DVector<f64>,
            _p: &DVector<f64>,
            _params: &GrampcLikeParams,
        ) -> DVector<f64> {
            DVector::zeros(1)
        }
    }

    impl OptimalControlProblem for StaticControlLimit {
        fn stage_cost(
            &self,
            _t: f64,
            _x: &DVector<f64>,
            u: &DVector<f64>,
            _p: &DVector<f64>,
            _params: &GrampcLikeParams,
        ) -> f64 {
            (u[0] - 1.0).powi(2)
        }

        fn inequality_constraints(
            &self,
            _t: f64,
            _x: &DVector<f64>,
            u: &DVector<f64>,
            _p: &DVector<f64>,
            _params: &GrampcLikeParams,
        ) -> DVector<f64> {
            DVector::from_vec(vec![u[0] - 0.2])
        }
    }

    #[test]
    fn shooting_solver_reduces_double_integrator_cost() {
        let problem = DoubleIntegrator::new([0.01, 0.1, 0.05, 10.0, 2.0, 0.0], -10.0);
        let params = GrampcLikeParams::new(
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0]),
        );
        let x0 = DVector::from_vec(vec![0.0, 0.0]);
        let p = DVector::zeros(0);
        let initial_controls = vec![DVector::from_vec(vec![0.0]); 8];
        let solver = ShootingSolver::new(0.1).unwrap().with_bounds(
            ControlBounds::new(DVector::from_vec(vec![-5.0]), DVector::from_vec(vec![5.0]))
                .unwrap(),
        );

        let initial_cost = solver
            .rollout(&problem, &x0, &initial_controls, &p, &params)
            .unwrap()
            .cost;
        let solution = solver
            .solve(&problem, x0, initial_controls, p, &params)
            .unwrap();

        assert!(solution.cost < initial_cost);
        assert_eq!(solution.controls.len(), 8);
        assert_eq!(solution.trajectory.len(), 9);
        for control in solution.controls {
            assert!((-5.0..=5.0).contains(&control[0]));
        }
    }

    #[test]
    fn augmented_lagrangian_handles_active_control_constraint() {
        let problem = StaticControlLimit;
        let params = GrampcLikeParams::new(DVector::zeros(1), DVector::zeros(1));
        let x0 = DVector::zeros(1);
        let p = DVector::zeros(0);
        let initial_controls = vec![DVector::from_vec(vec![1.0])];
        let mut solver = ShootingSolver::new(1.0)
            .unwrap()
            .with_bounds(
                ControlBounds::new(DVector::from_vec(vec![-2.0]), DVector::from_vec(vec![2.0]))
                    .unwrap(),
            )
            .with_augmented_lagrangian(AugmentedLagrangianConfig::new(6, 5.0, 1e-4).unwrap());
        solver.max_iterations = 80;
        solver.initial_step_size = 0.2;
        solver.inequality_penalty = 10.0;

        let solution = solver
            .solve(&problem, x0, initial_controls, p, &params)
            .unwrap();

        assert!(solution.controls[0][0] <= 0.205);
        assert!(solution.outer_iterations > 1);
    }
}
