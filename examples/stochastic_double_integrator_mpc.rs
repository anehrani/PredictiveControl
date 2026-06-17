use grampc_s_rs::{
    AugmentedLagrangianConfig, ChanceConstraintApproximation, ControlBounds, DoubleIntegrator,
    GrampcLikeParams, ShootingSolver, StochasticMpcProblem, UnscentedTransformation,
};
use nalgebra::{DMatrix, DVector};

fn main() -> grampc_s_rs::Result<()> {
    let base_problem = DoubleIntegrator::new([0.01, 0.1, 0.05, 10.0, 2.0, 0.0], -0.2);
    let stochastic_problem = StochasticMpcProblem::new(
        base_problem,
        UnscentedTransformation::new(2, 2, 1.0, 2.0, 0.0)?,
        UnscentedTransformation::new(2, 1, 1.0, 2.0, 0.0)?,
    )?
    .with_chance_constraints(ChanceConstraintApproximation::gaussian(DVector::from_vec(
        vec![0.95],
    ))?)?
    .with_covariance_trace_weight(0.1)?
    .with_covariance_jitter(1e-12)?;

    let params = GrampcLikeParams::new(
        DVector::from_vec(vec![1.0, 0.0]),
        DVector::from_vec(vec![0.0]),
    );
    let initial_state = stochastic_problem.initial_state(
        DVector::from_vec(vec![0.0, 0.0]),
        DMatrix::from_diagonal(&DVector::from_vec(vec![0.04, 0.01])),
    )?;
    let initial_controls = vec![DVector::from_vec(vec![0.0]); 8];
    let solver = ShootingSolver::new(0.1)?
        .with_bounds(ControlBounds::new(
            DVector::from_vec(vec![-5.0]),
            DVector::from_vec(vec![5.0]),
        )?)
        .with_augmented_lagrangian(AugmentedLagrangianConfig::default());

    let initial_cost = solver
        .rollout(
            &stochastic_problem,
            &initial_state,
            &initial_controls,
            &DVector::zeros(0),
            &params,
        )?
        .cost;
    let solution = solver.solve(
        &stochastic_problem,
        initial_state,
        initial_controls,
        DVector::zeros(0),
        &params,
    )?;
    let final_moments = stochastic_problem.unpack_state(&solution.trajectory.last().unwrap().x)?;

    println!("initial stochastic cost: {:.6}", initial_cost);
    println!("optimized stochastic cost: {:.6}", solution.cost);
    println!("iterations: {}", solution.iterations);
    println!("outer iterations: {}", solution.outer_iterations);
    println!("final mean: {}", final_moments.mean.transpose());
    println!("final covariance:\n{}", final_moments.covariance);
    println!(
        "first optimized control: {}",
        solution.controls[0].transpose()
    );

    Ok(())
}
