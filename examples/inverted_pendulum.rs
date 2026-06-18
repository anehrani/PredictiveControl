use grampc_s_rs::{
    ChanceConstraintApproximation, Gaussian, GrampcLikeParams, InvertedPendulum,
    OptimalControlProblem, TaylorDynamics, simulate_constant_control,
};
use nalgebra::{DMatrix, DVector};

fn main() -> grampc_s_rs::Result<()> {
    let problem = InvertedPendulum::benchmark();
    let params = GrampcLikeParams::new(
        DVector::from_vec(vec![0.0, 0.0, std::f64::consts::PI, 0.0]),
        DVector::from_vec(vec![0.0]),
    );

    let x0 = DVector::from_vec(vec![0.0, 0.0, std::f64::consts::PI, 0.2]);
    let u = DVector::from_vec(vec![0.0]);
    let p = DVector::zeros(0);
    let trajectory = simulate_constant_control(
        &problem,
        x0.clone(),
        u.clone(),
        p.clone(),
        &params,
        0.02,
        35,
    );
    let final_state = &trajectory.last().expect("non-empty trajectory").x;
    let constraints = problem.inequality_constraints(0.7, final_state, &u, &p, &params);

    println!("final pendulum state: {}", final_state.transpose());
    println!(
        "final velocity constraint residuals: {}",
        constraints.transpose()
    );

    let state_distribution = Gaussian::new(x0, DMatrix::identity(4, 4) * 1e-6)?;
    let moments = TaylorDynamics::new(1e-5)?.approximate(
        &problem,
        0.0,
        &state_distribution,
        &u,
        &p,
        &params,
    )?;
    let chance = ChanceConstraintApproximation::gaussian(DVector::from_vec(vec![0.95, 0.95]))?;

    println!("initial Taylor dynamics mean: {}", moments.mean.transpose());
    println!(
        "initial Taylor dynamics covariance:\n{}",
        moments.covariance
    );
    println!(
        "95% Gaussian tightening coefficients: {}",
        chance.tightening_coefficients().transpose()
    );

    Ok(())
}
