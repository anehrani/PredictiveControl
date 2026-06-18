use grampc_s_rs::{
    ChanceConstraintApproximation, Gaussian, GrampcLikeParams, OptimalControlProblem,
    PceTransformation, PolynomialFamily, Reactor, SigmaPointDynamics, simulate_constant_control,
};
use nalgebra::{DMatrix, DVector};

fn main() -> grampc_s_rs::Result<()> {
    let problem = Reactor::benchmark();
    let params = GrampcLikeParams::new(
        DVector::from_vec(vec![0.215, 0.09]),
        DVector::from_vec(vec![19.6]),
    );

    let x0 = DVector::from_vec(vec![0.8, 0.01]);
    let u = DVector::from_vec(vec![19.6]);
    let p = DVector::zeros(0);
    let trajectory = simulate_constant_control(
        &problem,
        x0.clone(),
        u.clone(),
        p.clone(),
        &params,
        1e-4,
        100,
    );
    let final_state = &trajectory.last().expect("non-empty trajectory").x;
    let final_constraint = problem.inequality_constraints(0.01, final_state, &u, &p, &params);

    println!("final reactor state: {}", final_state.transpose());
    println!(
        "final concentration constraint residual: {:.6}",
        final_constraint[0]
    );

    let state_distribution = Gaussian::new(
        x0,
        DMatrix::from_diagonal(&DVector::from_vec(vec![1e-5, 1e-5])),
    )?;
    let pce = PceTransformation::with_uniform_order(
        2,
        2,
        &[PolynomialFamily::Hermite, PolynomialFamily::Hermite],
        2,
        3,
    )?;
    let dynamics_moments = SigmaPointDynamics::new(pce).approximate(
        &problem,
        0.0,
        &state_distribution,
        &u,
        &p,
        &params,
    )?;
    let chance = ChanceConstraintApproximation::chebyshev(DVector::from_vec(vec![0.95]))?;

    println!(
        "initial PCE dynamics mean: {}",
        dynamics_moments.mean.transpose()
    );
    println!(
        "initial PCE dynamics covariance:\n{}",
        dynamics_moments.covariance
    );
    println!(
        "95% Chebyshev tightening coefficient: {:.6}",
        chance.tightening_coefficients()[0]
    );

    Ok(())
}
