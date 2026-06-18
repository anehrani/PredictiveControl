use grampc_s_rs::{
    Gaussian, GrampcLikeParams, MassSpringDamper, SigmaPointDynamics, UnscentedTransformation,
    simulate_constant_control,
};
use nalgebra::{DMatrix, DVector};

fn main() -> grampc_s_rs::Result<()> {
    let number_of_masses = 5;
    let states = 2 * number_of_masses;
    let problem = MassSpringDamper::new(number_of_masses, 1.0)?;
    let params = GrampcLikeParams::new(DVector::zeros(states), DVector::zeros(2));

    let mut x0 = DVector::zeros(states);
    x0[0] = 1.0;
    x0[1] = 0.5;

    let u = DVector::from_vec(vec![0.2, -0.1]);
    let p = DVector::from_vec(vec![1.0, 0.2]);
    let trajectory = simulate_constant_control(
        &problem,
        x0.clone(),
        u.clone(),
        p.clone(),
        &params,
        0.01,
        200,
    );
    let final_state = &trajectory.last().expect("non-empty trajectory").x;

    println!("masses: {}", problem.number_of_masses());
    println!("spring constant: {:.3}", p[0]);
    println!("damping constant: {:.3}", p[1]);
    println!("final deterministic state: {}", final_state.transpose());

    let mut initial_covariance = DMatrix::identity(states, states) * 1e-4;
    for i in 0..number_of_masses {
        initial_covariance[(i, i)] = 1e-3;
    }
    let state_distribution = Gaussian::new(x0, initial_covariance)?;
    let dynamics_moments =
        SigmaPointDynamics::new(UnscentedTransformation::new(states, states, 1.0, 2.0, 0.0)?)
            .approximate(&problem, 0.0, &state_distribution, &u, &p, &params)?;

    println!(
        "initial dynamics mean: {}",
        dynamics_moments.mean.transpose()
    );
    println!(
        "initial dynamics covariance diagonal: {}",
        dynamics_moments.covariance.diagonal().transpose()
    );

    Ok(())
}
