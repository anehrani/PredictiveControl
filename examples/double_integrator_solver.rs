use grampc_s_rs::{
    AugmentedLagrangianConfig, ControlBounds, DoubleIntegrator, GrampcLikeParams, ShootingSolver,
};
use nalgebra::DVector;

fn main() -> grampc_s_rs::Result<()> {
    let problem = DoubleIntegrator::new([0.01, 0.1, 0.05, 10.0, 2.0, 0.0], -10.0);
    let params = GrampcLikeParams::new(
        DVector::from_vec(vec![1.0, 0.0]),
        DVector::from_vec(vec![0.0]),
    );
    let x0 = DVector::from_vec(vec![0.0, 0.0]);
    let p = DVector::zeros(0);
    let initial_controls = vec![DVector::from_vec(vec![0.0]); 8];
    let solver = ShootingSolver::new(0.1)?
        .with_bounds(ControlBounds::new(
            DVector::from_vec(vec![-5.0]),
            DVector::from_vec(vec![5.0]),
        )?)
        .with_augmented_lagrangian(AugmentedLagrangianConfig::default());

    let initial_cost = solver
        .rollout(&problem, &x0, &initial_controls, &p, &params)?
        .cost;
    let solution = solver.solve(&problem, x0, initial_controls, p, &params)?;
    let final_state = &solution.trajectory.last().expect("non-empty trajectory").x;

    println!("initial cost: {:.6}", initial_cost);
    println!("optimized cost: {:.6}", solution.cost);
    println!("iterations: {}", solution.iterations);
    println!("outer iterations: {}", solution.outer_iterations);
    println!("final state: {}", final_state.transpose());
    println!(
        "first optimized control: {}",
        solution.controls[0].transpose()
    );

    Ok(())
}
