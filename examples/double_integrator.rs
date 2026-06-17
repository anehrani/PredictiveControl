use grampc_s_rs::{
    ChanceConstraintApproximation, DoubleIntegrator, Gaussian, GrampcLikeParams,
    PointTransformation, UnscentedTransformation, simulate_constant_control,
};
use nalgebra::{DMatrix, DVector};

fn main() -> grampc_s_rs::Result<()> {
    let problem = DoubleIntegrator::new([0.1, 1.0, 0.5, 10.0, 5.0, 0.0], 1.5);
    let params = GrampcLikeParams::new(
        DVector::from_vec(vec![1.0, 0.0]),
        DVector::from_vec(vec![0.0]),
    );

    let trajectory = simulate_constant_control(
        &problem,
        DVector::from_vec(vec![0.0, 0.0]),
        DVector::from_vec(vec![0.8]),
        DVector::zeros(0),
        &params,
        0.05,
        20,
    );
    println!("final state: {}", trajectory.last().unwrap().x.transpose());

    let state_distribution = Gaussian::new(
        DVector::from_vec(vec![0.0, 0.0]),
        DMatrix::from_row_slice(2, 2, &[0.04, 0.0, 0.0, 0.01]),
    )?;
    let ut = UnscentedTransformation::new(2, 2, 1.0, 2.0, 0.0)?;
    let sigma_points = ut.points_from_distribution(&state_distribution)?;

    let propagated = DMatrix::from_columns(
        &(0..sigma_points.ncols())
            .map(|i| {
                let x = sigma_points.column(i).into_owned();
                let next = simulate_constant_control(
                    &problem,
                    x,
                    DVector::from_vec(vec![0.8]),
                    DVector::zeros(0),
                    &params,
                    0.05,
                    1,
                );
                next.last().unwrap().x.clone()
            })
            .collect::<Vec<_>>(),
    );

    let mean = ut.mean(&propagated)?;
    let covariance = ut.covariance(&propagated, &propagated)?;
    println!("propagated mean: {}", mean.transpose());
    println!("propagated covariance:\n{covariance}");

    let chance = ChanceConstraintApproximation::gaussian(DVector::from_vec(vec![0.95]))?;
    println!(
        "95% Gaussian tightening coefficient: {:.6}",
        chance.tightening_coefficients()[0]
    );

    Ok(())
}
