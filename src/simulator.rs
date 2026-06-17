use nalgebra::DVector;

use crate::problem::{Dynamics, GrampcLikeParams};

#[derive(Debug, Clone)]
pub struct SimulationStep {
    pub t: f64,
    pub x: DVector<f64>,
}

pub fn rk4_step<P: Dynamics>(
    problem: &P,
    t: f64,
    x: &DVector<f64>,
    u: &DVector<f64>,
    p: &DVector<f64>,
    params: &GrampcLikeParams,
    dt: f64,
) -> DVector<f64> {
    let half = 0.5 * dt;
    let k1 = problem.dynamics(t, x, u, p, params);
    let k2 = problem.dynamics(t + half, &(x + half * &k1), u, p, params);
    let k3 = problem.dynamics(t + half, &(x + half * &k2), u, p, params);
    let k4 = problem.dynamics(t + dt, &(x + dt * &k3), u, p, params);
    x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
}

pub fn simulate_constant_control<P: Dynamics>(
    problem: &P,
    x0: DVector<f64>,
    u: DVector<f64>,
    p: DVector<f64>,
    params: &GrampcLikeParams,
    dt: f64,
    steps: usize,
) -> Vec<SimulationStep> {
    let mut t = 0.0;
    let mut x = x0;
    let mut trajectory = Vec::with_capacity(steps + 1);
    trajectory.push(SimulationStep { t, x: x.clone() });
    for _ in 0..steps {
        x = rk4_step(problem, t, &x, &u, &p, params, dt);
        t += dt;
        trajectory.push(SimulationStep { t, x: x.clone() });
    }
    trajectory
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{DoubleIntegrator, GrampcLikeParams};

    #[test]
    fn double_integrator_constant_acceleration() {
        let problem = DoubleIntegrator::new([1.0; 6], 0.0);
        let params = GrampcLikeParams::new(
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![0.0]),
        );
        let trajectory = simulate_constant_control(
            &problem,
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![2.0]),
            DVector::zeros(0),
            &params,
            0.1,
            10,
        );
        let final_x = &trajectory.last().unwrap().x;
        assert!((final_x[0] - 1.0).abs() < 1e-12);
        assert!((final_x[1] - 2.0).abs() < 1e-12);
    }
}
