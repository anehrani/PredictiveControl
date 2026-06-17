use nalgebra::DVector;

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
