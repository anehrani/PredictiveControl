use nalgebra::{DMatrix, DVector};

use crate::error::{Error, Result, dim_error};

pub trait StationaryKernel {
    fn input_dimension(&self) -> usize;
    fn evaluate(&self, tau: &DVector<f64>) -> f64;

    fn gradient(&self, tau: &DVector<f64>, derivative_indices: &[usize]) -> DVector<f64> {
        let eps = 1e-6;
        DVector::from_iterator(
            derivative_indices.len(),
            derivative_indices.iter().map(|&idx| {
                let mut plus = tau.clone();
                let mut minus = tau.clone();
                plus[idx] += eps;
                minus[idx] -= eps;
                (self.evaluate(&plus) - self.evaluate(&minus)) / (2.0 * eps)
            }),
        )
    }
}

#[derive(Debug, Clone)]
pub struct SquaredExponentialKernel {
    sigma_squared: f64,
    length_scale_squared: DVector<f64>,
}

impl SquaredExponentialKernel {
    pub fn new(sigma: f64, length_scale: DVector<f64>) -> Result<Self> {
        if sigma <= 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "sigma",
                value: sigma,
            });
        }
        if length_scale.is_empty() {
            return Err(Error::Empty("length_scale"));
        }
        for value in length_scale.iter().copied() {
            if value <= 0.0 {
                return Err(Error::NonPositiveParameter {
                    name: "length_scale",
                    value,
                });
            }
        }
        Ok(Self {
            sigma_squared: sigma.powi(2),
            length_scale_squared: length_scale.map(|value| value.powi(2)),
        })
    }
}

impl StationaryKernel for SquaredExponentialKernel {
    fn input_dimension(&self) -> usize {
        self.length_scale_squared.len()
    }

    fn evaluate(&self, tau: &DVector<f64>) -> f64 {
        assert_eq!(tau.len(), self.input_dimension());
        let scaled_norm: f64 = tau
            .iter()
            .zip(self.length_scale_squared.iter())
            .map(|(tau_i, ell_sq)| tau_i.powi(2) / ell_sq)
            .sum();
        self.sigma_squared * (-0.5 * scaled_norm).exp()
    }

    fn gradient(&self, tau: &DVector<f64>, derivative_indices: &[usize]) -> DVector<f64> {
        let kernel = self.evaluate(tau);
        DVector::from_iterator(
            derivative_indices.len(),
            derivative_indices
                .iter()
                .map(|&idx| -kernel * tau[idx] / self.length_scale_squared[idx]),
        )
    }
}

#[derive(Debug, Clone)]
pub struct GaussianProcessData {
    pub input_data: DMatrix<f64>,
    pub output_data: DVector<f64>,
    pub output_noise_variance: f64,
}

impl GaussianProcessData {
    pub fn new(
        input_data: DMatrix<f64>,
        output_data: DVector<f64>,
        output_noise_variance: f64,
    ) -> Result<Self> {
        if input_data.ncols() != output_data.len() {
            return Err(dim_error(
                "gaussian process data",
                format!("{} outputs", input_data.ncols()),
                format!("{} outputs", output_data.len()),
            ));
        }
        if output_noise_variance < 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "output_noise_variance",
                value: output_noise_variance,
            });
        }
        Ok(Self {
            input_data,
            output_data,
            output_noise_variance,
        })
    }
}

#[derive(Debug, Clone)]
pub struct GaussianProcess<K> {
    kernel: K,
    input_data: DMatrix<f64>,
    k_inv: DMatrix<f64>,
    k_inv_y: DVector<f64>,
    state_dependency: Vec<bool>,
    control_dependency: Vec<bool>,
    state_indices: Vec<usize>,
    control_indices: Vec<usize>,
    kernel_zero: f64,
}

impl<K: StationaryKernel> GaussianProcess<K> {
    pub fn new(
        data: GaussianProcessData,
        kernel: K,
        state_dependency: Vec<bool>,
        control_dependency: Vec<bool>,
    ) -> Result<Self> {
        let input_dim = data.input_data.nrows();
        if input_dim != kernel.input_dimension() {
            return Err(dim_error(
                "gaussian process kernel",
                kernel.input_dimension().to_string(),
                input_dim.to_string(),
            ));
        }
        let active_dim = state_dependency.iter().filter(|&&x| x).count()
            + control_dependency.iter().filter(|&&x| x).count();
        if active_dim != input_dim {
            return Err(dim_error(
                "gaussian process dependencies",
                input_dim.to_string(),
                active_dim.to_string(),
            ));
        }

        let n = data.input_data.ncols();
        let mut covariance = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let tau = data.input_data.column(i) - data.input_data.column(j);
                covariance[(i, j)] = kernel.evaluate(&tau.into_owned());
            }
        }
        for i in 0..n {
            covariance[(i, i)] += data.output_noise_variance;
        }
        let lu = covariance.lu();
        let identity = DMatrix::identity(n, n);
        let k_inv = lu
            .solve(&identity)
            .ok_or(Error::LinearSolve("gaussian process covariance"))?;
        let k_inv_y = &k_inv * &data.output_data;
        let state_indices: Vec<usize> =
            (0..state_dependency.iter().filter(|&&x| x).count()).collect();
        let control_indices = (state_indices.len()..input_dim).collect::<Vec<usize>>();
        let kernel_zero = kernel.evaluate(&DVector::zeros(input_dim));

        Ok(Self {
            kernel,
            input_data: data.input_data,
            k_inv,
            k_inv_y,
            state_dependency,
            control_dependency,
            state_indices,
            control_indices,
            kernel_zero,
        })
    }

    pub fn mean(&self, state: &DVector<f64>, control: &DVector<f64>) -> Result<f64> {
        let diff = self.point_differences(state, control)?;
        let k_star = self.kernel_vector(&diff);
        Ok(k_star.dot(&self.k_inv_y))
    }

    pub fn variance(&self, state: &DVector<f64>, control: &DVector<f64>) -> Result<f64> {
        let diff = self.point_differences(state, control)?;
        let k_star = self.kernel_vector(&diff);
        Ok(self.kernel_zero - k_star.dot(&(&self.k_inv * &k_star)))
    }

    pub fn mean_gradient_state(
        &self,
        state: &DVector<f64>,
        control: &DVector<f64>,
    ) -> Result<DVector<f64>> {
        let diff = self.point_differences(state, control)?;
        let active = self.gradient_active(&diff, &self.state_indices, &self.k_inv_y);
        Ok(scatter_active(&self.state_dependency, &active))
    }

    pub fn mean_gradient_control(
        &self,
        state: &DVector<f64>,
        control: &DVector<f64>,
    ) -> Result<DVector<f64>> {
        let diff = self.point_differences(state, control)?;
        let active = self.gradient_active(&diff, &self.control_indices, &self.k_inv_y);
        Ok(scatter_active(&self.control_dependency, &active))
    }

    pub fn variance_gradient_state(
        &self,
        state: &DVector<f64>,
        control: &DVector<f64>,
    ) -> Result<DVector<f64>> {
        let diff = self.point_differences(state, control)?;
        let k_star = self.kernel_vector(&diff);
        let projected = &self.k_inv * k_star;
        let active = -2.0 * self.gradient_active(&diff, &self.state_indices, &projected);
        Ok(scatter_active(&self.state_dependency, &active))
    }

    pub fn variance_gradient_control(
        &self,
        state: &DVector<f64>,
        control: &DVector<f64>,
    ) -> Result<DVector<f64>> {
        let diff = self.point_differences(state, control)?;
        let k_star = self.kernel_vector(&diff);
        let projected = &self.k_inv * k_star;
        let active = -2.0 * self.gradient_active(&diff, &self.control_indices, &projected);
        Ok(scatter_active(&self.control_dependency, &active))
    }

    fn kernel_vector(&self, point_diff: &DMatrix<f64>) -> DVector<f64> {
        DVector::from_fn(point_diff.ncols(), |i, _| {
            self.kernel.evaluate(&point_diff.column(i).into_owned())
        })
    }

    fn gradient_active(
        &self,
        point_diff: &DMatrix<f64>,
        indices: &[usize],
        projected: &DVector<f64>,
    ) -> DVector<f64> {
        let mut out = DVector::zeros(indices.len());
        for i in 0..point_diff.ncols() {
            let gradient = self
                .kernel
                .gradient(&point_diff.column(i).into_owned(), indices);
            out += gradient * projected[i];
        }
        out
    }

    fn point_differences(
        &self,
        state: &DVector<f64>,
        control: &DVector<f64>,
    ) -> Result<DMatrix<f64>> {
        if state.len() != self.state_dependency.len() {
            return Err(dim_error(
                "gaussian process state",
                self.state_dependency.len().to_string(),
                state.len().to_string(),
            ));
        }
        if control.len() != self.control_dependency.len() {
            return Err(dim_error(
                "gaussian process control",
                self.control_dependency.len().to_string(),
                control.len().to_string(),
            ));
        }
        let mut evaluation_point = DVector::zeros(self.input_data.nrows());
        let mut index = 0;
        for (i, depends) in self.state_dependency.iter().copied().enumerate() {
            if depends {
                evaluation_point[index] = state[i];
                index += 1;
            }
        }
        for (i, depends) in self.control_dependency.iter().copied().enumerate() {
            if depends {
                evaluation_point[index] = control[i];
                index += 1;
            }
        }
        let mut out = DMatrix::zeros(self.input_data.nrows(), self.input_data.ncols());
        for i in 0..self.input_data.ncols() {
            out.set_column(i, &(&evaluation_point - self.input_data.column(i)));
        }
        Ok(out)
    }
}

fn scatter_active(mask: &[bool], active: &DVector<f64>) -> DVector<f64> {
    let mut out = DVector::zeros(mask.len());
    let mut index = 0;
    for (i, depends) in mask.iter().copied().enumerate() {
        if depends {
            out[i] = active[index];
            index += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gp_interpolates_low_noise_training_point() {
        let data = GaussianProcessData::new(
            DMatrix::from_row_slice(1, 3, &[0.0, 1.0, 2.0]),
            DVector::from_vec(vec![0.0, 1.0, 0.0]),
            1e-10,
        )
        .unwrap();
        let kernel = SquaredExponentialKernel::new(1.0, DVector::from_vec(vec![0.5])).unwrap();
        let gp = GaussianProcess::new(data, kernel, vec![true], vec![]).unwrap();
        let mean = gp
            .mean(&DVector::from_vec(vec![1.0]), &DVector::zeros(0))
            .unwrap();
        assert!((mean - 1.0).abs() < 1e-6);
    }
}
