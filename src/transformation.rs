use nalgebra::{DMatrix, DVector, linalg::SymmetricEigen};

use crate::distribution::{Distribution, PolynomialFamily};
use crate::error::{Error, Result, dim_error};

pub trait PointTransformation {
    fn input_dimension(&self) -> usize;
    fn output_dimension(&self) -> usize;
    fn number_of_points(&self) -> usize;
    fn normalized_points(&self) -> &DMatrix<f64>;

    fn points_from_distribution<D: Distribution>(&self, dist: &D) -> Result<DMatrix<f64>> {
        self.points_from_moments(dist.mean(), dist.cov_cholesky())
    }

    fn points_from_moments(
        &self,
        mean: &DVector<f64>,
        cov_cholesky: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>> {
        if mean.len() != self.input_dimension() {
            return Err(dim_error(
                "transformation mean",
                self.input_dimension().to_string(),
                mean.len().to_string(),
            ));
        }
        if cov_cholesky.nrows() != self.input_dimension()
            || cov_cholesky.ncols() != self.input_dimension()
        {
            return Err(dim_error(
                "transformation covariance cholesky",
                format!("{}x{}", self.input_dimension(), self.input_dimension()),
                format!("{}x{}", cov_cholesky.nrows(), cov_cholesky.ncols()),
            ));
        }
        let mut points = DMatrix::zeros(self.input_dimension(), self.number_of_points());
        for i in 0..self.number_of_points() {
            points.set_column(
                i,
                &(mean + cov_cholesky * self.normalized_points().column(i)),
            );
        }
        Ok(points)
    }

    fn mean(&self, points: &DMatrix<f64>) -> Result<DVector<f64>>;
    fn covariance(&self, points_x: &DMatrix<f64>, points_y: &DMatrix<f64>) -> Result<DMatrix<f64>>;
    fn variance(&self, points: &DVector<f64>) -> Result<f64>;
}

#[derive(Debug, Clone)]
pub struct UnscentedTransformation {
    dim_x: usize,
    dim_y: usize,
    alpha: f64,
    beta: f64,
    kappa: f64,
    normalized_points: DMatrix<f64>,
    weights_mean: DVector<f64>,
    weights_var: DVector<f64>,
}

impl UnscentedTransformation {
    pub fn new(dim_x: usize, dim_y: usize, alpha: f64, beta: f64, kappa: f64) -> Result<Self> {
        Self::with_uncertain(dim_x, dim_y, alpha, beta, kappa, vec![true; dim_x])
    }

    pub fn with_uncertain(
        dim_x: usize,
        dim_y: usize,
        alpha: f64,
        beta: f64,
        kappa: f64,
        consider_uncertain: Vec<bool>,
    ) -> Result<Self> {
        if alpha <= 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "alpha",
                value: alpha,
            });
        }
        if consider_uncertain.len() != dim_x {
            return Err(dim_error(
                "consider_uncertain",
                dim_x.to_string(),
                consider_uncertain.len().to_string(),
            ));
        }
        let uncertain_indices: Vec<_> = consider_uncertain
            .iter()
            .enumerate()
            .filter_map(|(i, uncertain)| uncertain.then_some(i))
            .collect();
        let n_uncertain = uncertain_indices.len();
        if n_uncertain == 0 {
            return Err(Error::Empty("uncertain variables"));
        }
        let num_points = 2 * n_uncertain + 1;
        let mut normalized_points = DMatrix::zeros(dim_x, num_points);
        let scale = alpha * ((n_uncertain as f64) + kappa).sqrt();
        for (j, idx) in uncertain_indices.into_iter().enumerate() {
            normalized_points[(idx, 1 + j)] = scale;
            normalized_points[(idx, 1 + n_uncertain + j)] = -scale;
        }

        let denom = alpha.powi(2) * ((n_uncertain as f64) + kappa);
        let off_weight = 1.0 / (2.0 * denom);
        let mut weights_mean = DVector::from_element(num_points, off_weight);
        weights_mean[0] = 1.0 - (n_uncertain as f64) / denom;
        let mut weights_var = weights_mean.clone();
        weights_var[0] += 1.0 - alpha.powi(2) + beta;

        Ok(Self {
            dim_x,
            dim_y,
            alpha,
            beta,
            kappa,
            normalized_points,
            weights_mean,
            weights_var,
        })
    }

    pub fn weights_mean(&self) -> &DVector<f64> {
        &self.weights_mean
    }

    pub fn weights_variance(&self) -> &DVector<f64> {
        &self.weights_var
    }

    pub fn parameters(&self) -> (f64, f64, f64) {
        (self.alpha, self.beta, self.kappa)
    }
}

impl PointTransformation for UnscentedTransformation {
    fn input_dimension(&self) -> usize {
        self.dim_x
    }

    fn output_dimension(&self) -> usize {
        self.dim_y
    }

    fn number_of_points(&self) -> usize {
        self.normalized_points.ncols()
    }

    fn normalized_points(&self) -> &DMatrix<f64> {
        &self.normalized_points
    }

    fn mean(&self, points: &DMatrix<f64>) -> Result<DVector<f64>> {
        require_points(points, self.dim_y, self.number_of_points(), "mean points")?;
        Ok(points * &self.weights_mean)
    }

    fn covariance(&self, points_x: &DMatrix<f64>, points_y: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        require_points(
            points_x,
            self.dim_x,
            self.number_of_points(),
            "covariance x points",
        )?;
        require_points(
            points_y,
            self.dim_y,
            self.number_of_points(),
            "covariance y points",
        )?;
        let mean_x = points_x * &self.weights_mean;
        let mean_y = points_y * &self.weights_mean;
        let mut covariance = DMatrix::zeros(self.dim_x, self.dim_y);
        for i in 0..self.number_of_points() {
            let dx = points_x.column(i) - &mean_x;
            let dy = points_y.column(i) - &mean_y;
            covariance += self.weights_var[i] * dx * dy.transpose();
        }
        Ok(covariance)
    }

    fn variance(&self, points: &DVector<f64>) -> Result<f64> {
        if points.len() != self.number_of_points() {
            return Err(dim_error(
                "variance points",
                self.number_of_points().to_string(),
                points.len().to_string(),
            ));
        }
        let mean = points.dot(&self.weights_mean);
        Ok(points
            .iter()
            .zip(self.weights_var.iter())
            .map(|(point, weight)| weight * (point - mean).powi(2))
            .sum())
    }
}

#[derive(Debug, Clone)]
pub struct StirlingFirstOrder {
    dim_x: usize,
    dim_y: usize,
    step_size: f64,
    num_uncertain: usize,
    normalized_points: DMatrix<f64>,
}

impl StirlingFirstOrder {
    pub fn new(dim_x: usize, dim_y: usize, step_size: f64) -> Result<Self> {
        Self::with_uncertain(dim_x, dim_y, step_size, vec![true; dim_x])
    }

    pub fn with_uncertain(
        dim_x: usize,
        dim_y: usize,
        step_size: f64,
        consider_uncertain: Vec<bool>,
    ) -> Result<Self> {
        if step_size <= 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "step_size",
                value: step_size,
            });
        }
        if consider_uncertain.len() != dim_x {
            return Err(dim_error(
                "consider_uncertain",
                dim_x.to_string(),
                consider_uncertain.len().to_string(),
            ));
        }
        let uncertain_indices: Vec<_> = consider_uncertain
            .iter()
            .enumerate()
            .filter_map(|(i, uncertain)| uncertain.then_some(i))
            .collect();
        let num_uncertain = uncertain_indices.len();
        if num_uncertain == 0 {
            return Err(Error::Empty("uncertain variables"));
        }
        let mut normalized_points = DMatrix::zeros(dim_x, 2 * num_uncertain + 1);
        for (j, idx) in uncertain_indices.into_iter().enumerate() {
            normalized_points[(idx, 1 + j)] = step_size;
            normalized_points[(idx, 1 + num_uncertain + j)] = -step_size;
        }
        Ok(Self {
            dim_x,
            dim_y,
            step_size,
            num_uncertain,
            normalized_points,
        })
    }
}

impl PointTransformation for StirlingFirstOrder {
    fn input_dimension(&self) -> usize {
        self.dim_x
    }

    fn output_dimension(&self) -> usize {
        self.dim_y
    }

    fn number_of_points(&self) -> usize {
        self.normalized_points.ncols()
    }

    fn normalized_points(&self) -> &DMatrix<f64> {
        &self.normalized_points
    }

    fn mean(&self, points: &DMatrix<f64>) -> Result<DVector<f64>> {
        require_points(points, self.dim_y, self.number_of_points(), "mean points")?;
        Ok(points.column(0).into_owned())
    }

    fn covariance(&self, points_x: &DMatrix<f64>, points_y: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        require_points(
            points_x,
            self.dim_x,
            self.number_of_points(),
            "covariance x points",
        )?;
        require_points(
            points_y,
            self.dim_y,
            self.number_of_points(),
            "covariance y points",
        )?;
        let mut covariance = DMatrix::zeros(self.dim_x, self.dim_y);
        for i in 1..=self.num_uncertain {
            let dx = points_x.column(i) - points_x.column(i + self.num_uncertain);
            let dy = points_y.column(i) - points_y.column(i + self.num_uncertain);
            covariance += dx * dy.transpose();
        }
        Ok(covariance / (4.0 * self.step_size.powi(2)))
    }

    fn variance(&self, points: &DVector<f64>) -> Result<f64> {
        if points.len() != self.number_of_points() {
            return Err(dim_error(
                "variance points",
                self.number_of_points().to_string(),
                points.len().to_string(),
            ));
        }
        let mut variance = 0.0;
        for i in 1..=self.num_uncertain {
            variance += (points[i] - points[i + self.num_uncertain]).powi(2);
        }
        Ok(variance / (4.0 * self.step_size.powi(2)))
    }
}

#[derive(Debug, Clone)]
pub struct ComposedQuadrature {
    dim_x: usize,
    dim_y: usize,
    normalized_points: DMatrix<f64>,
    roots: DMatrix<f64>,
    weights: DVector<f64>,
}

impl ComposedQuadrature {
    pub fn new(
        dim_x: usize,
        dim_y: usize,
        polynomial_family: &[PolynomialFamily],
        quadrature_order: &[usize],
    ) -> Result<Self> {
        if polynomial_family.len() != dim_x || quadrature_order.len() != dim_x {
            return Err(dim_error(
                "quadrature inputs",
                dim_x.to_string(),
                format!("{}/{}", polynomial_family.len(), quadrature_order.len()),
            ));
        }
        let rules: Result<Vec<_>> = polynomial_family
            .iter()
            .zip(quadrature_order.iter())
            .map(|(family, order)| QuadratureRule::new(*family, *order))
            .collect();
        let rules = rules?;
        let num_points = quadrature_order.iter().product();
        let mut normalized_points = DMatrix::zeros(dim_x, num_points);
        let mut roots = DMatrix::zeros(dim_x, num_points);
        let mut weights = DVector::from_element(num_points, 1.0);
        let mut num_combinations = num_points;
        for (i, rule) in rules.iter().enumerate() {
            let n = rule.roots.len();
            for j in 0..num_points {
                let index = (j % num_combinations) / (num_combinations / n);
                roots[(i, j)] = rule.roots[index];
                normalized_points[(i, j)] = rule.normalized_points[index];
                weights[j] *= rule.weights[index];
            }
            num_combinations /= n;
        }
        Ok(Self {
            dim_x,
            dim_y,
            normalized_points,
            roots,
            weights,
        })
    }

    pub fn with_uniform_order(
        dim_x: usize,
        dim_y: usize,
        polynomial_family: &[PolynomialFamily],
        quadrature_order: usize,
    ) -> Result<Self> {
        Self::new(
            dim_x,
            dim_y,
            polynomial_family,
            &vec![quadrature_order; dim_x],
        )
    }

    pub fn roots(&self) -> &DMatrix<f64> {
        &self.roots
    }

    pub fn weights(&self) -> &DVector<f64> {
        &self.weights
    }
}

impl PointTransformation for ComposedQuadrature {
    fn input_dimension(&self) -> usize {
        self.dim_x
    }

    fn output_dimension(&self) -> usize {
        self.dim_y
    }

    fn number_of_points(&self) -> usize {
        self.weights.len()
    }

    fn normalized_points(&self) -> &DMatrix<f64> {
        &self.normalized_points
    }

    fn mean(&self, points: &DMatrix<f64>) -> Result<DVector<f64>> {
        require_points(points, self.dim_y, self.number_of_points(), "mean points")?;
        Ok(points * &self.weights)
    }

    fn covariance(&self, points_x: &DMatrix<f64>, points_y: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        require_points(
            points_x,
            self.dim_x,
            self.number_of_points(),
            "covariance x points",
        )?;
        require_points(
            points_y,
            self.dim_y,
            self.number_of_points(),
            "covariance y points",
        )?;
        let mean_x = points_x * &self.weights;
        let mean_y = points_y * &self.weights;
        let mut covariance = DMatrix::zeros(self.dim_x, self.dim_y);
        for i in 0..self.number_of_points() {
            let dx = points_x.column(i) - &mean_x;
            let dy = points_y.column(i) - &mean_y;
            covariance += self.weights[i] * dx * dy.transpose();
        }
        Ok(covariance)
    }

    fn variance(&self, points: &DVector<f64>) -> Result<f64> {
        if points.len() != self.number_of_points() {
            return Err(dim_error(
                "variance points",
                self.number_of_points().to_string(),
                points.len().to_string(),
            ));
        }
        let mean = points.dot(&self.weights);
        Ok(points
            .iter()
            .zip(self.weights.iter())
            .map(|(point, weight)| weight * (point - mean).powi(2))
            .sum())
    }
}

#[derive(Debug, Clone)]
struct QuadratureRule {
    roots: DVector<f64>,
    normalized_points: DVector<f64>,
    weights: DVector<f64>,
}

impl QuadratureRule {
    fn new(family: PolynomialFamily, order: usize) -> Result<Self> {
        if order == 0 {
            return Err(Error::NonPositiveParameter {
                name: "quadrature_order",
                value: 0.0,
            });
        }
        match family {
            PolynomialFamily::Hermite => Ok(gauss_hermite_probability(order)),
            PolynomialFamily::Legendre => Ok(gauss_legendre_probability(order)),
            PolynomialFamily::MomentOnly => Err(Error::UnsupportedPolynomialFamily("MomentOnly")),
        }
    }
}

fn gauss_hermite_probability(order: usize) -> QuadratureRule {
    let mut jacobi = DMatrix::zeros(order, order);
    for i in 1..order {
        let value = (i as f64).sqrt();
        jacobi[(i - 1, i)] = value;
        jacobi[(i, i - 1)] = value;
    }
    eigen_rule(jacobi, |root| root)
}

fn gauss_legendre_probability(order: usize) -> QuadratureRule {
    let mut jacobi = DMatrix::zeros(order, order);
    for i in 1..order {
        let i = i as f64;
        let value = i / (4.0 * i.powi(2) - 1.0).sqrt();
        jacobi[(i as usize - 1, i as usize)] = value;
        jacobi[(i as usize, i as usize - 1)] = value;
    }
    eigen_rule(jacobi, |root| 3.0_f64.sqrt() * root)
}

fn eigen_rule(jacobi: DMatrix<f64>, normalize: impl Fn(f64) -> f64) -> QuadratureRule {
    let eig = SymmetricEigen::new(jacobi);
    let mut entries: Vec<_> = (0..eig.eigenvalues.len())
        .map(|i| {
            let root = eig.eigenvalues[i];
            let weight = eig.eigenvectors[(0, i)].powi(2);
            (root, normalize(root), weight)
        })
        .collect();
    entries.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite root"));
    QuadratureRule {
        roots: DVector::from_iterator(entries.len(), entries.iter().map(|entry| entry.0)),
        normalized_points: DVector::from_iterator(
            entries.len(),
            entries.iter().map(|entry| entry.1),
        ),
        weights: DVector::from_iterator(entries.len(), entries.iter().map(|entry| entry.2)),
    }
}

fn require_points(
    points: &DMatrix<f64>,
    rows: usize,
    cols: usize,
    context: &'static str,
) -> Result<()> {
    if points.nrows() != rows || points.ncols() != cols {
        return Err(dim_error(
            context,
            format!("{rows}x{cols}"),
            format!("{}x{}", points.nrows(), points.ncols()),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unscented_recovers_identity_moments() {
        let ut = UnscentedTransformation::new(2, 2, 1.0, 2.0, 0.0).unwrap();
        let mean = DVector::from_vec(vec![1.0, -2.0]);
        let chol = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.3, 1.0]);
        let points = ut.points_from_moments(&mean, &chol).unwrap();
        let recovered_mean = ut.mean(&points).unwrap();
        let recovered_cov = ut.covariance(&points, &points).unwrap();
        assert!((&recovered_mean - mean).amax() < 1e-12);
        assert!((recovered_cov - &chol * chol.transpose()).amax() < 1e-12);
    }

    #[test]
    fn quadrature_weights_sum_to_one() {
        let quad = ComposedQuadrature::with_uniform_order(
            2,
            2,
            &[PolynomialFamily::Hermite, PolynomialFamily::Legendre],
            3,
        )
        .unwrap();
        assert!((quad.weights().sum() - 1.0).abs() < 1e-12);
    }
}
