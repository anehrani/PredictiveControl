#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial {
    coefficients: Vec<f64>,
}

impl Polynomial {
    pub fn new(coefficients: Vec<f64>) -> Self {
        Self { coefficients }
    }

    pub fn zero() -> Self {
        Self {
            coefficients: Vec::new(),
        }
    }

    pub fn constant(value: f64) -> Self {
        Self {
            coefficients: vec![value],
        }
    }

    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    pub fn coefficient(&self, index: usize) -> Option<f64> {
        self.coefficients.get(index).copied()
    }

    pub fn num_coefficients(&self) -> usize {
        self.coefficients.len()
    }

    pub fn evaluate(&self, x: f64) -> f64 {
        self.coefficients
            .iter()
            .rev()
            .fold(0.0, |acc, coeff| acc * x + coeff)
    }

    pub fn gradient(&self, x: f64) -> f64 {
        self.coefficients
            .iter()
            .enumerate()
            .skip(1)
            .map(|(order, coeff)| order as f64 * coeff * x.powi(order as i32 - 1))
            .sum()
    }

    pub fn hessian(&self, x: f64) -> f64 {
        self.coefficients
            .iter()
            .enumerate()
            .skip(2)
            .map(|(order, coeff)| {
                order as f64 * (order as f64 - 1.0) * coeff * x.powi(order as i32 - 2)
            })
            .sum()
    }

    pub fn add_polynomial(&mut self, other: &Self) {
        self.resize_to_match(other);
        for (index, coeff) in other.coefficients.iter().copied().enumerate() {
            self.coefficients[index] += coeff;
        }
    }

    pub fn subtract_polynomial(&mut self, other: &Self) {
        self.resize_to_match(other);
        for (index, coeff) in other.coefficients.iter().copied().enumerate() {
            self.coefficients[index] -= coeff;
        }
    }

    pub fn multiply_polynomial(&mut self, other: &Self) {
        if self.coefficients.is_empty() || other.coefficients.is_empty() {
            self.coefficients.clear();
            return;
        }
        let mut coefficients = vec![0.0; self.coefficients.len() + other.coefficients.len() - 1];
        for (i, left) in self.coefficients.iter().copied().enumerate() {
            for (j, right) in other.coefficients.iter().copied().enumerate() {
                coefficients[i + j] += left * right;
            }
        }
        self.coefficients = coefficients;
    }

    pub fn multiply_scalar(&mut self, factor: f64) {
        for coeff in &mut self.coefficients {
            *coeff *= factor;
        }
    }

    pub fn scaled(mut self, factor: f64) -> Self {
        self.multiply_scalar(factor);
        self
    }

    fn resize_to_match(&mut self, other: &Self) {
        if other.coefficients.len() > self.coefficients.len() {
            self.coefficients.resize(other.coefficients.len(), 0.0);
        }
    }
}

#[derive(Debug, Clone)]
pub struct HermitePolynomialGenerator {
    polynomials: Vec<Polynomial>,
    squared_norms: Vec<f64>,
}

impl HermitePolynomialGenerator {
    pub fn new(max_order: usize) -> Self {
        let mut polynomials = Vec::with_capacity(max_order + 1);
        polynomials.push(Polynomial::constant(1.0));
        if max_order >= 1 {
            polynomials.push(Polynomial::new(vec![0.0, 1.0]));
        }
        for order in 2..=max_order {
            let mut next = multiply_by_x(&polynomials[order - 1]);
            let lower = polynomials[order - 2].clone().scaled((order - 1) as f64);
            next.subtract_polynomial(&lower);
            polynomials.push(next);
        }
        let squared_norms = (0..=max_order)
            .map(|order| factorial(order) as f64)
            .collect();
        Self {
            polynomials,
            squared_norms,
        }
    }

    pub fn polynomial(&self, order: usize) -> Option<&Polynomial> {
        self.polynomials.get(order)
    }

    pub fn squared_norm(&self, order: usize) -> Option<f64> {
        self.squared_norms.get(order).copied()
    }

    pub fn maximum_order(&self) -> usize {
        self.polynomials.len().saturating_sub(1)
    }
}

#[derive(Debug, Clone)]
pub struct LegendrePolynomialGenerator {
    polynomials: Vec<Polynomial>,
    squared_norms: Vec<f64>,
}

impl LegendrePolynomialGenerator {
    pub fn new(max_order: usize) -> Self {
        let mut polynomials = Vec::with_capacity(max_order + 1);
        polynomials.push(Polynomial::constant(1.0));
        if max_order >= 1 {
            polynomials.push(Polynomial::new(vec![0.0, 1.0]));
        }
        for order in 2..=max_order {
            let mut next = multiply_by_x(&polynomials[order - 1])
                .scaled((2 * order - 1) as f64 / order as f64);
            let lower = polynomials[order - 2]
                .clone()
                .scaled((order - 1) as f64 / order as f64);
            next.subtract_polynomial(&lower);
            polynomials.push(next);
        }
        let squared_norms = (0..=max_order)
            .map(|order| 1.0 / (2.0 * order as f64 + 1.0))
            .collect();
        Self {
            polynomials,
            squared_norms,
        }
    }

    pub fn polynomial(&self, order: usize) -> Option<&Polynomial> {
        self.polynomials.get(order)
    }

    pub fn squared_norm(&self, order: usize) -> Option<f64> {
        self.squared_norms.get(order).copied()
    }

    pub fn maximum_order(&self) -> usize {
        self.polynomials.len().saturating_sub(1)
    }
}

fn multiply_by_x(poly: &Polynomial) -> Polynomial {
    let mut coefficients = Vec::with_capacity(poly.num_coefficients() + 1);
    coefficients.push(0.0);
    coefficients.extend_from_slice(poly.coefficients());
    Polynomial::new(coefficients)
}

fn factorial(value: usize) -> usize {
    (1..=value).product::<usize>().max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial_evaluates_derivatives_and_arithmetic() {
        let poly = Polynomial::new(vec![1.0, -3.0, 2.0]);
        assert!((poly.evaluate(3.0) - 10.0).abs() < 1e-12);
        assert!((poly.gradient(3.0) - 9.0).abs() < 1e-12);
        assert!((poly.hessian(3.0) - 4.0).abs() < 1e-12);

        let mut product = Polynomial::new(vec![1.0, 1.0]);
        product.multiply_polynomial(&Polynomial::new(vec![1.0, -1.0]));
        assert_eq!(product.coefficients(), &[1.0, 0.0, -1.0]);
    }

    #[test]
    fn hermite_generator_matches_probabilists_polynomials() {
        let generator = HermitePolynomialGenerator::new(3);
        assert_eq!(generator.maximum_order(), 3);
        assert_eq!(
            generator.polynomial(2).unwrap().coefficients(),
            &[-1.0, 0.0, 1.0]
        );
        assert_eq!(
            generator.polynomial(3).unwrap().coefficients(),
            &[0.0, -3.0, 0.0, 1.0]
        );
        assert_eq!(generator.squared_norm(3), Some(6.0));
    }

    #[test]
    fn legendre_generator_matches_classical_polynomials() {
        let generator = LegendrePolynomialGenerator::new(3);
        assert_eq!(generator.maximum_order(), 3);
        assert_eq!(
            generator.polynomial(2).unwrap().coefficients(),
            &[-0.5, 0.0, 1.5]
        );
        assert_eq!(
            generator.polynomial(3).unwrap().coefficients(),
            &[0.0, -1.5, 0.0, 2.5]
        );
        assert_eq!(generator.squared_norm(3), Some(1.0 / 7.0));
    }
}
