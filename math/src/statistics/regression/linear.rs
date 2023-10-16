/*
    Appellation: linear <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::statistics::{covariance, deviation, mean, variance};

pub struct LinearRegression {
    pub slope: f64,
    pub intercept: f64,
}

impl LinearRegression {
    pub fn new(x: &[f64], y: &[f64]) -> LinearRegression {
        let (x_mean, y_mean) = (mean(x), mean(y));
        let (x_dev, y_dev) = (deviation(x), deviation(y));
        let slope = covariance(x_dev.clone(), y_dev) / variance(&x_dev);
        let intercept = y_mean - slope * x_mean;
        LinearRegression { slope, intercept }
    }

    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&x| self.slope * x + self.intercept).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let lr = LinearRegression::new(&x, &y);
        assert_eq!(lr.slope, 1.0);
        assert_eq!(lr.intercept, 0.0);
        assert_eq!(lr.predict(&x), y);
    }
}
