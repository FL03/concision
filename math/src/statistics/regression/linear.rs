/*
    Appellation: linear <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Regression;
use crate::statistics::{covariance, deviation, mean, variance};

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct LinearRegression {
    intercept: f64,
    slope: f64,
}

impl LinearRegression {
    pub fn new(x: &[f64], y: &[f64]) -> LinearRegression {
        let (x_mean, y_mean) = (mean(x), mean(y));
        let (x_dev, y_dev) = (deviation(x), deviation(y));
        let slope = covariance(&x_dev, &y_dev) / variance(&x_dev);
        let intercept = y_mean - slope * x_mean;
        LinearRegression { intercept, slope }
    }

    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&x| self.slope * x + self.intercept).collect()
    }

    pub fn slope(&self) -> f64 {
        self.slope
    }
}

impl Regression for LinearRegression {
    type Item = f64;

    fn fit(&mut self, args: &[Self::Item], target: &[Self::Item]) {
        let (x_mean, y_mean) = (mean(args), mean(target));
        let (x_dev, y_dev) = (deviation(args), deviation(target));
        self.slope = covariance(&x_dev, &y_dev) / variance(&x_dev);
        self.intercept = y_mean - self.slope * x_mean;
    }

    fn predict(&self, args: &[Self::Item]) -> Vec<Self::Item> {
        args.iter().map(|&x| self.slope * x + self.intercept).collect()
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
