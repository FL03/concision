/*
    Appellation: statistics <module>
    Contrib: FL03 <jo3mccain@icloud.com>
    Description: ... Summary ...
*/
pub use self::deviation::*;

pub mod regression;

mod deviation;

/// Covariance is the average of the products of the deviations from the mean.
pub fn covariance(x: Vec<f64>, y: Vec<f64>) -> f64 {
    x.iter().zip(y.iter()).map(|(&x, &y)| x * y).sum::<f64>() / x.len() as f64
}
/// Deviation is the distance from the mean.
pub fn deviation(x: &[f64], mean: f64) -> Vec<f64> {
    x.iter().map(|&x| x - mean).collect()
}
/// Mean is the average of the data.
pub fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}
/// Variance is the average of the squared deviations from the mean.
pub fn variance(x: Vec<f64>) -> f64 {
    let mean = mean(&x);
    let dev = deviation(&x, mean);
    dev.iter().map(|&x| x * x).sum::<f64>() / dev.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_covariance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(covariance(x, y), 2.0);
    }

    #[test]
    fn test_deviation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = mean(&x);
        assert_eq!(deviation(&x, mean), vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_mean() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean(&x), 3.0);
    }

    #[test]
    fn test_variance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(variance(x), 2.0);
    }
}
