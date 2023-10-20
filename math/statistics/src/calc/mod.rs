/*
    Appellation: statistics <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # statistics
pub use self::{deviation::*, utils::*};

pub(crate) mod deviation;

pub(crate) mod utils {
    use std::iter::Sum;

    /// Covariance is the average of the products of the deviations from the mean.
    pub fn covariance<T: num::Float + Sum>(x: &[T], y: &[T]) -> T {
        let dx = deviation(&x);
        let dy = deviation(&y);
        dx.iter().zip(dy.iter()).map(|(&x, &y)| x * y).sum::<T>() / T::from(dx.len()).unwrap()
    }
    /// Deviation is the distance from the mean.
    pub fn deviation<T: num::Float + Sum>(x: &[T]) -> Vec<T> {
        let mean = mean(x);
        x.iter().map(|&x| x - mean).collect()
    }
    /// Mean is the average of the data.
    pub fn mean<T: num::Float + Sum>(x: &[T]) -> T {
        x.iter().cloned().sum::<T>() / T::from(x.len()).unwrap()
    }
    /// Variance is the average of the squared deviations from the mean.
    pub fn variance<T: num::Float + Sum>(x: &[T]) -> T {
        let dev = deviation(&x);
        dev.iter().map(|&x| x * x).sum::<T>() / T::from(dev.len()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Statistics;
    use rand::Rng;

    fn random_vec() -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut v = Vec::new();
        for _ in 0..100 {
            v.push(rng.gen_range(0.0..25.0));
        }
        v
    }

    #[test]
    fn test_statistics() {
        let x = random_vec();
        let y = random_vec();
        assert_eq!(covariance(&x, &y), x.covariance(&y));
        assert_eq!(deviation(&x), x.deviation());
        assert_eq!(mean(&x), x.mean());
        assert_eq!(variance(&x), x.variance());
    }

    #[test]
    fn test_covariance() {
        let x: Vec<f64> = (1..=5).map(|i| i as f64).collect();
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(covariance(&x, &y), 2.0);
        assert_eq!(x.covariance(&y), 2.0);
    }

    #[test]
    fn test_deviation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(deviation(&x), vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_mean() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean(&x), 3.0);
    }

    #[test]
    fn test_variance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(variance(&x), 2.0);
    }
}
