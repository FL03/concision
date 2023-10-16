/*
    Appellation: statistics <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # statistics
pub use self::{deviation::*, utils::*};

pub mod regression;

pub(crate) mod deviation;

pub trait Statistics<T: num::Float + std::iter::Sum> {}

pub trait Mean<T: num::Float + std::iter::Sum>:
    Clone + IntoIterator<Item = T> + ExactSizeIterator
{
    fn mean(&self) -> T {
        self.clone().into_iter().sum::<T>() / T::from(self.len()).unwrap()
    }
}

pub(crate) mod utils {
    use std::iter::Sum;

    /// Covariance is the average of the products of the deviations from the mean.
    pub fn covariance<T: num::Float + Sum>(x: Vec<T>, y: Vec<T>) -> T {
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

    #[test]
    fn test_covariance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(covariance(x, y), 2.0);
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
