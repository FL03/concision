/*
    Appellation: statistics <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # statistics
pub use self::{deviation::*, utils::*};

pub mod regression;

pub(crate) mod deviation;

pub trait Statistics<T>
where
    T: num::Float + std::iter::Sum,
    Self: Clone + IntoIterator<Item = T>,
{
    fn covariance(&self, other: &Self) -> T {
        let dx = self.deviation();
        let dy = other.deviation();
        dx.iter().zip(dy.iter()).map(|(&x, &y)| x * y).sum::<T>() / T::from(dx.len()).unwrap()
    }

    fn deviation(&self) -> Vec<T> {
        let mean = self.mean();
        self.clone().into_iter().map(|x| x - mean).collect()
    }
    fn len(&self) -> usize {
        Vec::from_iter(self.clone().into_iter()).len()
    }
    /// [Statistics::mean] calculates the mean or average of the data
    fn mean(&self) -> T {
        self.clone().into_iter().sum::<T>() / T::from(self.len()).unwrap()
    }
    /// [Statistics::std] calculates the standard deviation of the data
    fn std(&self) -> T {
        let mean = self.mean();
        let mut res = self
            .clone()
            .into_iter()
            .map(|x| (x - mean).powi(2))
            .sum::<T>();
        res = res / T::from(self.len()).unwrap();
        res.sqrt()
    }

    fn variance(&self) -> T {
        let dev = self.deviation();
        dev.iter().map(|&x| x * x).sum::<T>() / T::from(dev.len()).unwrap()
    }
}

impl<T> Statistics<T> for Vec<T> where T: num::Float + std::iter::Sum {}

impl<T> Statistics<T> for ndarray::Array1<T> where T: num::Float + std::iter::Sum {}

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
