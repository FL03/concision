/*
    Appellation: specs <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

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
