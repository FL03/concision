/*
    Appellation: trunc <distr>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use num::traits::Float;
use rand::{Rng, RngCore};
use rand_distr::{Distribution, Normal, StandardNormal};

/// A truncated normal distribution is similar to a [normal](rand_distr::Normal) [distribution](rand_distr::Distribution), however,
/// any generated value over two standard deviations from the mean is discarded and re-generated.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TruncatedNormal<T>
where
    StandardNormal: Distribution<T>,
{
    mean: T,
    std: T,
}

impl<T> TruncatedNormal<T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    /// create a new [`TruncatedNormal`] distribution with the given mean and standard
    /// deviation; both of which are type `T`.
    pub const fn new(mean: T, std: T) -> crate::Result<Self> {
        Ok(Self { mean, std })
    }
    /// compute the boundary of the truncated normal distribution
    /// which is two standard deviations from the mean:
    /// $$
    /// \text{boundary} = \mu + 2\sigma
    /// $$
    pub(crate) fn boundary(&self) -> T {
        self.mean() + self.std_dev() * T::from(2).unwrap()
    }

    pub(crate) fn score(&self, x: T) -> T {
        self.mean() - self.std_dev() * x
    }

    pub fn distr(&self) -> Normal<T> {
        Normal::new(self.mean(), self.std_dev()).unwrap()
    }

    pub fn mean(&self) -> T {
        self.mean
    }

    pub fn std_dev(&self) -> T {
        self.std
    }
}

impl<T> Distribution<T> for TruncatedNormal<T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    fn sample<R>(&self, rng: &mut R) -> T
    where
        R: RngCore + ?Sized,
    {
        let bnd = self.boundary();
        let mut x = self.score(rng.sample(StandardNormal));
        // if x is outside of the boundary, re-sample
        while x < -bnd || x > bnd {
            x = self.score(rng.sample(StandardNormal));
        }
        x
    }
}

impl<T> From<Normal<T>> for TruncatedNormal<T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    fn from(normal: Normal<T>) -> Self {
        Self {
            mean: normal.mean(),
            std: normal.std_dev(),
        }
    }
}
