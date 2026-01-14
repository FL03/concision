/*
    Appellation: trunc <distr>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use num_traits::Float;
use rand::{Rng, RngCore};
use rand_distr::{Distribution, Normal, StandardNormal};

/// The [`TruncatedNormal`] distribution is similar to the [`StandardNormal`] distribution,
/// differing in that is computes a boundary equal to two standard deviations from the mean
/// ($`\mu`$). More formally, the boundary is defined as:
///
/// ```math
/// \text{boundary} = \mu + 2\sigma
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TruncatedNormal<T>
where
    StandardNormal: Distribution<T>,
{
    pub(crate) mean: T,
    pub(crate) std: T,
}

impl<T> TruncatedNormal<T>
where
    T: Copy,
    StandardNormal: Distribution<T>,
{
    /// create a new [`TruncatedNormal`] distribution with the given mean and standard
    /// deviation; both of which are type `T`.
    pub const fn new(mean: T, std: T) -> crate::Result<Self> {
        Ok(Self { mean, std })
    }
    /// returns a copy of the mean for the distribution
    pub const fn mean(&self) -> T {
        self.mean
    }
    /// returns a copy of the standard deviation for the distribution
    pub const fn std_dev(&self) -> T {
        self.std
    }
    /// compute the boundary of the truncated normal distribution
    /// which is two standard deviations from the mean:
    /// ```math
    /// \text{boundary}=\mu + 2\sigma
    /// ```
    pub fn boundary(&self) -> T
    where
        T: Float + core::ops::Mul<Output = T> + core::ops::Add<Output = T>,
    {
        self.mean() + self.std_dev() * T::from(2).unwrap()
    }
    /// returns a new [`Normal`] distribution instance created from the current mean and
    /// standard deviation.
    pub fn distr(&self) -> Normal<T>
    where
        T: Float,
    {
        Normal::new(self.mean(), self.std_dev()).unwrap()
    }
    /// compute the score of the distribution at point `x`. The score is calculated by
    /// subtracing a scaled standard deviation from the mean:
    /// ```math
    /// \text{score}(x)=\mu-\sigma\cdot{x}
    /// ```
    ///
    /// where $\mu$ is the mean and $\sigma$ is the standard deviation.
    pub fn score(&self, x: T) -> T
    where
        T: core::ops::Mul<Output = T> + core::ops::Sub<Output = T>,
    {
        self.mean() - self.std_dev() * x
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
// impl<T> Distribution<T> for TruncatedNormal<T>
// where
//     T: Copy
//         + PartialOrd
//         + FromPrimitive
//         + core::ops::Add<Output = T>
//         + core::ops::Mul<Output = T>
//         + core::ops::Sub<Output = T>
//         + core::ops::Neg<Output = T>,
//     StandardNormal: Distribution<T>,
// {
//     fn sample<R>(&self, rng: &mut R) -> T
//     where
//         R: RngCore + ?Sized,
//     {
//         let bnd = self.boundary();
//         let mut x = self.score(rng.sample(StandardNormal));
//         // if x is outside of the boundary, re-sample
//         while x < -bnd || x > bnd {
//             x = self.score(rng.sample(StandardNormal));
//         }
//         x
//     }
// }

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
