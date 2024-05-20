/*
    Appellation: trunc <distr>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use num::traits::Float;
use rand::Rng;
use rand_distr::{Distribution, Normal, NormalError, StandardNormal};

/// A truncated normal distribution is similar to a [normal](rand_distr::Normal) [distribution](rand_distr::Distribution), however,
/// any generated value over two standard deviations from the mean is discarded and re-generated.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TruncatedNormal<F>
where
    StandardNormal: Distribution<F>,
{
    mean: F,
    std: F,
}

impl<F> TruncatedNormal<F>
where
    F: Float,
    StandardNormal: Distribution<F>,
{
    /// Create a new truncated normal distribution with a given mean and standard deviation
    pub fn new(mean: F, std: F) -> Result<Self, NormalError> {
        Ok(Self { mean, std })
    }

    pub(crate) fn boundary(&self) -> F {
        self.mean() + self.std_dev() * F::from(2).unwrap()
    }

    pub(crate) fn score(&self, x: F) -> F {
        self.mean() - self.std_dev() * x
    }

    pub fn distr(&self) -> Normal<F> {
        Normal::new(self.mean(), self.std_dev()).unwrap()
    }

    pub fn mean(&self) -> F {
        self.mean
    }

    pub fn std_dev(&self) -> F {
        self.std
    }
}

impl<F> Distribution<F> for TruncatedNormal<F>
where
    F: Float,
    StandardNormal: Distribution<F>,
{
    fn sample<R>(&self, rng: &mut R) -> F
    where
        R: Rng + ?Sized,
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

impl<F> From<Normal<F>> for TruncatedNormal<F>
where
    F: Float,
    StandardNormal: Distribution<F>,
{
    fn from(normal: Normal<F>) -> Self {
        Self {
            mean: normal.mean(),
            std: normal.std_dev(),
        }
    }
}
