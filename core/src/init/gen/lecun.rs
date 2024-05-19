/*
    Appellation: lecun <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use num::Float;
use rand::Rng;
use rand_distr::{Distribution, Normal, NormalError, StandardNormal};

/// [LecunNormal] is a truncated [normal](rand_distr::Normal) distribution centered at 0
/// with a standard deviation that is calculated as `σ = sqrt(1/n_in)`
/// where `n_in` is the number of input units.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LecunNormal {
    n: usize,
}

impl LecunNormal {
    pub fn new(n: usize) -> Self {
        Self { n }
    }
    /// Create a [normal](rand_distr::Normal) [distribution](Distribution) centered at 0;
    /// See [Self::std_dev] for the standard deviation calculations.
    pub fn distr<F>(&self) -> Result<Normal<F>, NormalError>
    where
        F: Float,
        StandardNormal: Distribution<F>,
    {
        Normal::new(F::zero(), self.std_dev())
    }
    /// Calculate the standard deviation (`σ`) of the distribution.
    /// This is done by computing the root of the reciprocal of the number of inputs
    ///
    /// Symbolically: `σ = sqrt(1/n)`
    pub fn std_dev<F>(&self) -> F
    where
        F: Float,
    {
        F::from(self.n).unwrap().recip().sqrt()
    }
}

impl<F> Distribution<F> for LecunNormal
where
    F: Float,
    StandardNormal: Distribution<F>,
{
    fn sample<R>(&self, rng: &mut R) -> F
    where
        R: Rng + ?Sized,
    {
        self.distr().unwrap().sample(rng)
    }
}
