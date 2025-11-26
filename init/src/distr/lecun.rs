/*
    Appellation: lecun <distr>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::TruncatedNormal;
use num_traits::Float;
use rand::RngCore;
use rand_distr::{Distribution, StandardNormal};

/// [LecunNormal] is a truncated [normal](rand_distr::Normal) distribution centered at 0
/// with a standard deviation that is calculated as:
///
/// $$
/// \sigma = {n_{in}}^{-\frac{1}{2}}
/// $$
///
/// where $`n_{in}`$ is the number of input units.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LecunNormal {
    n: usize,
}

impl LecunNormal {
    pub const fn new(n: usize) -> Self {
        Self { n }
    }
    /// Create a [truncated normal](TruncatedNormal) [distribution](Distribution) centered at 0;
    /// See [Self::std_dev] for the standard deviation calculations.
    pub fn distr<F>(&self) -> crate::InitResult<TruncatedNormal<F>>
    where
        F: Float,
        StandardNormal: Distribution<F>,
    {
        TruncatedNormal::new(F::zero(), self.std_dev())
    }
    /// Calculate the standard deviation ($`\sigma`$) of the distribution.
    /// This is done by computing the root of the reciprocal of the number of inputs
    /// ($`n_{in}`$) as follows:
    ///
    /// ```math
    /// \sigma = {n_{in}}^{-\frac{1}{2}}
    /// ```
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
        R: RngCore + ?Sized,
    {
        self.distr().expect("NormalError").sample(rng)
    }
}
