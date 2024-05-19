/*
    Appellation: lecun <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use num::Float;
use rand_distr::{Distribution, Normal, NormalError, StandardNormal};

/// Create a [Normal](rand_distr::Normal) distribution with a standard deviation of sqrt(1/n)
/// where n is the number of inputs.
pub fn lecun_normal<F>(n: usize) -> Result<Normal<F>, NormalError>
where
    F: Float,
    StandardNormal: Distribution<F>,
{
    let std_dev = F::from(n).unwrap().recip().sqrt();
    Normal::new(F::zero(), std_dev)
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LecunNormal {
    n: usize,
}

impl LecunNormal {
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    pub fn distr<F>(&self) -> Result<Normal<F>, NormalError>
    where
        F: Float,
        StandardNormal: Distribution<F>,
    {
        lecun_normal(self.n)
    }

    pub fn std<F>(&self) -> F
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
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> F {
        self.distr().unwrap().sample(rng)
    }
}
