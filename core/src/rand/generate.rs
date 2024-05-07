/*
   Appellation: generate <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use core::ops::Neg;
use ndarray::*;
use ndrand::rand::rngs::StdRng;
use ndrand::rand::{Rng, SeedableRng};
use ndrand::rand_distr::uniform::{SampleUniform, Uniform};
use ndrand::rand_distr::{Bernoulli, BernoulliError, Distribution, StandardNormal};
use ndrand::RandomExt;
use num::traits::real::Real;
use num::traits::Float;

pub trait GenerateRandom<T = f64, D = Ix2>: Sized
where
    D: Dimension,
{
    fn rand<Sh, IdS>(dim: Sh, distr: IdS) -> Self
    where
        IdS: Distribution<T>,
        Sh: ShapeBuilder<Dim = D>;

    fn rand_using<Sh, IdS, R: ?Sized>(dim: Sh, distr: IdS, rng: &mut R) -> Self
    where
        IdS: Distribution<T>,
        R: Rng,
        Sh: ShapeBuilder<Dim = D>;

    fn bernoulli(dim: impl IntoDimension<Dim = D>, p: Option<f64>) -> Result<Self, BernoulliError>
    where
        Bernoulli: Distribution<T>,
    {
        let dist = Bernoulli::new(p.unwrap_or(0.5))?;
        Ok(Self::rand(dim.into_dimension(), dist))
    }

    fn stdnorm(dim: impl IntoDimension<Dim = D>) -> Self
    where
        StandardNormal: Distribution<T>,
    {
        Self::rand(dim, StandardNormal)
    }

    fn normal_from_key<R: ?Sized>(key: u64, dim: impl IntoDimension<Dim = D>) -> Self
    where
        StandardNormal: Distribution<T>,
        R: Rng,
    {
        Self::rand_using(
            dim.into_dimension(),
            StandardNormal,
            &mut StdRng::seed_from_u64(key),
        )
    }

    fn uniform(axis: usize, dim: impl IntoDimension<Dim = D>) -> Self
    where
        T: Real + SampleUniform,
    {
        let dim = dim.into_dimension();
        let dk = T::from(dim[axis]).unwrap().recip().sqrt();
        Self::uniform_between(dk, dim)
    }

    fn uniform_between(dk: T, dim: impl IntoDimension<Dim = D>) -> Self
    where
        T: Copy + Neg<Output = T> + SampleUniform,
    {
        Self::rand(dim, Uniform::new(-dk, dk))
    }
}

impl<T, D> GenerateRandom<T, D> for Array<T, D>
where
    T: Float + SampleUniform,
    D: Dimension,
    StandardNormal: Distribution<T>,
{
    fn rand<Sh, Dtr>(dim: Sh, distr: Dtr) -> Self
    where
        Dtr: Distribution<T>,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::random(dim, distr)
    }

    fn rand_using<Sh, Dtr, R>(dim: Sh, distr: Dtr, rng: &mut R) -> Self
    where
        Dtr: Distribution<T>,
        R: Rng + ?Sized,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::random_using(dim, distr, rng)
    }
}
