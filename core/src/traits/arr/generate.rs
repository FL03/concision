/*
   Appellation: generate <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
// #![cfg(feature = "rand")]
use core::ops::Neg;
use ndarray::{Array, Dimension, IntoDimension, Ix2};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_rand::rand_distr::uniform::{SampleUniform, Uniform};
use ndarray_rand::rand_distr::{Bernoulli, BernoulliError, Distribution, StandardNormal};
use ndarray_rand::RandomExt;
use num::traits::real::Real;
use num::traits::Float;

pub trait GenerateRandom<T = f64, D = Ix2>: Sized
where
    D: Dimension,
{
    fn rand<IdS>(dim: impl IntoDimension<Dim = D>, distr: IdS) -> Self
    where
        IdS: Distribution<T>;

    fn rand_using<IdS, R: ?Sized>(
        dim: impl IntoDimension<Dim = D>,
        distr: IdS,
        rng: &mut R,
    ) -> Self
    where
        IdS: Distribution<T>,
        R: Rng;

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
    fn rand<IdS>(dim: impl IntoDimension<Dim = D>, distr: IdS) -> Self
    where
        IdS: Distribution<T>,
    {
        Self::random(dim.into_dimension(), distr)
    }

    fn rand_using<IdS, R: ?Sized>(dim: impl IntoDimension<Dim = D>, distr: IdS, rng: &mut R) -> Self
    where
        IdS: Distribution<T>,
        R: Rng,
    {
        Self::random_using(dim.into_dimension(), distr, rng)
    }
}
