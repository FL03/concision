/*
    Appellation: initialize <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::init::distr::*;

use core::ops::Neg;
use ndarray::{ArrayBase, DataOwned, Dimension, RawData, ShapeBuilder};
use ndarray_rand::RandomExt;
use num::complex::ComplexDistribution;
use num::traits::Float;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::uniform::{SampleUniform, Uniform};
use rand_distr::{Bernoulli, BernoulliError, Distribution, Normal, NormalError, StandardNormal};

/// [Initialize]
///
/// This trait provides the base methods required for initializing an [ndarray](ndarray::ArrayBase) with random values.
/// [Initialize] is similar to [RandomExt](ndarray_rand::RandomExt), however, it focuses on flexibility while implementing additional
/// features geared towards machine-learning models; such as lecun_normal initialization.
pub trait Initialize<A, D>
where
    D: Dimension,
{
    type Data: RawData<Elem = A>;

    fn rand<Sh, Ds>(shape: Sh, distr: Ds) -> Self
    where
        Ds: Clone + Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned;

    fn rand_with<Sh, Ds, R>(shape: Sh, distr: Ds, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
        Ds: Clone + Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned;
}

/// This trait extends the [Initialize] trait with methods for generating random arrays from various distributions.
pub trait InitializeExt<A, S, D>: Initialize<A, D, Data = S> + Sized
where
    A: Clone,
    D: Dimension,
    S: RawData<Elem = A>,
{
    fn bernoulli<Sh>(shape: Sh, p: f64) -> Result<Self, BernoulliError>
    where
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        Bernoulli: Distribution<A>,
    {
        let dist = Bernoulli::new(p)?;
        Ok(Self::rand(shape, dist))
    }
    /// Initialize the object according to the Lecun Initialization scheme.
    /// LecunNormal distributions are truncated [Normal](rand_distr::Normal)
    /// distributions centered at 0 with a standard deviation equal to the
    /// square root of the reciprocal of the number of inputs.
    fn lecun_normal<Sh>(shape: Sh, n: usize) -> Self
    where
        A: Float,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        StandardNormal: Distribution<A>,
    {
        let distr = LecunNormal::new(n);
        Self::rand(shape, distr)
    }
    /// Given a shape, mean, and standard deviation generate a new object using the [Normal](rand_distr::Normal) distribution
    fn normal<Sh>(shape: Sh, mean: A, std: A) -> Result<Self, NormalError>
    where
        A: Float,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        StandardNormal: Distribution<A>,
    {
        let distr = Normal::new(mean, std)?;
        Ok(Self::rand(shape, distr))
    }

    fn randc<Sh>(shape: Sh, re: A, im: A) -> Self
    where
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        ComplexDistribution<A, A>: Distribution<A>,
    {
        let distr = ComplexDistribution::new(re, im);
        Self::rand(shape, distr)
    }
    /// Generate a random array using the [StandardNormal](rand_distr::StandardNormal) distribution
    fn stdnorm<Sh>(shape: Sh) -> Self
    where
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        StandardNormal: Distribution<A>,
    {
        Self::rand(shape, StandardNormal)
    }
    /// Generate a random array using the [StandardNormal](rand_distr::StandardNormal) distribution with a given seed
    fn stdnorm_from_seed<Sh>(shape: Sh, seed: u64) -> Self
    where
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        StandardNormal: Distribution<A>,
    {
        Self::rand_with(shape, StandardNormal, &mut StdRng::seed_from_u64(seed))
    }
    /// Initialize the object using the [TruncatedNormal](crate::init::distr::TruncatedNormal) distribution
    fn truncnorm<Sh>(shape: Sh, mean: A, std: A) -> Result<Self, NormalError>
    where
        A: Float,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        StandardNormal: Distribution<A>,
    {
        let distr = TruncatedNormal::new(mean, std)?;
        Ok(Self::rand(shape, distr))
    }
    /// A [uniform](rand_distr::uniform::Uniform) generator with values between u(-dk, dk)
    fn uniform<Sh>(shape: Sh, dk: A) -> Self
    where
        A: Neg<Output = A> + SampleUniform,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        <A as SampleUniform>::Sampler: Clone,
    {
        Self::rand(shape, Uniform::new(dk.clone().neg(), dk))
    }

    fn uniform_from_seed<Sh>(shape: Sh, start: A, stop: A, key: u64) -> Self
    where
        A: SampleUniform,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        <A as SampleUniform>::Sampler: Clone,
    {
        Self::rand_with(
            shape,
            Uniform::new(start, stop),
            &mut StdRng::seed_from_u64(key),
        )
    }
    /// Generate a random array with values between u(-a, a) where a is the reciprocal of the value at the given axis
    fn uniform_along<Sh>(shape: Sh, axis: usize) -> Self
    where
        A: Copy + Float + SampleUniform,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        <A as SampleUniform>::Sampler: Clone,
    {
        let dim = shape.into_shape_with_order().raw_dim().clone();
        let dk = A::from(dim[axis]).unwrap().recip();
        Self::uniform(dim, dk)
    }
    /// A [uniform](rand_distr::uniform::Uniform) generator with values between u(-dk, dk)
    fn uniform_between<Sh>(shape: Sh, a: A, b: A) -> Self
    where
        A: SampleUniform,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        <A as SampleUniform>::Sampler: Clone,
    {
        Self::rand(shape, Uniform::new(a, b))
    }
}
/*
 ************ Implementations ************
*/

impl<S, A, D> Initialize<A, D> for ArrayBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
    ArrayBase<S, D>: RandomExt<S, A, D>,
{
    type Data = S;

    fn rand<Sh, Ds>(shape: Sh, distr: Ds) -> Self
    where
        Ds: Clone + Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
        S: DataOwned,
    {
        Self::random(shape, distr)
    }

    fn rand_with<Sh, Ds, R>(shape: Sh, distr: Ds, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
        Ds: Clone + Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
        S: DataOwned,
    {
        Self::random_using(shape, distr, rng)
    }
}

impl<U, A, S, D> InitializeExt<A, S, D> for U
where
    A: Clone,
    D: Dimension,
    S: RawData<Elem = A>,
    U: Initialize<A, D, Data = S> + Sized,
{
}
