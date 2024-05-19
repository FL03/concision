/*
    Appellation: initialize <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use core::ops::Neg;
use nd::{ArrayBase, DataOwned, Dimension, RawData, ShapeBuilder};
use ndrand::RandomExt;
use num::complex::ComplexDistribution;
use num::traits::Float;
use rand::{rngs, Rng, SeedableRng};
use rand_distr::uniform::{SampleUniform, Uniform};
use rand_distr::{Bernoulli, BernoulliError, Distribution, Normal, StandardNormal};

use super::LecunNormal;

/// This trait provides the base methods required for initializing an [ndarray](ndarray::ArrayBase) with random values.
/// [Initialize] is similar to [RandomExt](ndarray_rand::RandomExt), however, it focuses on flexibility while implementing additional
/// features geared towards machine-learning models; such as lecun_normal initialization.
pub trait Initialize<A, D>
where
    D: Dimension,
{
    type Data: RawData<Elem = A>;
    /// Generate a random array using the given distribution
    fn rand<Sh, Ds>(shape: Sh, distr: Ds) -> Self
    where
        Ds: Clone + Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned;
    /// Generate a random array using the given distribution and random number generator
    fn rand_with<Sh, Ds, R>(shape: Sh, distr: Ds, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
        Ds: Clone + Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned;
    /// Initialize an array with random values using the given distribution and current shape
    fn init_rand<Ds>(self, distr: Ds) -> Self
    where
        Ds: Clone + Distribution<A>,
        Self: Sized,
        Self::Data: DataOwned;
    /// Initialize an array with random values from the current shape using the given distribution and random number generator
    fn init_rand_with<Ds, R>(self, distr: Ds, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
        Ds: Clone + Distribution<A>,
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
    fn normal<Sh>(shape: Sh, mean: A, std: A) -> Result<Self, rand_distr::NormalError>
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
        Self::rand_with(
            shape,
            StandardNormal,
            &mut rngs::StdRng::seed_from_u64(seed),
        )
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
    /// Generate a random array with values between u(-a, a) where a is the reciprocal of the value at the given axis
    fn uniform_along<Sh>(shape: Sh, axis: usize) -> Self
    where
        A: Copy + Float + SampleUniform,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        <A as SampleUniform>::Sampler: Clone,
    {
        let dim = shape.into_shape().raw_dim().clone();
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
impl<A, S, D> Initialize<A, D> for ArrayBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
    ArrayBase<S, D>: RandomExt<S, A, D>,
{
    type Data = S;

    fn rand<Sh, Ds>(shape: Sh, distr: Ds) -> ArrayBase<S, D>
    where
        S: DataOwned,
        Ds: Clone + Distribution<S::Elem>,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::random(shape, distr)
    }

    fn rand_with<Sh, Ds, R>(shape: Sh, distr: Ds, rng: &mut R) -> ArrayBase<S, D>
    where
        R: Rng + ?Sized,
        S: DataOwned,
        Ds: Clone + Distribution<S::Elem>,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::random_using(shape, distr, rng)
    }

    fn init_rand<Ds>(self, distr: Ds) -> ArrayBase<S, D>
    where
        S: DataOwned,
        Ds: Clone + Distribution<S::Elem>,
    {
        Self::rand(self.dim(), distr)
    }

    fn init_rand_with<Ds, R>(self, distr: Ds, rng: &mut R) -> ArrayBase<S, D>
    where
        R: Rng + ?Sized,
        S: DataOwned,
        Ds: Clone + Distribution<S::Elem>,
    {
        Self::rand_with(self.dim(), distr, rng)
    }
}

impl<U, A, S, D> InitializeExt<A, S, D> for U
where
    A: Clone,
    D: Dimension,
    S: RawData<Elem = A>,
    U: Initialize<A, D, Data = S>,
{
}
