/*
    Appellation: initialize <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::distr::*;

use core::ops::Neg;
use ndarray::{ArrayBase, DataOwned, Dimension, RawData, Shape, ShapeBuilder};
use num_traits::{Float, FromPrimitive};
use rand::rngs::{SmallRng, StdRng};
use rand::{Rng, SeedableRng};
use rand_distr::uniform::{SampleUniform, Uniform};
use rand_distr::{Bernoulli, BernoulliError, Distribution, Normal, NormalError, StandardNormal};

/// This trait provides the base methods required for initializing tensors with random values.
/// The trait is similar to the `RandomExt` trait provided by the `ndarray_rand` crate,
/// however, it is designed to be more generic, extensible, and optimized for neural network
/// initialization routines. [Initialize] is implemented for [`ArrayBase`] as well as
/// [`ParamsBase`](crate::ParamsBase) allowing you to randomly initialize new tensors and
/// parameters.
pub trait Initialize<S, D>: Sized
where
    D: Dimension,
    S: RawData,
{
    fn rand<Sh, Ds>(shape: Sh, distr: Ds) -> Self
    where
        Ds: Distribution<S::Elem>,
        Sh: ShapeBuilder<Dim = D>,
        S: DataOwned;

    fn rand_with<Sh, Ds, R>(shape: Sh, distr: Ds, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
        Ds: Distribution<S::Elem>,
        Sh: ShapeBuilder<Dim = D>,
        S: DataOwned;

    fn bernoulli<Sh>(shape: Sh, p: f64) -> Result<Self, BernoulliError>
    where
        Bernoulli: Distribution<S::Elem>,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        let dist = Bernoulli::new(p)?;
        Ok(Self::rand(shape, dist))
    }
    /// Initialize the object according to the Glorot Initialization scheme.
    fn glorot_normal<Sh>(shape: Sh, inputs: usize, outputs: usize) -> Self
    where
        S::Elem: Float + FromPrimitive,
        StandardNormal: Distribution<S::Elem>,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        let distr = XavierNormal::new(inputs, outputs);
        Self::rand(shape, distr)
    }
    /// Initialize the object according to the Glorot Initialization scheme.
    fn glorot_uniform<Sh>(shape: Sh, inputs: usize, outputs: usize) -> super::UniformResult<Self>
    where
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        S::Elem: Float + FromPrimitive + SampleUniform,
        <S::Elem as SampleUniform>::Sampler: Clone,
    {
        let distr = XavierUniform::new(inputs, outputs)?;
        Ok(Self::rand(shape, distr))
    }
    /// Initialize the object according to the Lecun Initialization scheme.
    /// LecunNormal distributions are truncated [Normal](rand_distr::Normal)
    /// distributions centered at 0 with a standard deviation equal to the
    /// square root of the reciprocal of the number of inputs.
    fn lecun_normal<Sh>(shape: Sh) -> Self
    where
        StandardNormal: Distribution<S::Elem>,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        S::Elem: Float,
    {
        let shape = shape.into_shape_with_order();
        let distr = LecunNormal::new(shape.size());
        Self::rand(shape, distr)
    }
    /// Given a shape, mean, and standard deviation generate a new object using the [Normal](rand_distr::Normal) distribution
    fn normal<Sh>(shape: Sh, mean: S::Elem, std: S::Elem) -> Result<Self, NormalError>
    where
        StandardNormal: Distribution<S::Elem>,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        S::Elem: Float,
    {
        let distr = Normal::new(mean, std)?;
        Ok(Self::rand(shape, distr))
    }
    #[cfg(feature = "complex")]
    fn randc<Sh>(shape: Sh, re: S::Elem, im: S::Elem) -> Self
    where
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        num_complex::ComplexDistribution<S::Elem, S::Elem>: Distribution<S::Elem>,
    {
        let distr = num_complex::ComplexDistribution::new(re, im);
        Self::rand(shape, &distr)
    }
    /// Generate a random array using the [StandardNormal](rand_distr::StandardNormal) distribution
    fn stdnorm<Sh>(shape: Sh) -> Self
    where
        StandardNormal: Distribution<S::Elem>,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::rand(shape, StandardNormal)
    }
    /// Generate a random array using the [StandardNormal](rand_distr::StandardNormal) distribution with a given seed
    fn stdnorm_from_seed<Sh>(shape: Sh, seed: u64) -> Self
    where
        StandardNormal: Distribution<S::Elem>,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::rand_with(shape, StandardNormal, &mut StdRng::seed_from_u64(seed))
    }
    /// Initialize the object using the [TruncatedNormal](crate::init::distr::TruncatedNormal) distribution
    fn truncnorm<Sh>(shape: Sh, mean: S::Elem, std: S::Elem) -> Result<Self, NormalError>
    where
        StandardNormal: Distribution<S::Elem>,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        S::Elem: Float,
    {
        let distr = TruncatedNormal::new(mean, std)?;
        Ok(Self::rand(shape, distr))
    }
    /// initialize the object using the [`Uniform`] distribution with values bounded by `+/- dk`
    fn uniform<Sh>(shape: Sh, dk: S::Elem) -> super::UniformResult<Self>
    where
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        S::Elem: Clone + Neg<Output = S::Elem> + SampleUniform,
        <S::Elem as SampleUniform>::Sampler: Clone,
    {
        Self::uniform_between(shape, dk.clone().neg(), dk)
    }
    /// randomly initialize the object using the [`Uniform`] distribution with values between
    /// the `start` and `stop` params using some random seed.
    fn uniform_from_seed<Sh>(
        shape: Sh,
        start: S::Elem,
        stop: S::Elem,
        key: u64,
    ) -> super::UniformResult<Self>
    where
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        S::Elem: Clone + SampleUniform,
        <S::Elem as SampleUniform>::Sampler: Clone,
    {
        let distr = Uniform::new(start, stop)?;
        Ok(Self::rand_with(
            shape,
            distr,
            &mut StdRng::seed_from_u64(key),
        ))
    }
    /// initialize the object using the [`Uniform`] distribution with values bounded by the
    /// size of the specified axis.
    /// The values are bounded by `+/- dk` where `dk = 1 / size(axis)`.
    fn uniform_along<Sh>(shape: Sh, axis: usize) -> super::UniformResult<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
        S: DataOwned,
        S::Elem: Float + FromPrimitive + SampleUniform,
        <S::Elem as SampleUniform>::Sampler: Clone,
    {
        // extract the shape
        let shape: Shape<D> = shape.into_shape_with_order();
        let dim: D = shape.raw_dim().clone();
        let dk = S::Elem::from_usize(dim[axis]).map(|i| i.recip()).unwrap();
        Self::uniform(dim, dk)
    }
    /// initialize the object using the [`Uniform`] distribution with values between then given
    /// bounds, `a` and `b`.
    fn uniform_between<Sh>(shape: Sh, a: S::Elem, b: S::Elem) -> super::UniformResult<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
        S: DataOwned,
        S::Elem: Clone + SampleUniform,
        <S::Elem as SampleUniform>::Sampler: Clone,
    {
        let distr = Uniform::new(a, b)?;
        Ok(Self::rand(shape, distr))
    }
}
/*
 ************ Implementations ************
*/

impl<A, S, D> Initialize<S, D> for ArrayBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    fn rand<Sh, Ds>(shape: Sh, distr: Ds) -> Self
    where
        Ds: Distribution<S::Elem>,
        Sh: ShapeBuilder<Dim = D>,
        S: DataOwned,
    {
        Self::rand_with(shape, distr, &mut SmallRng::from_rng(&mut rand::rng()))
    }

    fn rand_with<Sh, Ds, R>(shape: Sh, distr: Ds, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
        Ds: Distribution<S::Elem>,
        Sh: ShapeBuilder<Dim = D>,
        S: DataOwned,
    {
        Self::from_shape_simple_fn(shape, move || distr.sample(rng))
    }
}
