/*
    Appellation: initialize <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::distr::*;

use core::ops::Neg;
use ndarray::{ArrayBase, DataOwned, Dimension, RawData, Shape, ShapeBuilder};
use num_traits::{Float, FromPrimitive};
use rand::{Rng, RngCore, SeedableRng};
use rand_distr::uniform::{SampleUniform, Uniform};
use rand_distr::{Bernoulli, BernoulliError, Distribution, Normal, NormalError, StandardNormal};

fn _extract_xy_from_shape<D>(dim: &D, x: usize, y: usize) -> (usize, usize)
where
    D: Dimension,
{
    let tmp = dim.as_array_view();
    let a = tmp.get(x).copied().unwrap_or(1_usize);
    let b = tmp.get(y).copied().unwrap_or(1_usize);
    (a, b)
}

/// The [`NdRandom`] trait focuses on providing an interface for initializing n-dimensional
/// tensors. Similar to the `RandomExt` trait from the `ndarray_rand` crate, it offers methods to
/// create tensors filled with random values drawn from various probability distributions.
/// The trait is similar to the `RandomExt` trait provided by the `ndarray_rand` crate,
/// however, it is designed to be more generic, extensible, and optimized for neural network
/// initialization routines.
pub trait NdRandom<S, D, A = <S as RawData>::Elem>: Sized
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Cont<_S, _D>
    where
        _D: Dimension,
        _S: RawData<Elem = A>;

    fn rand<Sh, Ds>(shape: Sh, distr: Ds) -> Self::Cont<S, D>
    where
        Ds: Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
        S: DataOwned;

    fn rand_with<Sh, Ds, R>(shape: Sh, distr: Ds, rng: &mut R) -> Self::Cont<S, D>
    where
        R: RngCore + ?Sized,
        Ds: Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
        S: DataOwned;

    fn bernoulli<Sh>(shape: Sh, p: f64) -> Result<Self::Cont<S, D>, BernoulliError>
    where
        Bernoulli: Distribution<A>,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        let dist = Bernoulli::new(p)?;
        Ok(Self::rand(shape, dist))
    }
    /// Initialize the object according to the Glorot Initialization scheme.
    fn glorot_normal<Sh: ShapeBuilder<Dim = D>>(shape: Sh) -> Self::Cont<S, D>
    where
        StandardNormal: Distribution<A>,
        S: DataOwned,
        A: Float + FromPrimitive,
    {
        let shape = shape.into_shape_with_order();
        let (inputs, outputs) = _extract_xy_from_shape(shape.raw_dim(), 0, 1);
        let distr = XavierNormal::new(inputs, outputs);
        Self::rand(shape, distr)
    }
    /// Initialize the object according to the Glorot Initialization scheme.
    fn glorot_uniform<Sh>(shape: Sh) -> crate::Result<Self::Cont<S, D>>
    where
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        A: Float + FromPrimitive + SampleUniform,
        <A as SampleUniform>::Sampler: Clone,
    {
        let shape = shape.into_shape_with_order();
        let (inputs, outputs) = _extract_xy_from_shape(shape.raw_dim(), 0, 1);
        let distr = XavierUniform::new(inputs, outputs)?;
        Ok(Self::rand(shape, distr))
    }
    /// Initialize the object according to the Lecun Initialization scheme.
    /// LecunNormal distributions are truncated [Normal](rand_distr::Normal)
    /// distributions centered at 0 with a standard deviation equal to the
    /// square root of the reciprocal of the number of inputs.
    fn lecun_normal<Sh>(shape: Sh) -> Self::Cont<S, D>
    where
        StandardNormal: Distribution<A>,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        A: Float,
    {
        let shape = shape.into_shape_with_order();
        let distr = LecunNormal::new(shape.size());
        Self::rand(shape, distr)
    }
    /// Given a shape, mean, and standard deviation generate a new object using the [Normal](rand_distr::Normal) distribution
    fn normal<Sh>(shape: Sh, mean: A, std: A) -> Result<Self::Cont<S, D>, NormalError>
    where
        StandardNormal: Distribution<A>,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        A: Float,
    {
        let distr = Normal::new(mean, std)?;
        Ok(Self::rand(shape, distr))
    }
    #[cfg(feature = "complex")]
    fn randc<Sh>(shape: Sh, re: A, im: A) -> Self::Cont<S, D>
    where
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        num_complex::ComplexDistribution<A, A>: Distribution<A>,
    {
        let distr = num_complex::ComplexDistribution::new(re, im);
        Self::rand(shape, &distr)
    }
    /// Generate a random array using the [StandardNormal](rand_distr::StandardNormal) distribution
    fn stdnorm<Sh>(shape: Sh) -> Self::Cont<S, D>
    where
        StandardNormal: Distribution<A>,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::rand(shape, StandardNormal)
    }
    #[cfg(feature = "std")]
    /// Generate a random array using the [`StandardNormal`] distribution with a given seed
    fn stdnorm_from_seed<Sh>(shape: Sh, seed: u64) -> Self::Cont<S, D>
    where
        StandardNormal: Distribution<A>,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::rand_with(
            shape,
            StandardNormal,
            &mut rand::rngs::StdRng::seed_from_u64(seed),
        )
    }
    /// Initialize the object using the [`TruncatedNormal`] distribution
    fn truncnorm<Sh>(shape: Sh, mean: A, std: A) -> crate::Result<Self::Cont<S, D>>
    where
        StandardNormal: Distribution<A>,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        A: Float,
    {
        let distr = TruncatedNormal::new(mean, std)?;
        Ok(Self::rand(shape, distr))
    }
    /// initialize the object using the [`Uniform`] distribution with values bounded by `+/- dk`
    fn uniform<Sh>(shape: Sh, dk: A) -> crate::Result<Self::Cont<S, D>>
    where
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        A: Clone + Neg<Output = A> + SampleUniform,
        <A as SampleUniform>::Sampler: Clone,
    {
        Self::uniform_between(shape, dk.clone().neg(), dk)
    }
    #[cfg(feature = "std")]
    /// randomly initialize the object using the [`Uniform`] distribution with values between
    /// the `start` and `stop` params using some random seed.
    fn uniform_from_seed<Sh>(
        shape: Sh,
        start: A,
        stop: A,
        key: u64,
    ) -> crate::Result<Self::Cont<S, D>>
    where
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        A: Clone + SampleUniform,
        <A as SampleUniform>::Sampler: Clone,
    {
        let distr = Uniform::new(start, stop)?;
        Ok(Self::rand_with(
            shape,
            distr,
            &mut rand::rngs::StdRng::seed_from_u64(key),
        ))
    }
    /// initialize the object using the [`Uniform`] distribution with values bounded by the
    /// size of the specified axis.
    /// The values are bounded by `+/- dk` where `dk = 1 / size(axis)`.
    fn uniform_along<Sh>(shape: Sh, axis: usize) -> crate::Result<Self::Cont<S, D>>
    where
        Sh: ShapeBuilder<Dim = D>,
        S: DataOwned,
        A: Float + FromPrimitive + SampleUniform,
        <A as SampleUniform>::Sampler: Clone,
    {
        // extract the shape
        let shape: Shape<D> = shape.into_shape_with_order();
        let dim: D = shape.raw_dim().clone();
        let dk = A::from_usize(dim[axis]).map(|i| i.recip()).unwrap();
        Self::uniform(dim, dk)
    }
    /// initialize the object using the [`Uniform`] distribution with values between then given
    /// bounds, `a` and `b`.
    fn uniform_between<Sh>(shape: Sh, a: A, b: A) -> crate::Result<Self::Cont<S, D>>
    where
        Sh: ShapeBuilder<Dim = D>,
        S: DataOwned,
        A: Clone + SampleUniform,
        <A as SampleUniform>::Sampler: Clone,
    {
        let distr = Uniform::new(a, b)?;
        Ok(Self::rand(shape, distr))
    }
}
/*
 ************ Implementations ************
*/

impl<A, S, D> NdRandom<S, D, A> for ArrayBase<S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Cont<_S, _D>
        = ArrayBase<_S, _D, A>
    where
        _D: Dimension,
        _S: RawData<Elem = A>;

    fn rand<Sh, Ds>(shape: Sh, distr: Ds) -> Self::Cont<S, D>
    where
        Ds: Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
        S: DataOwned,
    {
        Self::rand_with(
            shape,
            distr,
            &mut rand::rngs::SmallRng::from_rng(&mut rand::rng()),
        )
    }

    fn rand_with<Sh, Ds, R>(shape: Sh, distr: Ds, rng: &mut R) -> Self::Cont<S, D>
    where
        R: Rng + ?Sized,
        Ds: Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
        S: DataOwned,
    {
        Self::from_shape_simple_fn(shape, move || distr.sample(rng))
    }
}
