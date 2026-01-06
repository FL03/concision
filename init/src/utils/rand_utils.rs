/*
   Appellation: utils <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::InitTensor;
use ndarray::{Array, ArrayBase, DataOwned, Dimension, IntoDimension, ShapeBuilder};
use rand::{SeedableRng, rngs};
use rand_distr::{
    Distribution, StandardNormal,
    uniform::{SampleUniform, Uniform},
};

#[cfg(feature = "complex")]
/// Generate a random array of complex numbers with real and imaginary parts in the range [0, 1)
pub fn randc<A, S, D>(shape: impl IntoDimension<Dim = D>) -> ArrayBase<S, D>
where
    A: Clone + num_traits::Num,
    D: Dimension,
    S: DataOwned<Elem = num_complex::Complex<A>>,
    num_complex::ComplexDistribution<A, A>: Distribution<S::Elem>,
{
    let distr = num_complex::ComplexDistribution::<A, A>::new(A::one(), A::one());
    ArrayBase::rand(shape, distr)
}

/// Given a shape, generate a random array using the StandardNormal distribution
pub fn stdnorm<S, D, Sh>(shape: Sh) -> ArrayBase<S, D>
where
    D: Dimension,
    S: DataOwned,
    Sh: ShapeBuilder<Dim = D>,
    StandardNormal: Distribution<S::Elem>,
{
    ArrayBase::rand(shape, StandardNormal)
}

pub fn stdnorm_from_seed<S, D, Sh>(shape: Sh, seed: u64) -> ArrayBase<S, D>
where
    D: Dimension,
    S: DataOwned,
    Sh: ShapeBuilder<Dim = D>,
    StandardNormal: Distribution<S::Elem>,
{
    ArrayBase::rand_with(
        shape,
        StandardNormal,
        &mut rngs::StdRng::seed_from_u64(seed),
    )
}
/// Creates a random array from a uniform distribution using a given key
pub fn uniform_from_seed<T, D>(
    key: u64,
    start: T,
    stop: T,
    shape: impl IntoDimension<Dim = D>,
) -> crate::InitResult<Array<T, D>>
where
    D: Dimension,
    T: SampleUniform,
{
    Uniform::new(start, stop)
        .map(|distr| ArrayBase::rand_with(shape, &distr, &mut rngs::StdRng::seed_from_u64(key)))
        .map_err(Into::into)
}
