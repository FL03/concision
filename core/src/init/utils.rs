/*
   Appellation: utils <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::*;
use ndarray_rand::RandomExt;
use num::complex::{Complex, ComplexDistribution};
use num::Num;
use rand::distributions::uniform::{SampleUniform, Uniform};
use rand::rngs::StdRng;
use rand::{rngs, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

#[cfg(feature = "rand")]
/// Generate a random array of complex numbers with real and imaginary parts in the range [0, 1)
pub fn randc<A, S, D>(shape: impl IntoDimension<Dim = D>) -> ArrayBase<S, D>
where
    A: Clone + Num,
    D: Dimension,
    S: RawData + DataOwned<Elem = Complex<A>>,
    ComplexDistribution<A, A>: Distribution<S::Elem>,
{
    let distr = ComplexDistribution::<A, A>::new(A::one(), A::one());
    ArrayBase::random(shape, distr)
}

/// Given a shape, generate a random array using the StandardNormal distribution
pub fn stdnorm<S, D, Sh>(shape: Sh) -> ArrayBase<S, D>
where
    D: Dimension,
    S: DataOwned,
    Sh: ShapeBuilder<Dim = D>,
    StandardNormal: Distribution<S::Elem>,
{
    ArrayBase::random(shape, StandardNormal)
}

pub fn stdnorm_from_seed<S, D, Sh>(shape: Sh, seed: u64) -> ArrayBase<S, D>
where
    D: Dimension,
    S: DataOwned,
    Sh: ShapeBuilder<Dim = D>,
    StandardNormal: Distribution<S::Elem>,
{
    ArrayBase::random_using(shape, StandardNormal, &mut StdRng::seed_from_u64(seed))
}
/// Creates a random array from a uniform distribution using a given key
pub fn uniform_from_seed<T, D>(
    key: u64,
    start: T,
    stop: T,
    shape: impl IntoDimension<Dim = D>,
) -> Array<T, D>
where
    D: Dimension,
    T: SampleUniform,
{
    Array::random_using(
        shape,
        Uniform::new(start, stop),
        &mut rngs::StdRng::seed_from_u64(key),
    )
}
