/*
   Appellation: utils <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "rand")]
use ndarray::*;
use ndrand::rand::rngs::StdRng;
use ndrand::rand::SeedableRng;
use ndrand::rand_distr::{Distribution, StandardNormal};
use ndrand::RandomExt;
use num::complex::{Complex, ComplexDistribution};
use num::traits::real::Real;
use num::Num;
use rand::distributions::uniform::{SampleUniform, Uniform};

pub fn lecun_normal<T, D>(shape: impl IntoDimension<Dim = D>) -> Array<T, D>
where
    D: Dimension,
    T: Real + ScalarOperand,
    StandardNormal: Distribution<T>,
{
    let dim = shape.into_dimension();
    let n = dim.size();
    let scale = T::from(n).unwrap().recip().sqrt();
    Array::random(dim, StandardNormal) * scale
}

pub fn lecun_normal_seeded<T, D>(shape: impl IntoDimension<Dim = D>, seed: u64) -> Array<T, D>
where
    D: Dimension,
    T: Real + ScalarOperand,
    StandardNormal: Distribution<T>,
{
    let dim = shape.into_dimension();
    let n = dim.size();
    let scale = T::from(n).unwrap().recip().sqrt();
    Array::random_using(dim, StandardNormal, &mut StdRng::seed_from_u64(seed)) * scale
}

/// Generate a random array of complex numbers with real and imaginary parts in the range [0, 1)
pub fn randc<T, D>(shape: impl IntoDimension<Dim = D>) -> Array<Complex<T>, D>
where
    D: Dimension,
    T: Clone + Num,
    ComplexDistribution<T, T>: Distribution<Complex<T>>,
{
    let distr = ComplexDistribution::<T, T>::new(T::one(), T::one());
    Array::random(shape, distr)
}
///
pub fn randcomplex<T, D>(shape: impl IntoDimension<Dim = D>) -> Array<Complex<T>, D>
where
    D: Dimension,
    T: Copy + Num,
    StandardNormal: Distribution<T>,
{
    let dim = shape.into_dimension();
    let re = Array::random(dim.clone(), StandardNormal);
    let im = Array::random(dim.clone(), StandardNormal);
    let mut res = Array::zeros(dim);
    ndarray::azip!((re in &re, im in &im, res in &mut res) {
        *res = Complex::new(*re, *im);
    });
    res
}
/// Creates a random array from a uniform distribution using a given key
pub fn seeded_uniform<T, D>(
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
        &mut StdRng::seed_from_u64(key),
    )
}
///
pub fn seeded_stdnorm<T, D>(key: u64, shape: impl IntoDimension<Dim = D>) -> Array<T, D>
where
    D: Dimension,
    StandardNormal: Distribution<T>,
{
    Array::random_using(shape, StandardNormal, &mut StdRng::seed_from_u64(key))
}
///
pub fn randc_normal<T, D>(key: u64, shape: impl IntoDimension<Dim = D>) -> Array<Complex<T>, D>
where
    D: Dimension,
    T: Copy + Num,
    StandardNormal: Distribution<T>,
{
    let dim = shape.into_dimension();
    let re = seeded_stdnorm(key, dim.clone());
    let im = seeded_stdnorm(key, dim.clone());
    let mut res = Array::zeros(dim);
    ndarray::azip!((re in &re, im in &im, res in &mut res) {
        *res = Complex::new(*re, *im);
    });
    res
}
/// Given a shape, generate a random array using the StandardNormal distribution
pub fn stdnorm<T, D>(shape: impl IntoDimension<Dim = D>) -> Array<T, D>
where
    D: Dimension,
    StandardNormal: Distribution<T>,
{
    Array::random(shape, StandardNormal)
}
