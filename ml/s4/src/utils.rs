/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::fft::*;

use ndarray::prelude::*;
use ndarray::{IntoDimension, ScalarOperand};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::{Distribution, Uniform};
use ndarray_rand::RandomExt;
use num::complex::{Complex, ComplexDistribution,};
use num::traits::Num;
use std::ops::Neg;

///
pub fn cauchy<T, A, B>(a: &Array<T, A>, b: &Array<T, B>, c: &Array<T, A>) -> Array<T, B>
where
    A: Dimension,
    B: Dimension,
    T: Num + Neg<Output = T> + ScalarOperand,
{
    let cdot = |b: T| (a / (c * T::one().neg() + b)).sum();
    b.mapv(cdot)
}
///
pub fn logstep<T, D>(a: T, b: T, shape: impl IntoDimension<Dim = D>) -> Array<T, D>
where
    D: Dimension,
    T: NdFloat + SampleUniform,
{
    Array::random(shape, Uniform::new(a, b)) * (b.ln() - a.ln()) + a.ln()
}
/// Generate a random array of complex numbers with real and imaginary parts in the range [0, 1)
pub fn randc<T, D>(shape: impl IntoDimension<Dim = D>) -> Array<Complex<T>, D>
where
    D: Dimension,
    T: Distribution<T> + Num,
    ComplexDistribution<T, T>: Distribution<Complex<T>>,
{
    let distr = ComplexDistribution::<T, T>::new(T::one(), T::one());
    Array::random(shape, distr)
}

pub(crate) mod fft {
    use num::{Complex, NumCast};
    use realfft::RealFftPlanner;
    use rustfft::FftNum;

    pub fn rfft<T>(args: impl IntoIterator<Item = T>) -> Vec<Complex<T>>
    where
        T: FftNum,
    {
        let mut buffer = Vec::from_iter(args);
        // make a planner
        let mut real_planner = RealFftPlanner::<T>::new();
        // create a FFT
        let r2c = real_planner.plan_fft_forward(buffer.len());
        // make a vector for storing the spectrum
        let mut spectrum = r2c.make_output_vec();
        // forward transform the signal
        r2c.process(&mut buffer, &mut spectrum).unwrap();
        spectrum
    }

    pub fn irfft<T>(args: impl IntoIterator<Item = Complex<T>>, len: usize) -> Vec<T>
    where
        T: FftNum + NumCast,
    {
        let mut buffer = Vec::from_iter(args);
        // make a planner
        let mut real_planner = RealFftPlanner::<T>::new();
        // create a FFT
        let r2c = real_planner.plan_fft_inverse(len);
        // make a vector for storing the spectrum
        let mut spectrum = r2c.make_output_vec();
        // forward transform the signal
        r2c.process(&mut buffer, &mut spectrum).unwrap();
        let scale = T::one() / T::from(len).unwrap();
        spectrum.iter().cloned().map(|i| i * scale).collect()
    }
}
