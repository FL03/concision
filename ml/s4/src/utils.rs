/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::fft::*;

use ndarray::prelude::*;
use ndarray::{IntoDimension, ScalarOperand};
use ndarray_rand::rand_distr::{uniform, Uniform};
use ndarray_rand::RandomExt;
use num::traits::real::Real;
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
    T: Real + ScalarOperand + uniform::SampleUniform,
{
    Array::random(shape, Uniform::new(a, b)) * (b.ln() - a.ln()) + a.ln()
}

pub(crate) mod fft {
    use num::{Complex, NumCast};
    use realfft::RealFftPlanner;
    use rustfft::FftNum;

    /// A utilitarian function for computing the Fast Fourier Transform of a real-valued signal
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
    /// A utilitarian function for computing the Inverse Fast Fourier Transform of a real-valued signal
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
