/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::fft::*;

use ndarray::prelude::*;
use ndarray::{IntoDimension, ScalarOperand};
use ndarray_rand::rand_distr::{uniform::SampleUniform, Uniform};
use ndarray_rand::RandomExt;
use num::traits::real::Real;
use num::traits::Num;
use rand::distributions::Distribution;
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
pub fn logstep<T>(a: T, b: T,) -> T
where
    T: Real + ScalarOperand + SampleUniform,
{
    Uniform::new(a, b).sample(&mut rand::thread_rng()) * (b.ln() - a.ln()) + a.ln()
}

pub(crate) fn logstep_initializer<T, D>(between: Option<(T, T)>, shape: impl IntoDimension<Dim = D>) -> Array<T, D>
where
    D: Dimension,
    T: Real + ScalarOperand + SampleUniform,
{
    let (a, b) = between.unwrap_or((T::from(1e-3).unwrap(), T::from(1e-1).unwrap()));
    Array::random(shape, Uniform::new(a, b)) * (b.ln() - a.ln()) + a.ln()
}

pub(crate) mod fft {
    use ndarray::prelude::{Array2, Array,};
    use ndrustfft::{R2cFftHandler, ndifft_r2c, ndfft_r2c};
    use num::complex::Complex;
    use num::traits::{FloatConst, NumCast};
    use realfft::RealFftPlanner;
    use rustfft::FftNum;

    pub fn rfft_2d<T>(input: &Array2<T>,) -> Array2<Complex<T>> where T: FftNum + FloatConst {
        let axis = 1;
        let (m, _n) = input.dim();
        let d_out = input.shape()[axis] / 2 + 1;
        let mut handler = R2cFftHandler::<T>::new(input.shape()[axis]);
        let mut out = Array::zeros((m, d_out));
        ndfft_r2c(input, &mut out, &mut handler, axis);
        println!("real");
        out
    }

    pub fn irfft_2d<T>(input: &Array2<Complex<T>>, len: usize) -> Array2<T> where T: FftNum + FloatConst {
        let axis = 1;
        let (m, _n) = input.dim();
        let d_out = (input.shape()[axis] - 1) * 2;
        let mut handler = R2cFftHandler::<T>::new(d_out);
        let mut out = Array::zeros((m, len));
        ndifft_r2c(input, &mut out, &mut handler, axis);
        out
    }

    // pub fn ndrfft<T, D>(input: &Array<T, D>, axis: usize) -> Array<Complex<T>, D> where D: Dimension, T: FftNum + FloatConst {
    //     let dim: D = input.shape().into_dimension();
    //     let mut shape_out = shape..clone();
    //     shape[axis] = input.shape()[axis] / 2 + 1;
    //     let dim = shape_out.iter().collect_tuple().unwrap();
    //     let mut handler = R2cFftHandler::<T>::new(input.shape()[axis]);
    //     let mut out = Array::zeros(dim);
    //     ndfft_r2c(input, &mut out, &mut handler, axis);
    //     out
    // }

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
