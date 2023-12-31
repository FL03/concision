/*
    Appellation: specs <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::AsComplex;
use ndarray::prelude::{Array, Dimension};
use num::complex::ComplexFloat;
use rustfft::{Fft, FftNum, FftPlanner};

pub trait Scan<S, T> {
    type Output;

    fn scan(&self, args: &T, initial_state: &S) -> Self::Output;
}

pub trait StateSpace<T> {
    type Config;

    fn config(&self) -> &Self::Config;
}

pub trait NdFft {
    type Output;

    fn fft(&self, args: &Self) -> Self::Output;

    fn ifft(&self, args: &Self) -> Self::Output;
}

impl<T, D> NdFft for Array<T, D>
where
    D: Dimension,
    T: AsComplex + ComplexFloat + FftNum,
{
    type Output = Self;

    fn fft(&self, args: &Self) -> Self::Output {
        let dim = args.dim();
        let mut out = Self::ones(args.dim());
        let mut buffer = vec![T::zero().as_complex(); args.len()];
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(args.len());
        fft.process(buffer.as_mut_slice());
        let buffer = buffer
            .into_iter()
            .map(|i| T::from(i).unwrap())
            .collect::<Vec<_>>();
        Self::from_shape_vec(args.dim(), buffer).expect("")
    }

    fn ifft(&self, args: &Self) -> Self::Output {
        let dim = args.dim();
        let mut out = Self::ones(args.dim());
        let mut buffer = vec![T::zero().as_complex(); args.len()];
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_inverse(args.len());
        fft.process(buffer.as_mut_slice());
        let buffer = buffer
            .into_iter()
            .map(|i| T::from(i).unwrap())
            .collect::<Vec<_>>();
        Self::from_shape_vec(args.dim(), buffer).expect("")
    }
}
