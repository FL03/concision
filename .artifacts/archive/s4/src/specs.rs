/*
    Appellation: specs <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use concision::prelude::AsComplex;
use ndarray::prelude::{Array, Dimension};
use ndarray_linalg::Scalar;
use num::complex::{Complex, ComplexFloat};
use num::traits::{Num, Signed};
use rustfft::{FftNum, FftPlanner};

pub trait Conjugate {
    fn conj(&self) -> Self;
}

impl Conjugate for f32 {
    fn conj(&self) -> Self {
        *self
    }
}

impl Conjugate for f64 {
    fn conj(&self) -> Self {
        *self
    }
}

impl<T> Conjugate for Complex<T>
where
    T: Clone + Conjugate + Num + Signed,
{
    fn conj(&self) -> Self {
        Complex::conj(&self)
    }
}
impl<A, D> Conjugate for Array<A, D>
where
    A: Clone + Conjugate,
    D: Dimension,
{
    fn conj(&self) -> Self {
        self.mapv(|i| i.conj())
    }
}

pub trait Scan<S, T> {
    type Output;

    fn scan(&self, args: &T, initial_state: &S) -> Self::Output;
}

pub trait NdFft {
    type Output;

    fn fft(&self, args: &Self) -> Self::Output;

    fn ifft(&self, args: &Self) -> Self::Output;
}

impl<T, D> NdFft for Array<T, D>
where
    D: Dimension,
    T: AsComplex<Real = <T as ComplexFloat>::Real> + ComplexFloat,
    <T as ComplexFloat>::Real: FftNum,
{
    type Output = Self;

    fn fft(&self, args: &Self) -> Self::Output {
        let mut buffer = vec![T::zero().as_re(); args.len()];
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
        let mut buffer = vec![T::zero().as_re(); args.len()];
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
