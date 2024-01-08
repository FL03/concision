/*
    Appellation: kernel <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::ops::fft::*;
use crate::core::prelude::Conjugate;
use crate::prelude::cauchy;
use ndarray::prelude::{Array, Array1};
use ndarray::ScalarOperand;
use ndarray_linalg::Scalar;
use num::complex::{Complex, ComplexFloat};
use num::traits::{Float, FloatConst, NumOps};
use rustfft::FftPlanner;
use std::ops::Neg;

pub struct DPLRParams<T = f64> {
    pub lambda: Array1<T>,
    pub p: Array1<T>,
    pub q: Array1<T>,
    pub b: Array1<T>,
    pub c: Array1<T>,
}

impl<T> DPLRParams<T> {
    pub fn new(lambda: Array1<T>, p: Array1<T>, q: Array1<T>, b: Array1<T>, c: Array1<T>) -> Self {
        Self { lambda, p, q, b, c }
    }
}

// impl<T> DPLRParams<T>
// where
//     T: ComplexFloat,
//     <T as ComplexFloat>::Real: NumOps + NumOps<Complex<<T as ComplexFloat>::Real>, Complex<<T as ComplexFloat>::Real>>,
//     Complex<<T as ComplexFloat>::Real>: NumOps + NumOps<<T as ComplexFloat>::Real, Complex<<T as ComplexFloat>::Real>>
// {
//     pub fn kernel(&self, step: T, l: usize) -> Array1<<T as ComplexFloat>::Real> {
//         let lt = T::from(l).unwrap();
//         let omega_l = {
//             let f = | i: usize | -> Complex<<T as ComplexFloat>::Real> {
//                 Complex::<<T as ComplexFloat>::Real>::i().neg() * <T as ComplexFloat>::Real::from(i).unwrap() * <T as ComplexFloat>::Real::PI() / lt
//             };
//             Array::from_iter((0..l).map(f))
//         };
//     }
// }

pub fn omega_l<T>(l: usize) -> Array1<<T as Scalar>::Complex>
where
    T: Scalar<Real = T, Complex = Complex<T>>,
    <T as Scalar>::Real: FloatConst + NumOps<<T as Scalar>::Complex, <T as Scalar>::Complex>,
    <T as Scalar>::Complex:
        ComplexFloat<Real = <T as Scalar>::Real> + NumOps<<T as Scalar>::Real> + ScalarOperand,
{
    let f = |i: usize| -> <T as Scalar>::Complex {
        let im = T::PI().mul_complex(Complex::i() * T::from(2).unwrap());
        T::from(i)
            .unwrap()
            .div_real(T::from(l).unwrap())
            .mul_complex(im)
            .exp()
    };
    Array::from_iter((0..l).map(f))
}

pub fn kernel_dplr<T>(
    dplr: &DPLRParams<<T as Scalar>::Complex>,
    step: <T as Scalar>::Real,
    l: usize,
) -> Array1<<T as Scalar>::Real>
where
    T: Conjugate + Float + Scalar<Real = T, Complex = Complex<T>>,
    <T as Scalar>::Real:
        FloatConst + NumOps<<T as Scalar>::Complex, <T as Scalar>::Complex> + ScalarOperand,
    <T as Scalar>::Complex: Conjugate + ScalarOperand,
{
    let one = T::one();
    let two = T::from(2).unwrap();
    // get the lambda matrix
    let lambda = dplr.lambda.clone();
    // generate omega
    let omega_l: Array1<<T as Scalar>::Complex> = omega_l::<T>(l);
    // collect the relevant terms for A
    let aterm = (dplr.c.conj(), dplr.q.conj());
    // collect the relevant terms for B
    let bterm = (dplr.b.clone(), dplr.p.clone());

    let g = omega_l.mapv(|i| (one - i) * (one + i).recip()) * (two * step.recip());
    let c = omega_l.mapv(|i| two * (one + i).recip());
    // compute the cauchy matrix
    let k00: Array1<<T as Scalar>::Complex> = cauchy(&(&aterm.0 * &bterm.0), &g, &lambda);
    let k01: Array1<<T as Scalar>::Complex> = cauchy(&(&aterm.0 * &bterm.1), &g, &lambda);
    let k10: Array1<<T as Scalar>::Complex> = cauchy(&(&aterm.1 * &bterm.0), &g, &lambda);
    let k11: Array1<<T as Scalar>::Complex> = cauchy(&(&aterm.1 * &bterm.1), &g, &lambda);
    // compute the roots of unity
    let at_roots = &c * (&k00 - k01 * &k11.mapv(|i| (i + one).recip()) * &k10);

    let buffer = at_roots.into_raw_vec();
    let permute = FftPlan::new(l);
    let res = ifftr(buffer.as_slice(), &permute);
    Array::from_vec(res)
}

pub struct Kernel<T = f64> {
    kernal: Array1<T>,
}

impl<T> Kernel<T> {
    pub fn new(kernal: Array1<T>) -> Self {
        Self { kernal }
    }

    pub fn kernal(&self) -> &Array1<T> {
        &self.kernal
    }
}

impl<T> Kernel<T>
where
    T: Scalar<Real = T, Complex = Complex<T>>,
    <T as Scalar>::Real: Conjugate
        + Float
        + FloatConst
        + NumOps<<T as Scalar>::Complex, <T as Scalar>::Complex>
        + ScalarOperand,
    <T as Scalar>::Complex: Conjugate + ScalarOperand,
{
    pub fn dplr(
        dplr: &DPLRParams<<T as Scalar>::Complex>,
        step: <T as Scalar>::Real,
        l: usize,
    ) -> Self {
        let kernal = kernel_dplr::<T>(dplr, step, l);
        Self::new(kernal)
    }
}
