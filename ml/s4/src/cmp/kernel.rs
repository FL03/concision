/*
    Appellation: kernel <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::{AsComplex, Conjugate};
use crate::prelude::{cauchy, cauchy_complex};
use ndarray::prelude::{Array, Array1, Array2, Ix1, NdFloat};
use ndarray::ScalarOperand;
use ndarray_linalg::Scalar;
use num::complex::{Complex, ComplexFloat};
use num::traits::{Float, FloatConst, FromPrimitive, NumOps, Signed, Zero};
use rustfft::{FftNum, FftPlanner};
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

impl DPLRParams<Complex<f64>> {
    pub fn kernel_s(&self, step: f64, l: usize) -> Array1<f64> {
        let lt = l as f64; // T::from(l).unwrap();
        let omega_l = {
            let f = |i: usize| -> Complex<f64> {
                (Complex::i().neg() * (i as f64) * f64::PI() / lt).exp()
            };
            Array::from_iter((0..l).map(f))
        };

        let aterm = (self.c.conj(), self.q.conj());
        let bterm = (self.b.clone(), self.p.clone());

        let g = ((&omega_l.clone().neg() + 1.0) / (&omega_l + 1.0)) * (2.0 * step.recip());
        let c = omega_l.mapv(|i| 2.0 / (1.0 + i));

        let k00 = cauchy_complex(&(&aterm.0 * &bterm.0), &g, &self.lambda);
        let k01 = cauchy_complex(&(&aterm.0 * &bterm.1), &g, &self.lambda);
        let k10 = cauchy_complex(&(&aterm.1 * &bterm.0), &g, &self.lambda);
        let k11 = cauchy_complex(&(&aterm.1 * &bterm.1), &g, &self.lambda);

        let at_roots = &c * (&k00 - k01 * (&k11 + 1.0).mapv(ComplexFloat::recip) * &k10);

        let mut fft_planner = FftPlanner::new();
        let fft = fft_planner.plan_fft_inverse(l);
        // create a buffer to hold the complex numbers
        let mut buffer = at_roots.into_raw_vec();
        fft.process(buffer.as_mut_slice());
        Array::from_iter(buffer.into_iter().map(|i| i.re()))
    }
}

pub fn kernel_dplr<T>(
    dplr: &DPLRParams<<T as Scalar>::Complex>,
    step: <T as Scalar>::Real,
    l: usize,
) -> Array1<<T as Scalar>::Real>
where
    T: Conjugate + FftNum + Float + Scalar<Real = T, Complex = Complex<T>>,
    <T as Scalar>::Real:
        FloatConst + NumOps<<T as Scalar>::Complex, <T as Scalar>::Complex> + ScalarOperand,
    <T as Scalar>::Complex: ScalarOperand,
{
    let lt = <T as Scalar>::Real::from(l).unwrap();
    let two = <T as Scalar>::Real::from(2).unwrap();
    let omega_l: Array1<<T as Scalar>::Complex> = {
        let f = |i: usize| -> <T as Scalar>::Complex {
            ((<T as Scalar>::Real::from(i).unwrap() * <T as Scalar>::Real::PI()) / lt)
                .mul_complex(Complex::i().neg())
                .exp()
        };
        Array::from_iter((0..l).map(f))
    };

    let aterm = (dplr.c.conj(), dplr.q.conj());
    let bterm = (dplr.b.clone(), dplr.p.clone());

    let g = ((&omega_l.clone().neg() + <T as Scalar>::Real::one()) / (&omega_l + T::one()))
        * (T::one() * step.recip());
    let c = omega_l.mapv(|i| two.div_complex(<T as Scalar>::Real::one().add_complex(i)));

    let k00: Array1<<T as Scalar>::Complex> = cauchy(&(&aterm.0 * &bterm.0), &g, &dplr.lambda);
    let k01: Array1<<T as Scalar>::Complex> = cauchy(&(&aterm.0 * &bterm.1), &g, &dplr.lambda);
    let k10: Array1<<T as Scalar>::Complex> = cauchy(&(&aterm.1 * &bterm.0), &g, &dplr.lambda);
    let k11: Array1<<T as Scalar>::Complex> = cauchy(&(&aterm.1 * &bterm.1), &g, &dplr.lambda);

    let at_roots = &c * (&k00 - k01 * (&k11 + T::one()).mapv(ComplexFloat::recip) * &k10);

    let mut fft_planner = FftPlanner::new();
    let fft = fft_planner.plan_fft_inverse(l);
    // create a buffer to hold the complex numbers
    let mut buffer = at_roots.into_raw_vec();
    fft.process(buffer.as_mut_slice());
    Array::from_iter(buffer.into_iter().map(|i| i.re()))
}

pub struct Kernel<T = f64> {
    kernal: Array2<T>,
}

impl<T> Kernel<T>
where
    T: Float,
{
    pub fn new(kernal: Array2<T>) -> Self {
        Self { kernal }
    }

    pub fn square(features: usize) -> Self
    where
        T: Default,
    {
        let kernal = Array2::<T>::default((features, features));
        Self::new(kernal)
    }

    pub fn kernal(&self) -> &Array2<T> {
        &self.kernal
    }
}
