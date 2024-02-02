/*
    Appellation: kernel <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::ops::fft::{ifft, FftPlan};
use crate::core::prelude::Conjugate;
use crate::params::DPLRParams;
use crate::prelude::cauchy;
use ndarray::prelude::{Array, Array1};
use ndarray::ScalarOperand;
use ndarray_linalg::Scalar;
use num::complex::{Complex, ComplexFloat};
use num::traits::{Float, FloatConst, NumOps};
use rustfft::FftNum;

pub fn omega_l<T>(l: usize) -> Array1<<T as Scalar>::Complex>
where
    T: Scalar<Real = T, Complex = Complex<T>>,
    <T as Scalar>::Real: FloatConst + NumOps<<T as Scalar>::Complex, <T as Scalar>::Complex>,
    <T as Scalar>::Complex:
        ComplexFloat<Real = <T as Scalar>::Real> + NumOps<<T as Scalar>::Real> + ScalarOperand,
{
    let f = |i: usize| -> <T as Scalar>::Complex {
        let im = T::PI().mul_complex(Complex::i() * T::from(2).unwrap()); // .neg()
        T::from(i)
            .unwrap()
            .div_real(T::from(l).unwrap())
            .mul_complex(im)
            .exp()
    };
    Array::from_iter((0..l).map(f))
}

pub struct Omega<T> {
    omega: Array1<Complex<T>>,
}

impl<T> Omega<T>
where
    T: FloatConst + NumOps<Complex<T>, Complex<T>> + Scalar<Real = T, Complex = Complex<T>>,
    <T as Scalar>::Complex: ComplexFloat<Real = T> + NumOps<T>,
{
    pub fn new(l: usize) -> Self {
        let lt = T::from(l).unwrap();
        let f = |i: usize| -> Complex<T> {
            let im = T::PI() * (Complex::i() * T::from(2).unwrap()); // .neg()
            let out = im * (T::from(i).unwrap() / lt);
            out.exp()
            // T::from(i)
            //     .unwrap()
            //     .div_real(T::from(l).unwrap())
            //     .mul_complex(im)
            //     .exp()
        };
        let omega = Array::from_iter((0..l).map(f));
        Self { omega }
    }
}

impl<T> Omega<T> {
    pub fn omega(&self) -> &Array1<Complex<T>> {
        &self.omega
    }
}

pub fn kernel_dplr<T>(
    dplr: &DPLRParams<<T as Scalar>::Complex>,
    step: <T as Scalar>::Real,
    l: usize,
) -> Array1<<T as Scalar>::Real>
where
    T: Conjugate + FftNum + Float + Scalar<Real = T, Complex = Complex<T>> + ScalarOperand,
    <T as Scalar>::Real: FloatConst + NumOps<<T as Scalar>::Complex, <T as Scalar>::Complex>,
    <T as Scalar>::Complex: ScalarOperand,
{
    // initialize some constants
    let two = T::from(2).unwrap();
    // get the lambda matrix
    let lambda = dplr.lambda.clone();
    // collect the relevant terms for A
    let aterm = (dplr.c.conj(), dplr.q.conj());
    // collect the relevant terms for B
    let bterm = (dplr.b.clone(), dplr.p.clone());
    // generate omega
    let omega_l = omega_l::<T>(l);

    let g = omega_l.mapv(|i| (T::one() - i) * (T::one() + i).recip()) * (two * step.recip());
    let c = omega_l.mapv(|i| two * (T::one() + i).recip());
    // compute the cauchy matrix
    let k00 = cauchy(&(&aterm.0 * &bterm.0), &g, &lambda);
    let k01 = cauchy(&(&aterm.0 * &bterm.1), &g, &lambda);
    let k10 = cauchy(&(&aterm.1 * &bterm.0), &g, &lambda);
    let k11 = cauchy(&(&aterm.1 * &bterm.1), &g, &lambda);
    // compute the roots of unity
    let at_roots = &c * (&k00 - k01 * &k11.mapv(|i| (i + T::one()).recip()) * &k10);
    // compute the ifft
    let plan = FftPlan::new(l);
    let out = ifft(at_roots.into_raw_vec().as_slice(), &plan);
    // compose the kernel
    let kernel = {
        let tmp = Array::from_vec(out);
        tmp.view().split_complex().re.to_owned()
    };
    kernel
}

pub fn kernel<T>(dplr: &DPLRParams<Complex<T>>, step: f64, l: usize) -> Array1<T>
where
    T: Conjugate + FftNum + Float + Scalar<Real = T, Complex = Complex<T>> + ScalarOperand,
    <T as Scalar>::Real: FloatConst + NumOps<<T as Scalar>::Complex, <T as Scalar>::Complex>,
    <T as Scalar>::Complex: ScalarOperand,
{
    // initialize some constants
    let step = T::from(step).unwrap();
    let two = T::from(2).unwrap();
    // get the lambda matrix
    let lambda = dplr.lambda.clone();
    // collect the relevant terms for A
    let aterm = (dplr.c.conj(), dplr.q.conj());
    // collect the relevant terms for B
    let bterm = (dplr.b.clone(), dplr.p.clone());
    // generate omega
    let omega_l = omega_l::<T>(l);

    let g = omega_l.mapv(|i| (T::one() - i) * (T::one() + i).recip()) * (two * step.recip());
    let c = omega_l.mapv(|i| two * (T::one() + i).recip());
    // compute the cauchy matrix
    let k00 = cauchy(&(&aterm.0 * &bterm.0), &g, &lambda);
    let k01 = cauchy(&(&aterm.0 * &bterm.1), &g, &lambda);
    let k10 = cauchy(&(&aterm.1 * &bterm.0), &g, &lambda);
    let k11 = cauchy(&(&aterm.1 * &bterm.1), &g, &lambda);
    // compute the roots of unity
    let at_roots = &c * (&k00 - k01 * &k11.mapv(|i| (i + T::one()).recip()) * &k10);
    // compute the ifft
    let plan = FftPlan::new(l);
    let out = ifft(at_roots.into_raw_vec().as_slice(), &plan);
    // compose the kernel
    let kernel = {
        let tmp = Array::from_vec(out);
        tmp.view().split_complex().re.to_owned()
    };
    kernel
}

pub struct Kernel<T = f64> {
    kernel: Array1<T>,
}

impl<T> Kernel<T> {
    pub fn new(kernel: Array1<T>) -> Self {
        Self { kernel }
    }

    pub fn kernal(&self) -> &Array1<T> {
        &self.kernel
    }
}

impl<T> Kernel<T>
where
    T: Scalar<Real = T, Complex = Complex<T>>,
    <T as Scalar>::Real: Conjugate
        + FftNum
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
        let kernel = kernel_dplr::<T>(dplr, step, l);
        Self::new(kernel)
    }
}
