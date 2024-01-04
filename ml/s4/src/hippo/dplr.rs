/*
    Appellation: dplr <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Diagonal Plus Low Rank (DPLR)
//!
//!
use super::nplr::NPLR;
use crate::core::prelude::{AsComplex, Conjugate, SquareRoot};
use ndarray::prelude::{Array, Array1, Array2, Axis};
use ndarray::ScalarOperand;
use ndarray_linalg::{Eigh, Lapack, Scalar, UPLO};
use num::traits::NumOps;
use num::{Complex, Num, Signed};
use serde::{Deserialize, Serialize};
use std::ops::{Mul, Neg};

pub(crate) trait DPLRScalar:
    AsComplex
    + Conjugate
    + Scalar<Real = Self>
    + ScalarOperand
    + Signed
    + SquareRoot
    + NumOps
    + NumOps<Complex<Self>, Complex<Self>>
where
    Complex<Self>: Lapack,
    <Self as Scalar>::Real: Mul<Complex<Self>, Output = Complex<Self>>,
{
}

impl<T> DPLRScalar for T
where
    T: AsComplex
        + Conjugate
        + NumOps
        + NumOps<Complex<T>, Complex<T>>
        + Scalar<Real = T>
        + ScalarOperand
        + Signed
        + SquareRoot,
    Complex<T>: Lapack,
    <T as Scalar>::Real: Mul<Complex<T>, Output = Complex<T>>,
{
}

pub(crate) fn dplr<T>(features: usize) -> DPLR<T>
where
    T: DPLRScalar,
    Complex<T>: Lapack,
    <T as Scalar>::Real: Mul<Complex<T>, Output = Complex<T>>,
{
    let (a, p, b) = NPLR::<T>::new(features).into();

    //
    let s = {
        let p2 = p.clone().insert_axis(Axis(1));
        &a + p2.dot(&p2.t())
    };
    //
    let sd = s.diag();

    let lambda_re = Array::ones(sd.dim()) * sd.mean().expect("");

    let (e, v) = s
        .mapv(|i: T| i * Complex::i().neg())
        .eigh(UPLO::Lower)
        .expect("");

    let lambda = {
        // let lambda_im = e.mapv(|i| i * Complex::i());
        let iter = lambda_re
            .into_iter()
            .zip(e.into_iter())
            .map(|(i, j)| Complex::new(i, T::zero()) + T::from(j).unwrap() * Complex::i());
        Array::from_iter(iter)
    };
    let p = p.mapv(AsComplex::as_re);
    let b = b.mapv(AsComplex::as_re);
    DPLR {
        lambda,
        p: v.conj().t().dot(&p),
        b: v.conj().t().dot(&b),
        v,
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct DPLR<T = f64>
where
    T: Clone + Num,
{
    pub lambda: Array1<Complex<T>>,
    pub p: Array1<Complex<T>>,
    pub b: Array1<Complex<T>>,
    pub v: Array2<Complex<T>>,
}

impl<T> DPLR<T>
where
    T: AsComplex
        + Conjugate
        + NumOps
        + NumOps<Complex<T>, Complex<T>>
        + Scalar<Real = T>
        + ScalarOperand
        + Signed
        + SquareRoot,
    Complex<T>: Lapack,
    <T as Scalar>::Real: NumOps<Complex<T>, Complex<T>>,
{
    pub fn new(features: usize) -> Self {
        dplr(features)
    }
}

// impl<T> DPLR<T>
// where
//     T: S4Float,
//     <T as ComplexFloat>::Real: FromPrimitive,
//     Complex<<T as ComplexFloat>::Real>: LinalgScalar + ScalarOperand,
// {
//     pub fn new(features: usize) -> Self {
//         make_dplr_hippo(features)
//     }
// }
