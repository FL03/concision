/*
    Appellation: dplr <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Diagonal Plus Low Rank (DPLR)
//!
//!
use super::nplr::NPLR;
use crate::core::prelude::Conjugate;
use ndarray::prelude::{Array, Array1, Array2, Axis};
use ndarray::ScalarOperand;
use ndarray_linalg::{Eigh, Lapack, Scalar, UPLO};
use num::traits::NumOps;
use num::{Complex, Num};
use serde::{Deserialize, Serialize};
use std::ops::Neg;

pub(crate) fn dplr<T>(features: usize) -> DPLR<T>
where
    T: Scalar<Complex = Complex<T>, Real = T> + ScalarOperand,
    <T as Scalar>::Complex: Conjugate + Lapack,
    <T as Scalar>::Real: NumOps<<T as Scalar>::Complex, <T as Scalar>::Complex>,
{
    let (a, p, b) = NPLR::<T>::new(features).into();

    //
    let s = {
        // reshape the p-array from NPLR into a two-dimensional matrix
        let p2 = p.clone().insert_axis(Axis(1));
        // compute s
        &a + p2.dot(&p2.t())
    };
    // find the diagonal of s
    let sd = s.diag();
    // create a matrix from the diagonals of s
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
    let p = p.mapv(|i| Complex::new(i, T::zero()));
    let b = b.mapv(|i| Complex::new(i, T::zero()));
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
    T: Scalar<Complex = Complex<T>, Real = T> + ScalarOperand,
    <T as Scalar>::Complex: Conjugate + Lapack,
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

impl<T>
    From<(
        Array1<Complex<T>>,
        Array1<Complex<T>>,
        Array1<Complex<T>>,
        Array2<Complex<T>>,
    )> for DPLR<T>
where
    T: Clone + Num,
{
    fn from(
        (lambda, p, b, v): (
            Array1<Complex<T>>,
            Array1<Complex<T>>,
            Array1<Complex<T>>,
            Array2<Complex<T>>,
        ),
    ) -> Self {
        DPLR { lambda, p, b, v }
    }
}

impl<T> From<DPLR<T>>
    for (
        Array1<Complex<T>>,
        Array1<Complex<T>>,
        Array1<Complex<T>>,
        Array2<Complex<T>>,
    )
where
    T: Clone + Num,
{
    fn from(dplr: DPLR<T>) -> Self {
        (dplr.lambda, dplr.p, dplr.b, dplr.v)
    }
}
