/*
    Appellation: dplr <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Diagonal Plus Low Rank (DPLR)
//!
//!
use super::nplr::NPLR;
use crate::core::prelude::{Conjugate, SquareRoot};
use crate::prelude::S4Float;
use ndarray::prelude::{Array1, Array2, Axis};
use ndarray::{LinalgScalar, ScalarOperand};
use ndarray_linalg::{Eigh, IntoTriangular, Lapack, Scalar, UPLO};
use num::complex::{Complex, ComplexFloat};
use num::{FromPrimitive, Num, One, Signed};
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul, Neg};

// #[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[derive(Clone, Debug, PartialEq)]
pub struct DPLR<T = f64>
where
    T: Clone + Num + Scalar,
{
    pub lambda: Array2<Complex<<T as Scalar>::Real>>,
    pub p: Array1<T>,
    pub b: Array1<T>,
    pub v: Array2<T>,
}

impl<T> DPLR<T>
where
    T: Conjugate + Lapack + Scalar + ScalarOperand + Signed + SquareRoot,
    T: Add<Complex<<T as Scalar>::Real>, Output = Complex<<T as Scalar>::Real>>
        + Mul<Complex<<T as Scalar>::Real>, Output = Complex<<T as Scalar>::Real>>,
    Complex<<T as Scalar>::Real>: Mul<T, Output = Complex<<T as Scalar>::Real>>,
    Complex<<T as Scalar>::Real>:
        Mul<Complex<<T as Scalar>::Real>, Output = Complex<<T as Scalar>::Real>>,
    Complex<T>: Add<Complex<<T as Scalar>::Real>, Output = Complex<T>>
        + Mul<Complex<<T as Scalar>::Real>, Output = Complex<T>>,
{
    pub fn create(features: usize) -> Self {
        let (a, p, b) = NPLR::<T>::new(features).into();

        //
        let s = {
            let p2 = p.clone().insert_axis(Axis(1));
            &a + p2.dot(&p2.t())
        };
        //
        let sd = s.diag().mean().expect("Average of diagonal is NaN");

        let a = Array2::ones(s.dim()) * sd;

        // TODO: Fix this
        // let (ee, vv) = {
        //     let si =
        // };
        // let (e, v) = s.mapv(|i: T| T::from(Complex::new(i.re(), i.im()) * Complex::i().neg()).unwrap())
        //     .eigh(UPLO::Lower)
        //     .expect("");
        let (e, v) = s.conj().eigh(UPLO::Lower).expect("");

        // let a = a + &e * Complex::new(<<T as ComplexFloat>::Real>::one(), <<T as ComplexFloat>::Real>::one());
        let a = a + e.mapv(|i: <T as Scalar>::Real| {
            Complex::new(i.re(), i.im())
                * Complex::new(<T as Scalar>::Real::one(), <T as Scalar>::Real::one().neg())
        });
        let p = v.conj().t().dot(&p);
        let b = v.conj().t().dot(&b);
        Self {
            lambda: a,
            p,
            b,
            v: v.clone(),
        }
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
