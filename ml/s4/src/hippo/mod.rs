/*
    Appellation: hippo <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # HiPPO
//!
//!
pub use self::{hippo::*, kinds::*, utils::*};

pub(crate) mod hippo;
pub(crate) mod kinds;

pub mod dplr;
pub mod nplr;

pub struct LowRank {
    pub mode: Mode,
}

pub(crate) mod utils {
    use super::dplr::DPLR;
    use super::nplr::NPLR;
    use crate::core::prelude::{rangespace, Conjugate};
    use crate::prelude::S4Float;
    use ndarray::prelude::{Array2, Axis};
    use ndarray::{LinalgScalar, ScalarOperand};
    use ndarray_linalg::{Eigh, IntoTriangular, UPLO};
    use num::complex::{Complex, ComplexFloat};
    // use num::traits::{Float, FloatConst};
    use std::ops::Neg;

    pub(crate) fn make_hippo<T>(features: usize) -> Array2<T>
    where
        T: ComplexFloat + ScalarOperand,
    {
        let base = rangespace(features).insert_axis(Axis(1));
        let p = (&base * T::from(2).unwrap() + T::one()).mapv(T::sqrt);
        let mut a = &p * &p.t();
        a = &a.into_triangular(UPLO::Lower) - &base.diag();
        -a
    }

    pub(crate) fn make_nplr_hippo<T>(features: usize) -> NPLR<T>
    where
        T: ComplexFloat + ScalarOperand,
    {
        let hippo = make_hippo(features);

        let base = rangespace((features,));
        let p = (&base + T::one() / T::from(2).unwrap()).mapv(T::sqrt);
        let b = (&base * T::from(2).unwrap() + T::one()).mapv(T::sqrt);
        NPLR { a: hippo, p, b }
    }

    pub(crate) fn make_dplr_hippo<T>(features: usize) -> DPLR<T>
    where
        T: S4Float,
        Complex<<T as ComplexFloat>::Real>: LinalgScalar + ScalarOperand,
    {
        let (a, p, b) = make_nplr_hippo(features).into();

        //
        let s = &a
            + p.clone()
                .insert_axis(Axis(1))
                .dot(&p.clone().insert_axis(Axis(0)));
        //
        let sd = s.diag();

        let a = Array2::ones(s.dim()) * sd.mean().expect("Average of diagonal is NaN");

        // TODO: replace with eigh
        let (e, v) = &(&s * T::from((Complex::<<T as ComplexFloat>::Real>::i().neg())).unwrap())
            .eigh(UPLO::Lower)
            .expect("");
        let e = e.mapv(|x| T::from(x).unwrap());

        let a = a + &e * T::from(T::one().as_imag()).unwrap();
        let p = v.conj().t().dot(&p);
        let b = v.conj().t().dot(&b);
        DPLR {
            lambda: a,
            p,
            b,
            v: v.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dplr::DPLR;
    use nplr::NPLR;
    use num::Complex;

    #[test]
    fn test_hippo() {
        let features = 10;

        let a = make_hippo::<f64>(features);
        let b = HiPPO::<f64>::new(features);
        assert_eq!(&a, b.as_ref());
    }

    #[test]
    fn test_dplr() {
        let features = 10;

        let a = make_dplr_hippo::<Complex<f64>>(features);
        let b = HiPPO::<Complex<f64>>::new(features).dplr();
        let c = DPLR::<Complex<f64>>::new(features);
        assert_eq!(&a, &b);
        assert_eq!(&a, &c);
    }

    #[test]
    fn test_nplr() {
        let features = 10;

        let a = make_nplr_hippo::<f64>(features);
        let b = HiPPO::<f64>::new(features).nplr();
        let c = NPLR::<f64>::new(features);
        assert_eq!(&a, &b);
        assert_eq!(&a, &c);
    }
}
