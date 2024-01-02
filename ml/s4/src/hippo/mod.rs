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
    use crate::core::prelude::{rangespace, Conjugate, SquareRoot};
    use crate::prelude::S4Float;
    use ndarray::prelude::{Array1, Array2, Axis};
    use ndarray::{LinalgScalar, ScalarOperand};
    use ndarray_linalg::{Eigh, IntoTriangular, Scalar, UPLO};
    use num::complex::{Complex, ComplexFloat};
    use num::traits::{FromPrimitive, Num, NumCast, Signed};
    use std::ops;

    pub fn genspace<T: NumCast>(features: usize) -> Array1<T> {
        Array1::from_iter((0..features).map(|x| T::from(x).unwrap()))
    }

    pub(crate) fn hippo<T>(features: usize) -> Array2<T>
    where
        T: Num + NumCast + ScalarOperand + Signed + SquareRoot,
    {
        let base = genspace::<T>(features).insert_axis(Axis(1));
        let p = (&base * T::from(2).unwrap() + T::one()).mapv(T::sqrt);
        let mut a = &p * &p.t();
        a = &a.into_triangular(UPLO::Lower) - &Array2::from_diag(&genspace::<T>(features));
        -a
    }

    pub(crate) fn make_hippo<T>(features: usize) -> Array2<T>
    where
        T: ComplexFloat + ScalarOperand,
    {
        let base = rangespace((features, 1));
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
        <T as ComplexFloat>::Real: FromPrimitive,
        Complex<<T as Scalar>::Real>:
            ops::Mul<Complex<<T as ComplexFloat>::Real>, Output = Complex<<T as Scalar>::Real>>,
        Complex<<T as ComplexFloat>::Real>: LinalgScalar + ScalarOperand,
        Array2<Complex<<T as ComplexFloat>::Real>>: ops::Add<
                Array1<Complex<<T as Scalar>::Real>>,
                Output = Array2<Complex<<T as ComplexFloat>::Real>>,
            > + ops::Mul<
                Array2<<T as ComplexFloat>::Real>,
                Output = Array2<Complex<<T as ComplexFloat>::Real>>,
            > + ops::Mul<
                Array2<Complex<<T as ComplexFloat>::Real>>,
                Output = Array2<Complex<<T as ComplexFloat>::Real>>,
            > + ops::Mul<
                Array2<Complex<<T as Scalar>::Real>>,
                Output = Array2<Complex<<T as ComplexFloat>::Real>>,
            >,
    {
        let (a, p, b) = make_nplr_hippo::<T>(features).into();

        //
        let s = {
            let p2 = p.clone().insert_axis(Axis(1));
            &a + p2.dot(&p2.t())
        };
        //
        let sd = s.diag().mapv(|i: T| Complex::new(i.re(), i.im()));

        let a = Array2::<Complex<<T as Scalar>::Real>>::ones(s.dim())
            * sd.mean().expect("Average of diagonal is NaN");

        // TODO: Fix this
        // let (ee, vv) = {
        //     let si =
        // };
        // let (e, v) = s.mapv(|i: T| T::from(Complex::new(i.re(), i.im()) * Complex::i().neg()).unwrap())
        //     .eigh(UPLO::Lower)
        //     .expect("");
        let (e, v) = s.conj().eigh(UPLO::Lower).expect("");
        // compute the conjugate transpose of the eigenvectors

        // let a = a + &e * Complex::new(<<T as ComplexFloat>::Real>::one(), <<T as ComplexFloat>::Real>::one());
        // let a = a + e.mapv(|i: <T as Scalar>::Real| Complex::new(i.re(), i.im()) * Complex::i());
        let lambda = {
            let a = Array2::ones(s.dim());
            a
        };
        DPLR {
            lambda,
            p: v.conj().t().dot(&p),
            b: v.conj().t().dot(&b),
            v,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::prelude::Conjugate;
    use dplr::DPLR;
    use nplr::NPLR;

    use ndarray::prelude::{Array, Axis, Dimension};
    use ndarray::ScalarOperand;
    use ndarray_linalg::aclose;
    use num::{Num, Signed};

    #[test]
    fn test_hippo() {
        let features = 10;

        let a = hippo::<f64>(features);
        let b = HiPPO::<f64>::new(features);
        assert_eq!(&a, b.as_ref());
    }

    fn close<T, D>(a: &Array<T, D>, b: &Array<T, D>, atol: T)
    where
        D: Dimension,
        T: Num + ScalarOperand + Signed + PartialOrd + std::fmt::Debug,
    {
        (a - b).for_each(|i| assert!(i.abs() <= atol, "Actual: {:?}\nTolerance: {:?}", i, atol))
    }

    #[test]
    fn test_dplr() {
        let features = 10;

        // let a = make_dplr_hippo::<Complex<f64>>(features);
        // let b = HiPPO::<Complex<f64>>::new(features).dplr();
        let nplr = NPLR::<f64>::new(features);
        let dplr = DPLR::<f64>::create(features);

        let hippo = nplr.a.clone();

        let v = dplr.v.clone();
        let vc = v.conj().t().to_owned();

        let lambda = dplr.lambda.diag().to_owned().mapv(|i| i.re);

        let p = nplr.p.insert_axis(Axis(1));

        let pc = dplr.p.insert_axis(Axis(1));

        let a = v.dot(&lambda).dot(&vc) - &p.dot(&p.t());
        let b = v.dot(&(lambda - pc.dot(&pc.t()))).dot(&vc);

        close(&hippo, &a, 1e-4);
        // close(&hippo, &b, 1e-4);
        // assert_eq!(&a, &b);
    }

    // #[test]
    // fn test_nplr() {
    //     let features = 10;

    //     let a = HiPPO::<f64>::new(features).nplr();
    //     let b = NPLR::<f64>::new(features);
    //     assert_eq!(&a, &b);
    // }
}
