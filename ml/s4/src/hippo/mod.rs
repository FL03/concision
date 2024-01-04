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
    use crate::core::prelude::SquareRoot;
    use ndarray::prelude::{Array1, Array2, Axis};
    use ndarray::ScalarOperand;
    use ndarray_linalg::{IntoTriangular, UPLO};
    use num::traits::{Num, NumCast, Signed};

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use dplr::DPLR;
    use nplr::NPLR;

    use crate::core::prelude::Conjugate;
    use ndarray::prelude::{Array, Axis};
    use num::complex::ComplexFloat;

    #[test]
    fn test_hippo() {
        let features = 10;

        let a = hippo::<f64>(features);
        let b = HiPPO::<f64>::new(features);
        assert_eq!(&a, b.as_ref());
    }

    #[test]
    fn test_low_rank() {
        let features = 8;

        let nplr = NPLR::<f64>::new(features);
        let dplr = DPLR::<f64>::new(features);

        let hippo = nplr.a.clone();

        let v = dplr.v.clone();
        // compute the conjugate transpose of the eigenvectors
        let vc = v.conj().t().to_owned();
        // create a two-dimensional array from the diagonal of the lambda matrix
        let lambda = {
            let ld = dplr.lambda.diag().to_owned();
            Array::from_diag(&ld)
        };
        // reshape the p values
        let p = nplr.p.clone().insert_axis(Axis(1));
        let pc = dplr.p.clone().insert_axis(Axis(1));
        // compute the expected values for NPLR
        let a = v.dot(&lambda).dot(&vc) - &p.dot(&p.t());
        // compute the expected values for DPLR
        let b = {
            let tmp = lambda - pc.dot(&pc.conj().t());
            v.dot(&tmp).dot(&vc)
        };

        let err_nplr = {
            let tmp = (&a - &hippo).mapv(|i| i.abs());
            tmp.mean().unwrap()
        };
        assert!(
            err_nplr <= 1e-4,
            "Actual: {:?}\nTolerance: {:?}",
            err_nplr,
            1e-4
        );
        let err_dplr = {
            let tmp = (&b - &hippo).mapv(|i| i.abs());
            println!("{:?}", &tmp);
            tmp.mean().unwrap()
            // tmp
        };
        assert!(
            err_dplr <= 1e-4,
            "Actual: {:?}\nTolerance: {:?}",
            err_dplr,
            1e-4
        );
    }

    // #[test]
    // fn test_nplr() {
    //     let features = 10;

    //     let a = HiPPO::<f64>::new(features).nplr();
    //     let b = NPLR::<f64>::new(features);
    //     assert_eq!(&a, &b);
    // }
}
