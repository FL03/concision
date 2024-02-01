/*
    Appellation: hippo <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # HiPPO
//!
//!
pub(crate) use self::utils::*;
pub use self::{hippo::*, kinds::*};

pub(crate) mod hippo;
pub(crate) mod kinds;

pub mod dplr;
pub mod nplr;

pub struct LowRank {
    pub mode: Mode,
}

pub(crate) mod utils {
    use ndarray::{Array, Array2, Axis, ScalarOperand};
    use ndarray_linalg::{IntoTriangular, Scalar, UPLO};

    pub(crate) fn hippo<T>(features: usize) -> Array2<T>
    where
        T: Scalar + ScalarOperand,
    {
        let base = Array::from_iter((0..features).map(|i| T::from(i).unwrap()));
        let b2 = base.clone().insert_axis(Axis(1));
        let p = (&b2 * T::from(2).unwrap() + T::one()).mapv(T::sqrt);
        let mut a = &p * &p.t();
        a = &a.into_triangular(UPLO::Lower) - &Array2::from_diag(&base);
        -a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dplr::DPLR;
    use nplr::NPLR;

    use crate::core::prelude::Conjugate;
    use ndarray::prelude::*;
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
}
