/*
    Appellation: dplr <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Diagonal Plus Low Rank (DPLR)
//!
//!
pub use self::{kinds::*, utils::*};

pub(crate) mod kinds;

pub mod hippo;

pub struct LowRank {
    pub mode: Mode,
}

pub(crate) mod utils {
    use crate::core::prelude::{rangespace, AsComplex, Conjugate};

    use ndarray::prelude::{Array1, Array2, Axis};
    use ndarray::ScalarOperand;
    use ndarray_linalg::{Eigh, IntoTriangular, Lapack, UPLO};
    use num::complex::{Complex, ComplexFloat};
    use num::FromPrimitive;

    pub fn make_hippo<T>(features: usize) -> Array2<T>
    where
        T: ComplexFloat + ScalarOperand,
    {
        let base = rangespace((features, 1));
        let p = (&base * T::from(2).unwrap() + T::one()).mapv(T::sqrt);
        let mut a = &p * &p.t();
        a = &a.into_triangular(UPLO::Lower) - &base.diag();
        -a
    }

    pub fn make_nplr_hippo<T>(features: usize) -> (Array2<T>, Array1<T>, Array1<T>)
    where
        T: ComplexFloat + ScalarOperand,
    {
        let hippo = make_hippo(features);

        let base = rangespace((features,));
        let p = (&base + T::one() / T::from(2).unwrap()).mapv(T::sqrt);
        let b = (&base * T::from(2).unwrap() + T::one()).mapv(T::sqrt);
        (hippo, p, b)
    }

    pub fn make_dplr_hippo<T>(features: usize) -> (Array2<T>, Array2<T>, Array2<T>, Array2<T>)
    where
        T: AsComplex + ComplexFloat + Conjugate + FromPrimitive + Lapack + ScalarOperand,
    {
        let (a, p, b) = make_nplr_hippo(features);
        let p = p.insert_axis(Axis(1));
        let b = b.insert_axis(Axis(1));

        //
        let s = &a + p.dot(&p.t());
        //
        let sd = s.diag();

        let a = Array2::ones(s.dim()) * sd.mean().expect("Average of diagonal is NaN");

        // TODO: replace with eigh
        let (e, v) = &(&s * T::from(T::one().neg().as_imag()).unwrap())
            .eigh(UPLO::Lower)
            .expect("");
        let e = e.mapv(|x| T::from(x).unwrap());

        let a = a + &e * T::from(T::one().as_imag()).unwrap();
        let p = v.conj().t().dot(&p);
        let b = v.conj().t().dot(&b);
        (a, p, b, v.clone())
    }
}
