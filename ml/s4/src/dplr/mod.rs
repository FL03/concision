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
    use ndarray_linalg::eigh::Eigh;
    use ndarray_linalg::{IntoTriangular, UPLO};
    use num::complex::ComplexFloat;
    use num::Complex;

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

    pub fn make_dplr_hippo(
        features: usize,
    ) -> (
        Array2<Complex<f64>>,
        Array2<Complex<f64>>,
        Array2<Complex<f64>>,
        Array2<Complex<f64>>,
    ) {
        let (a, p, b) = make_nplr_hippo::<Complex<f64>>(features);
        let p = p.insert_axis(Axis(1));
        let b = b.insert_axis(Axis(1));

        //
        let s = &a + p.dot(&p.t());
        //
        let sd = s.diag();

        let a = Array2::ones(s.dim()) * sd.mean().expect("Average of diagonal is NaN");

        // TODO: replace with eigh
        let (e, v) = &(&s * (-1.0).as_imag()).eigh(UPLO::Lower).expect("");
        let e = e.mapv(|x| x.as_complex());

        let a = a + &e * 1.0.as_imag();
        let p = v.conj().t().dot(&p);
        let b = v.conj().t().dot(&b);
        (a, p, b, v.clone())
    }
}