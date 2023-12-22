/*
    Appellation: hippo <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::{linarr, tril, AsComplex, Conjugate};
use crate::prelude::eig_csym;

use ndarray::prelude::{Array1, Array2, Ix2, NdFloat};
use num::{Complex, Float, FromPrimitive};

pub fn make_hippo<T>(features: usize) -> Array2<T>
where
    T: NdFloat,
{
    let base = linarr::<T, Ix2>((features, 1)).unwrap();
    let p = (&base * T::from(2).unwrap() + T::one()).mapv(T::sqrt);
    let mut a = &p * &p.t();
    a = tril(&a) - &base.diag();
    -a
}

pub fn make_nplr_hippo<T>(features: usize) -> (Array2<T>, Array1<T>, Array1<T>)
where
    T: NdFloat,
{
    let hippo = make_hippo(features);

    let base = Array1::linspace(T::zero(), T::from(features - 1).unwrap(), features);
    let p = (&base + T::one() / T::from(2).unwrap()).mapv(T::sqrt);
    let b = (&base * T::from(2).unwrap() + T::one()).mapv(T::sqrt);
    (hippo, p, b)
}

pub fn dplr_hippo(
    features: usize,
) -> (
    Array2<Complex<f64>>,
    Array2<Complex<f64>>,
    Array2<Complex<f64>>,
) {
    let (a, p, b) = make_nplr_hippo::<f64>(features);
    let a = a.mapv(|x| x.as_complex());
    let p = p
        .mapv(|x| Complex::new(x, 0.0))
        .into_shape((features, 1))
        .unwrap();
    let b = b
        .mapv(|x| Complex::new(x, 0.0))
        .into_shape((features, 1))
        .unwrap();

    //
    let s = &a + p.dot(&p.t());
    //
    let sd = s.diag();

    let a = Array2::ones(s.dim()) * sd.mean().expect("Average of diagonal is NaN");

    // TODO: replace with eigh
    let (e, v) = eig_csym(&(&s * Complex::new(0.0, -1.0)));
    let e = e.mapv(|x| Complex::new(x, 0.0));

    let a = a + &e * 1.0.as_imag();
    let p = v.conj().t().dot(&p);
    let b = v.conj().t().dot(&b);
    (a, p, b)
}

pub fn make_dplr_hippo<T>(features: usize) -> (Array2<T>, Array2<T>, Array2<T>)
where
    T: FromPrimitive + NdFloat,
{
    let (a, p, b) = make_nplr_hippo(features);
    // broadcast p into a 2D matrix
    let p = p.into_shape((features, 1)).unwrap();
    //
    let s = &a + p.dot(&p.t());
    //
    let sd = s.diag();

    let a = Array2::ones(s.dim()) * sd.mean().expect("Average of diagonal is NaN");

    // TODO: finish up by computing the eigh of s * -1j
    // let (e, v) = eig_sym(&ss);
    (a, p, b.into_shape((features, 1)).unwrap())
}

pub struct HiPPO<T = f64>(Array2<T>);

impl<T> HiPPO<T>
where
    T: Float,
{
    pub fn new(hippo: Array2<T>) -> Self {
        Self(hippo)
    }

    pub fn hippo(&self) -> &Array2<T> {
        &self.0
    }

    pub fn hippo_mut(&mut self) -> &mut Array2<T> {
        &mut self.0
    }
}

impl<T> HiPPO<T>
where
    T: NdFloat,
{
    pub fn create(features: usize) -> Self {
        Self(make_hippo(features))
    }
}
