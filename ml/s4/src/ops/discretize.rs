/*
    Appellation: discretize <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use faer::prelude::{FaerMat, IntoFaer, SolverCore};
use faer::IntoNdarray;
use faer_core::zip::ViewMut;
use faer_core::{ComplexField, Conjugate, SimpleEntity};
use ndarray::prelude::{Array2, NdFloat};
use num::ToPrimitive;

pub fn discretize<T>(
    a: &Array2<T>,
    b: &Array2<T>,
    c: &Array2<T>,
    _d: &Array2<T>,
    step: T,
) -> anyhow::Result<(Array2<T>, Array2<T>, Array2<T>)>
where
    T: NdFloat + Conjugate + SimpleEntity,
    <T as Conjugate>::Canonical: ComplexField + SimpleEntity + ToPrimitive,
{
    let ss = step / T::from(2).unwrap(); // half step
    let eye = Array2::<T>::eye(a.shape()[0]);
    let bl = &eye - a * ss;
    let be = {
        let mut tmp = bl.view().into_faer().qr().inverse();
        let arr = &tmp.view_mut().into_ndarray();
        arr.mapv(|i| T::from(i).unwrap())
    };
    let ab = be.dot(&(&eye + a * ss));
    let bb = (b * ss).dot(&b.t());

    Ok((ab, bb, c.clone()))
}

pub enum DiscretizeArgs {}

pub struct Discretize<T = f64> {
    pub step: T,
}

pub struct Discretized<T> {
    pub a: Array2<T>,
    pub b: Array2<T>,
    pub c: Array2<T>,
}
