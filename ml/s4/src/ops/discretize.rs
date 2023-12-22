/*
    Appellation: discretize <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use faer::prelude::{FaerMat, IntoFaer, SolverCore};
use faer::IntoNdarray;
use faer_core::zip::ViewMut;
use faer_core::{Conjugate, SimpleEntity};
use ndarray::prelude::{Array2, NdFloat};
use nshare::{ToNalgebra, ToNdarray2};
use num::ToPrimitive;

pub fn discretize_nalgebra(
    a: &Array2<f64>,
    b: &Array2<f64>,
    c: &Array2<f64>,
    step: f64,
) -> anyhow::Result<(Array2<f64>, Array2<f64>, Array2<f64>)> {
    let ss = step / 2.0; // half step
    let eye = Array2::<f64>::eye(a.shape()[0]);
    let bl = &eye - a * ss;
    let be = {
        let tmp = bl.into_nalgebra().lu().try_inverse();
        if let Some(arg) = tmp {
            arg.into_ndarray2()
        } else {
            return Err(anyhow::anyhow!("Could not invert matrix"));
        }
    };
    let ab = be.dot(&(&eye + a * ss));
    let bb = (b * ss).dot(&b.t());

    Ok((ab, bb, c.clone()))
}

pub fn discretize_faer<T>(
    a: &Array2<T>,
    b: &Array2<T>,
    c: &Array2<T>,
    step: T,
) -> (Array2<T>, Array2<T>, Array2<T>)
where
    T: NdFloat + Conjugate + SimpleEntity,
    <T as Conjugate>::Canonical: faer_core::ComplexField + SimpleEntity + ToPrimitive,
{
    let ss = step / T::from(2).unwrap(); // half step
    let eye = Array2::<T>::eye(a.shape()[0]);
    let bl = &eye - a * ss;
    let be = {
        let mut tmp = bl.view().into_faer().partial_piv_lu().inverse();
        let arr = &tmp.view_mut().into_ndarray();
        arr.mapv(|i| T::from(i).unwrap())
    };
    let ab = be.dot(&(&eye + a * ss));
    let bb = (b * ss).dot(&b.t());

    (ab, bb, c.clone())
}

pub trait Discretize<T> {
    type Output;
    fn discretize(&self, step: impl num::Float) -> Self::Output;
}

pub enum DiscretizeArgs {}

pub struct Discretizer<T = f64> {
    pub step: T,
}

pub struct Discretized<T>(T);
