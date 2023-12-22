/*
    Appellation: hippo <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::{linarr, tril};
use ndarray::prelude::{Array, Array2, Ix2, NdFloat};
use num::Float;

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

pub struct HiPPO;
