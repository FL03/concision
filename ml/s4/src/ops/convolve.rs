/*
    Appellation: convolve <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::Power;
use crate::prelude::powmat;
use ndarray::linalg::Dot;
use ndarray::prelude::{s, Array, Array1, Array2, Axis};
use ndarray::ScalarOperand;
use ndarray_linalg::flatten;
use num::Num;

pub fn convolve() {}

pub fn k_convolve<T>(a: &Array2<T>, b: &Array2<T>, c: &Array2<T>, l: usize) -> Array1<T>
where
    T: Num + ScalarOperand,
    Array2<T>: Dot<Array2<T>, Output = Array2<T>> + Dot<Array1<T>, Output = Array1<T>>,
{
    Array::from_iter((0..l).map(|i| c.dot(&a.pow(i).dot(b)).sum()))
}
