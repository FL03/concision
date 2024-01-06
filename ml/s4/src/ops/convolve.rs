/*
    Appellation: convolve <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::{concat_iter, Power};
use crate::prelude::powmat;
use ndarray::linalg::Dot;
use ndarray::prelude::{s, Array, Array1, Array2, Axis};
use ndarray::ScalarOperand;
use num::Num;

pub fn convolve() {}

pub fn k_convolve<T>(a: &Array2<T>, b: &Array2<T>, c: &Array2<T>, l: usize) -> Array1<T>
where
    T: Num + ScalarOperand,
    Array2<T>: Dot<Array2<T>, Output = Array2<T>> + Dot<Array1<T>, Output = Array1<T>>,
{
    let b = b.clone().remove_axis(Axis(1));
    let mut res = Array2::<T>::zeros((l, a.shape()[0]));
    for i in 0..l {
        let out = c.dot(&a.pow(i).dot(&b));
        res.slice_mut(s![i, ..]).assign(&out);
    }
    res.remove_axis(Axis(1))
}


pub fn k_conv<T>(a: &Array2<T>, b: &Array2<T>, c: &Array2<T>, l: usize) -> Array1<T>
where
    T: Num + ScalarOperand,
    Array2<T>: Dot<Array2<T>, Output = Array2<T>> + Dot<Array1<T>, Output = Array1<T>>,
{
    let mut store = Vec::new();
    for i in 0..l {
        let tmp = c.dot(&a.pow(i).dot(b));
        store.extend(tmp);
    
    }
    Array::from_iter(store)
}

pub struct Filter {
    
}