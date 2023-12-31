/*
    Appellation: convolve <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::powmat;
use ndarray::prelude::{s, Array2, Axis, NdFloat};

pub fn convolve() {}

pub fn k_convolve<T>(a: &Array2<T>, b: &Array2<T>, c: &Array2<T>, l: usize) -> Array2<T>
where
    T: NdFloat,
{
    let b = b.clone().remove_axis(Axis(1));
    let mut res = Array2::<T>::zeros((l, a.shape()[0]));
    for i in 0..l {
        let tmp = powmat(a, i);
        let out = c.dot(&tmp.dot(&b));
        res.slice_mut(s![i, ..]).assign(&out);
    }
    res
}
