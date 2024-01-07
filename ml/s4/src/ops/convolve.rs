/*
    Appellation: convolve <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::Power;
use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Array1, Array2,};
use ndarray::ScalarOperand;
use num::Num;

/// Generates a large convolution kernal 
pub fn k_conv<T>(a: &Array2<T>, b: &Array2<T>, c: &Array2<T>, l: usize) -> Array1<T>
where
    T: Num + ScalarOperand,
    Array2<T>: Dot<Array2<T>, Output = Array2<T>>,
{
    let f = | i: usize | {
        c.dot(&a.pow(i).dot(b))
    };

    let mut store = Vec::new();
    for i in 0..l {
        store.extend(f(i));
    }
    Array::from_vec(store)
}

pub struct Filter {
    
}