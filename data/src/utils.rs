/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::Array2;
use ndarray::linalg::Dot;
use num::traits::Num;

/// Raise a matrix to a power
pub fn powmat<T>(a: &Array2<T>, n: usize) -> Array2<T>
where
    T: Clone + Num + 'static,
    Array2<T>: Dot<Array2<T>, Output = Array2<T>>,
{
    if !a.is_square() {
        panic!("Matrix must be square");
    }
    let mut res = Array2::<T>::eye(a.nrows());
    for _ in 0..n {
        res = res.dot(a);
    }
    res
}