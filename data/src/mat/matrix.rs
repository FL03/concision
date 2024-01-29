/*
   Appellation: matrix <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::Array2;

pub struct Matrix<T> {
    store: Array2<T>,
}

impl<T> Matrix<T> {
    pub fn new(store: Array2<T>) -> Self {
        Self { store }
    }
}

impl<T> AsRef<Array2<T>> for Matrix<T> {
    fn as_ref(&self) -> &Array2<T> {
        &self.store
    }
}
