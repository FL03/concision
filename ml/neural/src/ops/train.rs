/*
   Appellation: train <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::BoxResult;
use ndarray::prelude::{Array2, Axis};

pub trait Train<T = f64> {
    fn train(&mut self, input: &Array2<T>, target: &Array2<T>) -> BoxResult<T>;

    fn train_batch(
        &mut self,
        batch_size: usize,
        input: &Array2<T>,
        target: &Array2<T>,
    ) -> BoxResult<T>
    where
        T: Clone + std::iter::Sum<T>,
    {
        let res = input
            .axis_chunks_iter(Axis(0), batch_size)
            .zip(target.axis_chunks_iter(Axis(0), batch_size))
            .map(|(x, y)| self.train(&x.to_owned(), &y.to_owned()).expect(""))
            .sum();
        Ok(res)
    }
}
