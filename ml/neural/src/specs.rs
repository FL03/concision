/*
    Appellation: specs <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::BoxResult;
use crate::func::loss::Loss;
use ndarray::prelude::{Array, Axis, Dimension, Ix2};
use num::Float;

pub trait Compile<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    type Opt;

    fn compile(&mut self, loss: impl Loss<Array<T, D>>, optimizer: Self::Opt) -> BoxResult<()>;
}

pub trait Predict<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    type Output;

    fn predict(&self, input: &Array<T, D>) -> BoxResult<Self::Output>;

    fn predict_batch(&self, input: &[Array<T, D>]) -> BoxResult<Vec<Self::Output>> {
        let res = input.iter().map(|x| self.predict(x).expect("")).collect();
        Ok(res)
    }
}

pub trait Train<T = f64>
where
    T: Float,
{
    fn train(&mut self, input: &Array<T, Ix2>, target: &Array<T, Ix2>) -> BoxResult<T>;

    fn train_batch(
        &mut self,
        batch_size: usize,
        input: &Array<T, Ix2>,
        target: &Array<T, Ix2>,
    ) -> BoxResult<T>
    where
        T: std::iter::Sum<T>,
    {
        let res = input
            .axis_chunks_iter(Axis(0), batch_size)
            .zip(target.axis_chunks_iter(Axis(0), batch_size))
            .map(|(x, y)| self.train(&x.to_owned(), &y.to_owned()).expect(""))
            .sum();
        Ok(res)
    }
}
