/*
    Appellation: specs <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::BoxResult;
use crate::func::loss::Loss;
use ndarray::prelude::{Array, Axis, Ix2};
use num::Float;

pub trait Backward<T>: Forward<T> {
    fn backward(&mut self, args: &T, grad: &T);
}

pub trait Forward<T> {
    type Output;

    fn forward(&self, args: &T) -> Self::Output;
}

pub trait ForwardIter<T, I>: Forward<T> + IntoIterator<Item = I>
where
    I: Forward<T, Output = Self::Output>,
{
}

impl<S, T, I> ForwardIter<T, I> for S
where
    S: Forward<T> + IntoIterator<Item = I>,
    I: Forward<T, Output = Self::Output>,
{
}

// impl<S, T> ForwardIter<T> for S
// where
//     S: Forward<T> + IntoIterator<Item = dyn Forward<T, Output = Self::Output>>,
// {
// }

pub trait Batched {
    type Output;

    fn batch(&self, batch_size: usize) -> Self::Output;
}

pub trait Compile<T> {
    type Opt;

    fn compile(&mut self, loss: impl Loss<T>, optimizer: Self::Opt) -> BoxResult<()>;
}

pub trait Predict<T> {
    type Output;

    fn predict(&self, input: &T) -> BoxResult<Self::Output>;
}

impl<S, T, O> Predict<T> for S
where
    S: Forward<T, Output = O>,
{
    type Output = O;

    fn predict(&self, input: &T) -> BoxResult<O> {
        Ok(self.forward(input))
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

pub trait Module<T>: Compile<T> + Predict<T> {
    type Config;
}
