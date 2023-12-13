/*
    Appellation: specs <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::BoxResult;
use crate::func::loss::Loss;
use ndarray::prelude::{Array, Array1, Axis, Dimension, Ix2};
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

    fn predict_batch(&self, input: &[Array<T, D>]) -> BoxResult<Array1<Self::Output>> {
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

pub trait Module<T = f64, D = Ix2>: Forward<Array<T, D>, Output=Array<T, D>>
where
    D: Dimension,
    T: Float,
{
    type Config;

    


}