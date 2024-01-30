/*
    Appellation: specs <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::ops::{Compile, Predict};

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

pub trait Module<T>: Compile<T> + Predict<T> {
    type Config;

    fn config(&self) -> &Self::Config;

    fn id(&self) -> &str;
}
