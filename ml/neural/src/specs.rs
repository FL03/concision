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

pub trait ForwardIter<T, I>: Forward<T, Output = T> + IntoIterator<Item = I>
where
    I: Forward<T, Output = T>,
    T: Clone,
{
    fn forward_iter(&self, args: &T) -> Vec<T>;
}

impl<S, T, I> ForwardIter<T, I> for S
where
    S: Clone + Forward<T, Output = T> + IntoIterator<Item = I>,
    I: Forward<T, Output = T>,
    T: Clone,
{
    fn forward_iter(&self, args: &T) -> Vec<T> {
        let mut store = vec![args.clone()];

        for item in self.clone().into_iter() {
            let res = item.forward(store.last().unwrap());
            store.push(res)
        }

        store
    }
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
    type Params;

    fn config(&self) -> &Self::Config;

    fn id(&self) -> &str;

    fn name(&self) -> &str;

    fn params(&self) -> &Self::Params;
}
