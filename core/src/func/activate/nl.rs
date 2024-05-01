/*
    Appellation: sigmoid <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::{Array, Axis, Dimension, NdFloat, RemoveAxis};
use num::{Float, Zero};

pub fn relu<T>(args: &T) -> T
where
    T: Clone + PartialOrd + Zero,
{
    if args > &T::zero() {
        args.clone()
    } else {
        T::zero()
    }
}

pub fn sigmoid<T>(args: &T) -> T
where
    T: Float,
{
    T::one() / (T::one() + (-args.clone()).exp())
}

pub fn softmax<T, D>(args: &Array<T, D>) -> Array<T, D>
where
    D: Dimension,
    T: Float,
{
    let denom = args.mapv(|x| x.exp()).sum();
    args.mapv(|x| x.exp() / denom)
}

pub fn softmax_axis<T, D>(args: &Array<T, D>, axis: Option<usize>) -> Array<T, D>
where
    D: Dimension + RemoveAxis,
    T: NdFloat,
{
    let exp = args.mapv(|x| x.exp());
    if let Some(axis) = axis {
        let denom = exp.sum_axis(Axis(axis));
        exp / denom
    } else {
        let denom = exp.sum();
        exp / denom
    }
}

pub fn tanh<T>(args: &T) -> T
where
    T: Float,
{
    args.tanh()
}
