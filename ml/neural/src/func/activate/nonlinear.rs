/*
    Appellation: nonlinear <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Activate, Objective};
use ndarray::prelude::{Array, Axis, Dimension, NdFloat};
use ndarray::RemoveAxis;
use num::{Float, One, Zero};
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }

    pub fn derivative<T>(x: T) -> T
    where
        T: One + PartialOrd + Zero,
    {
        if x > T::zero() {
            T::one()
        } else {
            T::zero()
        }
    }

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
}

impl<T, D> Activate<T, D> for ReLU
where
    D: Dimension,
    T: Clone + PartialOrd + Zero,
{
    fn activate(&self, x: &Array<T, D>) -> Array<T, D> {
        x.mapv(|x| Self::relu(&x))
    }
}

impl<T, D> Objective<T, D> for ReLU
where
    D: Dimension,
    T: Clone + One + PartialOrd + Zero,
{
    fn gradient(&self, args: &Array<T, D>) -> Array<T, D> {
        args.mapv(|x| Self::derivative(x))
    }
}

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }

    pub fn derivative<T>(x: T) -> T
    where
        T: Float,
    {
        (-x).exp() / (T::one() + (-x).exp()).powi(2)
    }

    pub fn sigmoid<T>(x: T) -> T
    where
        T: Float,
    {
        T::one() / (T::one() + (-x).exp())
    }
}

impl<T, D> Activate<T, D> for Sigmoid
where
    D: Dimension,
    T: Float,
{
    fn activate(&self, x: &Array<T, D>) -> Array<T, D> {
        x.mapv(|x| Self::sigmoid(x))
    }
}

impl<T, D> Objective<T, D> for Sigmoid
where
    D: Dimension,
    T: Float,
{
    fn gradient(&self, args: &Array<T, D>) -> Array<T, D> {
        args.mapv(|x| Self::derivative(x))
    }
}

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct Softmax {
    axis: Option<usize>,
}

impl Softmax {
    pub fn new(axis: Option<usize>) -> Self {
        Self { axis }
    }

    pub fn axis(&self) -> Option<usize> {
        self.axis
    }

    pub fn softmax<T, D>(args: Array<T, D>) -> Array<T, D>
    where
        D: Dimension,
        T: Float,
    {
        let denom = args.mapv(|x| x.exp()).sum();
        args.mapv(|x| x.exp() / denom)
    }

    pub fn softmax_axis<T, D>(&self, args: Array<T, D>) -> Array<T, D>
    where
        T: NdFloat,
        D: Dimension + RemoveAxis,
    {
        let exp = args.mapv(|x| x.exp());
        if let Some(axis) = self.axis {
            let denom = exp.sum_axis(Axis(axis));
            exp / denom
        } else {
            let denom = exp.sum();
            exp / denom
        }
    }
}

impl<T, D> Activate<T, D> for Softmax
where
    D: Dimension + RemoveAxis,
    T: NdFloat,
{
    fn activate(&self, x: &Array<T, D>) -> Array<T, D> {
        let exp = x.mapv(|x| x.exp());
        if let Some(axis) = self.axis {
            let denom = exp.sum_axis(Axis(axis));
            exp / denom
        } else {
            let denom = exp.sum();
            exp / denom
        }
    }
}

impl<T, D> Objective<T, D> for Softmax
where
    D: Dimension + RemoveAxis,
    T: NdFloat,
{
    fn gradient(&self, args: &Array<T, D>) -> Array<T, D> {
        let exp = args.mapv(|x| x.exp());
        if let Some(axis) = self.axis {
            let denom = exp.sum_axis(Axis(axis));
            exp / denom
        } else {
            let denom = exp.sum();
            exp / denom
        }
    }
}

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }

    pub fn derivative<T>(args: &T) -> T
    where
        T: Float,
    {
        T::one() - args.tanh().powi(2)
    }

    pub fn tanh<T>(args: &T) -> T
    where
        T: Float,
    {
        args.tanh()
    }
}

impl<T, D> Activate<T, D> for Tanh
where
    D: Dimension,
    T: Float,
{
    fn activate(&self, x: &Array<T, D>) -> Array<T, D> {
        x.mapv(Float::tanh)
    }
}

impl<T, D> Objective<T, D> for Tanh
where
    D: Dimension,
    T: Float,
{
    fn gradient(&self, args: &Array<T, D>) -> Array<T, D> {
        args.mapv(|x| Self::derivative(&x))
    }
}
