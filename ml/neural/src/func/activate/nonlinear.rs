/*
    Appellation: nonlinear <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Activate;
use ndarray::prelude::{Array, Axis, Dimension, NdFloat};
use ndarray::RemoveAxis;
use num::{Float, Zero};
use serde::{Deserialize, Serialize};

pub fn softmax<T, D>(args: Array<T, D>) -> Array<T, D>
where
    D: Dimension,
    T: Float,
{
    let denom = args.mapv(|x| x.exp()).sum();
    args.mapv(|x| x.exp() / denom)
}

pub fn softmax_axis<T, D>(args: Array<T, D>, axis: Option<usize>) -> Array<T, D>
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

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct ReLU;

impl ReLU {
    pub fn relu<T>(args: T) -> T
    where
        T: PartialOrd + Zero,
    {
        if args > T::zero() {
            args
        } else {
            T::zero()
        }
    }
}

impl<T, D> Activate<Array<T, D>> for ReLU
where
    D: Dimension,
    T: Float,
{
    fn activate(&self, x: Array<T, D>) -> Array<T, D> {
        x.mapv(|x| Self::relu(x))
    }
}

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn sigmoid<T>(x: T) -> T
    where
        T: Float,
    {
        (T::one() + (-x).exp()).powi(-2)
    }

    pub fn derivative<T>(x: T) -> T
    where
        T: Float,
    {
        -(T::one() + (-x).exp()).powi(-2) * (-x).exp()
    }

    pub fn gradient<T, D>(args: &Array<T, D>) -> Array<T, D>
    where
        D: Dimension,
        T: Float,
    {
        args.mapv(|x| Self::derivative(x))
    }
}

impl<T, D> Activate<Array<T, D>> for Sigmoid
where
    D: Dimension,
    T: Float,
{
    fn activate(&self, x: Array<T, D>) -> Array<T, D> {
        x.mapv(|x| Self::sigmoid(x))
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

impl<T, D> Activate<Array<T, D>> for Softmax
where
    D: Dimension + RemoveAxis,
    T: NdFloat,
{
    fn activate(&self, x: Array<T, D>) -> Array<T, D> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use computare::prelude::RoundTo;
    use ndarray::array;

    #[test]
    fn test_softmax() {
        let exp = array![0.09003057, 0.24472847, 0.66524096];
        let args = array![1.0, 2.0, 3.0];

        let res = Activate::activate(&Softmax::new(None), args).mapv(|i| i.round_to(8));
        assert_eq!(res, exp);
    }
}
