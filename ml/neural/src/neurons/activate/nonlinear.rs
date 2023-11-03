/*
    Appellation: nonlinear <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Activate;
use ndarray::prelude::{Array, Array1, Axis};
use ndarray::{Dimension, RemoveAxis, ScalarOperand};
use num::{Float, Zero};
use serde::{Deserialize, Serialize};

pub fn softmax<T>(args: Array1<T>) -> Array1<T>
where
    T: Float,
{
    let denom = args.mapv(|x| x.exp()).sum();
    args.mapv(|x| x.exp() / denom)
}

pub struct ReLU;

impl ReLU {
    pub fn compute<T: PartialOrd + Zero>(x: T) -> T {
        if x > T::zero() {
            x
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
        x.mapv(|x| Self::compute(x))
    }
}

pub struct Sigmoid;

impl Sigmoid {
    pub fn compute<T: Float>(x: T) -> T {
        T::one() / (T::one() + (-x).exp())
    }
}

impl<T, D> Activate<Array<T, D>> for Sigmoid
where
    D: Dimension,
    T: Float,
{
    fn activate(&self, x: Array<T, D>) -> Array<T, D> {
        x.mapv(|x| Self::compute(x))
    }
}

pub fn softmax_axis<T, D>(args: Array<T, D>, axis: Option<usize>) -> Array<T, D>
where
    T: Float + ScalarOperand,
    D: Dimension + RemoveAxis,
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
pub struct Softmax {
    axis: Option<usize>,
}

impl Softmax {
    pub fn new(axis: Option<usize>) -> Self {
        Self { axis }
    }
}

impl<T, D> Activate<Array<T, D>> for Softmax
where
    D: Dimension + RemoveAxis,
    T: Float + ScalarOperand,
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
