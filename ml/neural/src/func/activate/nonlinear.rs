/*
    Appellation: nonlinear <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Activate, ActivateMethod};
use ndarray::prelude::{Array, Axis, Dimension, NdFloat};
use ndarray::RemoveAxis;
use num::{Float, One, Zero};
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

    pub fn gradient<T, D>(args: &Array<T, D>) -> Array<T, D>
    where
        D: Dimension,
        T: Clone + One + PartialOrd + Zero,
    {
        args.mapv(|x| Self::derivative(x))
    }

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

impl<T> ActivateMethod<T> for ReLU
where
    T: PartialOrd + Zero,
{
    fn rho(&self, x: T) -> T {
        Self::relu(x)
    }
}

impl<T, D> Activate<T, D> for ReLU
where
    D: Dimension,
    T: Float,
{
    fn activate(&self, x: &Array<T, D>) -> Array<T, D> {
        x.mapv(|x| Self::relu(x))
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

    pub fn gradient<T, D>(args: &Array<T, D>) -> Array<T, D>
    where
        D: Dimension,
        T: Float,
    {
        args.mapv(|x| Self::derivative(x))
    }

    pub fn sigmoid<T>(x: T) -> T
    where
        T: Float,
    {
        T::one() / (T::one() + (-x).exp())
    }
}

impl<T> ActivateMethod<T> for Sigmoid
where
    T: Float,
{
    fn rho(&self, x: T) -> T {
        Self::sigmoid(x)
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

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }

    pub fn derivative<T>(x: T) -> T
    where
        T: Float,
    {
        T::one() - x.tanh().powi(2)
    }

    pub fn gradient<T, D>(args: &Array<T, D>) -> Array<T, D>
    where
        D: Dimension,
        T: Float,
    {
        args.mapv(|x| Self::derivative(x))
    }

    pub fn tanh<T>(x: T) -> T
    where
        T: Float,
    {
        x.tanh()
    }
}

impl<T> ActivateMethod<T> for Tanh
where
    T: Float,
{
    fn rho(&self, x: T) -> T {
        x.tanh()
    }
}

impl<T, D> Activate<T, D> for Tanh
where
    D: Dimension,
    T: Float,
{
    fn activate(&self, x: &Array<T, D>) -> Array<T, D> {
        x.mapv(|x| Self::tanh(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use computare::prelude::RoundTo;
    use ndarray::array;

    #[test]
    fn test_relu() {
        let exp = array![0.0, 0.0, 3.0];
        let args = array![-1.0, 0.0, 3.0];

        let res = ReLU::new().activate(&args);
        assert_eq!(res, exp);
    }

    #[test]
    fn test_sigmoid() {
        let exp = array![0.73105858, 0.88079708, 0.95257413];
        let args = array![1.0, 2.0, 3.0];

        let res = Sigmoid::new().activate(&args).mapv(|i| i.round_to(8));
        assert_eq!(res, exp);
    }

    #[test]
    fn test_softmax() {
        let exp = array![0.09003057, 0.24472847, 0.66524096];
        let args = array![1.0, 2.0, 3.0];

        let res = Softmax::new(None).activate(&args).mapv(|i| i.round_to(8));
        assert_eq!(res, exp);
    }

    #[test]
    fn test_tanh() {
        let exp = array![0.76159416, 0.96402758, 0.99505475];
        let args = array![1.0, 2.0, 3.0];

        let res = Tanh::new().activate(&args).mapv(|i| i.round_to(8));
        assert_eq!(res, exp);
    }
}
