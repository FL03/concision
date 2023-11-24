/*
    Appellation: activate <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # activate
//!
//! This module contains the activation functions for the neurons.
pub use self::{activator::*, binary::*, linear::*, nonlinear::*, utils::*};

pub(crate) mod activator;
pub(crate) mod binary;
pub(crate) mod linear;
pub(crate) mod nonlinear;

pub type ActivationFn<T = f64> = fn(T) -> T;

pub type ActivateDyn<T = f64, D = Ix2> = Box<dyn Activate<T, D>>;

use ndarray::prelude::{Array, Dimension, Ix2};

pub trait Activate<T = f64, D = Ix2>
where
    D: Dimension,
{
    fn activate(&self, args: &Array<T, D>) -> Array<T, D>;
}

// pub trait ActivateExt<T = f64, D = Ix2>
// where
//     D: Dimension,
// {
//     fn new() -> Self;

//     fn method(&self) -> impl;
// }

impl<T, D> Activate<T, D> for fn(&Array<T, D>) -> Array<T, D>
where
    D: Dimension,
{
    fn activate(&self, args: &Array<T, D>) -> Array<T, D> {
        self.call((args,))
    }
}

impl<T, D> Activate<T, D> for Box<dyn Activate<T, D>>
where
    D: Dimension,
{
    fn activate(&self, args: &Array<T, D>) -> Array<T, D> {
        self.as_ref().activate(args)
    }
}

pub trait Objective<T = f64, D = Ix2>: Activate<T, D>
where
    D: Dimension,
{
    fn gradient(&self, args: &Array<T, D>) -> Array<T, D>;
}

impl<T, D> Objective<T, D> for fn(&Array<T, D>) -> Array<T, D>
where
    D: Dimension,
{
    fn gradient(&self, args: &Array<T, D>) -> Array<T, D> {
        self.call((args,))
    }
}

impl<T, D> Activate<T, D> for Box<dyn Objective<T, D>>
where
    D: Dimension,
{
    fn activate(&self, args: &Array<T, D>) -> Array<T, D> {
        self.as_ref().activate(args)
    }
}

impl<T, D> Objective<T, D> for Box<dyn Objective<T, D>>
where
    D: Dimension,
{
    fn gradient(&self, args: &Array<T, D>) -> Array<T, D> {
        self.as_ref().gradient(args)
    }
}

pub(crate) mod utils {
    use ndarray::prelude::{Array, Axis, Dimension, NdFloat};
    use ndarray::RemoveAxis;
    use num::{Float, One, Zero};

    pub fn linear_activation<T>(args: &T) -> T
    where
        T: Clone,
    {
        args.clone()
    }

    pub fn heavyside<T>(args: &T) -> T
    where
        T: One + PartialOrd + Zero,
    {
        if args > &T::zero() {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use computare::prelude::RoundTo;
    use ndarray::array;

    #[test]
    fn test_heavyside() {
        let exp = array![0.0, 1.0, 1.0];
        let args = array![-1.0, 0.0, 1.0];

        let res = Heavyside::new().activate(&args);
        assert_eq!(res, exp);
    }

    #[test]
    fn test_linear() {
        let exp = array![0.0, 1.0, 2.0];
        let args = array![0.0, 1.0, 2.0];

        let res = Linear::new().activate(&args);
        assert_eq!(res, exp);
    }

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
