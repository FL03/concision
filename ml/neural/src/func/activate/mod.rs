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

pub type BoxedActivation = Box<dyn Activation>;

pub type ActivateDyn<T = f64, D = Ix2> = Box<dyn Activate<T, D>>;

use ndarray::prelude::{Array, Dimension, Ix2};
use num::Float;

pub trait Activation<T = f64> {
    fn activate<D: Dimension>(&self, args: &Array<T, D>) -> Array<T, D>;
}



pub trait Activate<T = f64, D = Ix2>
where
    D: Dimension,
{
    fn activate(&self, args: &Array<T, D>) -> Array<T, D>;
}

// impl<T, D, S> Activate<T, D> for S where D: Dimension, S: Activation<T>, {
//     fn activate(&self, args: &Array<T, D>) -> Array<T, D> {
//         Activation::activate::<D>(self, args)
//     }
// }

impl<T, D> Activate<T, D> for Box<dyn Activate<T, D>>
where
    D: Dimension,
{
    fn activate(&self, args: &Array<T, D>) -> Array<T, D> {
        self.as_ref().activate(args)
    }
}

pub trait ActivateExt<T = f64, D = Ix2>: Activate<T, D>
where
    D: Dimension,
    T: Float,
{
    fn gradient(&self, args: &Array<T, D>) -> Array<T, D>;
}

// pub trait ActivateMethod<T> {
//     fn rho(&self, x: &T) -> T;
// }

// impl<F, T> ActivateMethod<T> for F
// where
//     F: Fn(&T) -> T,
// {
//     fn rho(&self, x: &T) -> T {
//         self.call((x,))
//     }
// }

pub(crate) mod utils {
    use ndarray::RemoveAxis;
    use ndarray::prelude::{Array, Axis, Dimension, NdFloat};
    use num::{Float, One, Zero};

    pub fn linear_activation<T>(x: &T) -> T
    where
        T: Clone,
    {
        x.clone()
    }

    pub fn heavyside<T>(x: &T) -> T
    where
        T: Clone + One + PartialOrd + Zero,
    {
        if x.clone() > T::zero() {
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

    pub fn sigmoid<T>(x: &T) -> T
    where
        T: Float,
    {
        T::one() / (T::one() + (-x.clone()).exp())
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

    pub fn tanh<T>(x: &T) -> T
    where
        T: Float,
    {
        x.tanh()
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
}
