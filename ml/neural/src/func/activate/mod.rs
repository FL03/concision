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

pub type BoxedActivation<T = f64> = Box<dyn ActivateMethod<T>>;

use ndarray::prelude::{Array, Dimension, Ix2};
use num::Float;

pub trait Activate<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    fn activate(&self, args: &Array<T, D>) -> Array<T, D>;
}

pub trait RhoGradient<T = f64, D = Ix2>: Activate<T, D>
where
    D: Dimension,
    T: Float,
{
    fn gradient(&self, args: &Array<T, D>) -> Array<T, D>;
}

// impl<T, D, F> Rho<T, D> for F
// where
//     D: Dimension,
//     F: Fn(&Array<T, D>) -> Array<T, D>,
//     T: Float
// {
//     fn activate(&self, args: &Array<T, D>) -> Array<T, D> {
//         self.call((args,))
//     }
// }

// impl<T, D, A> Rho<T, D> for A
// where
//     A: Activate<T>,
//     D: Dimension,
//     T: Float
// {
//     fn activate(&self, args: &Array<T, D>) -> Array<T, D> {
//         args.mapv(|x| self.rho(x))
//     }
// }

pub trait ActivationMethod {
    fn method_name(&self) -> &str;
}

pub trait ActivateMethod<T> {
    fn rho(&self, x: T) -> T;
}

impl<F, T> ActivateMethod<T> for F
where
    F: Fn(T) -> T,
{
    fn rho(&self, x: T) -> T {
        self.call((x,))
    }
}

pub(crate) mod utils {
    use num::{One, Zero};

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
}

#[cfg(test)]
mod tests {
    // use super::*;
}
