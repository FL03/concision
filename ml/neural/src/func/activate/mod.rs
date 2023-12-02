/*
    Appellation: activate <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # activate
//!
//! This module contains the activation functions for the neurons.
pub use self::{activator::*, binary::*, linear::*, nl::*, utils::*};

pub(crate) mod activator;
pub(crate) mod binary;
pub(crate) mod linear;
pub(crate) mod nl;

// use crate::core::prelude::ShapeResult;
use ndarray::prelude::{Array, Dimension, Ix2, IxDyn};

pub type ActivationFn<T = f64, D = Ix2> = Box<dyn Fn(&Array<T, D>) -> Array<T, D>>;

pub type ActivateDyn<T = f64, D = IxDyn> = Box<dyn Activate<T, D>>;

pub trait Rho<T = f64> {
    fn rho(&self, args: &T) -> T;
}

impl<T, F> Rho<T> for F
where
    F: Fn(&T) -> T,
{
    fn rho(&self, args: &T) -> T {
        self.call((args,))
    }
}

pub trait Activate<T = f64, D = Ix2>
where
    D: Dimension,
{
    fn activate(&self, args: &Array<T, D>) -> Array<T, D>;
}

pub trait ActivateExt<T = f64, D = Ix2>: Activate<T, D> + Clone + 'static
where
    D: Dimension,
{
    fn boxed(&self) -> ActivateDyn<T, D> {
        Box::new(self.clone())
    }
}

impl<T, D, F> ActivateExt<T, D> for F
where
    D: Dimension,
    F: Activate<T, D> + Clone + 'static,
{
}

impl<T, D, F> Activate<T, D> for F
where
    D: Dimension,
    F: Fn(&Array<T, D>) -> Array<T, D>,
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

pub trait Gradient<T = f64, D = Ix2>
where
    D: Dimension,
{
    fn gradient(&self, args: &Array<T, D>) -> Array<T, D>;
}

// impl<T, D> Objective<T, D> for fn(&Array<T, D>) -> Array<T, D>
// where
//     D: Dimension,
// {
//     fn gradient(&self, args: &Array<T, D>) -> Array<T, D> {
//         self.call((args,))
//     }
// }

impl<T, D> Gradient<T, D> for Box<dyn Gradient<T, D>>
where
    D: Dimension,
{
    fn gradient(&self, args: &Array<T, D>) -> Array<T, D> {
        self.as_ref().gradient(args)
    }
}

pub(crate) mod utils {
    use num::{One, Zero};

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_heavyside() {
        let exp = array![0.0, 0.0, 1.0];
        let args = array![-1.0, 0.0, 1.0];

        assert_eq!(Heavyside::new().activate(&args), exp);
        assert_eq!(Heavyside(&args), exp);
    }

    #[test]
    fn test_linear() {
        let exp = array![0.0, 1.0, 2.0];
        let args = array![0.0, 1.0, 2.0];

        assert_eq!(Linear::new().activate(&args), exp);
        assert_eq!(Linear(&args), exp);
    }
}
