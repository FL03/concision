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
use ndarray::prelude::{Array, ArrayD, Dimension, Ix2, IxDyn};

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

pub trait A<T = f64>
where
    T: Clone,
{
    // fn activate<D: Dimension>(&self, args: &Array<T, D>) -> ShapeResult<Array<T, D>> {
    //     let shape = args.shape();
    //     let res = self.activate_dyn(&args.into_dyn());
    //     let res: Array<T, D> = res.to_shape(shape)?.to_owned();
    //     Ok(res)
    // }

    fn activate_dyn(&self, args: &ArrayD<T>) -> ArrayD<T>;
}

pub trait Activate<T = f64, D = Ix2>
where
    D: Dimension,
{
    fn activate(&self, args: &Array<T, D>) -> Array<T, D>;
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

pub trait Objective<T = f64, D = Ix2>: Activate<T, D>
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
