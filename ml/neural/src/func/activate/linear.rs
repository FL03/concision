/*
    Appellation: linear <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Gradient;
use ndarray::prelude::{Array, Dimension};
use num::One;
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct LinearActivation;

impl LinearActivation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn derivative<T>(_x: T) -> T
    where
        T: One,
    {
        T::one()
    }

    pub fn gradient<T, D>(args: &Array<T, D>) -> Array<T, D>
    where
        D: Dimension,
        T: Clone + One,
    {
        args.mapv(|x| Self::derivative(x))
    }

    pub fn linear<T: Clone>(args: &T) -> T {
        LinearActivation::method()(args)
    }

    pub fn method<T: Clone>() -> fn(&T) -> T {
        |x| x.clone()
    }

    pub fn rho<T>(args: T) -> T {
        args
    }
}

impl<T, D> Gradient<T, D> for LinearActivation
where
    D: Dimension,
    T: Clone + One,
{
    fn gradient(&self, args: &Array<T, D>) -> Array<T, D> {
        args.mapv(|x| Self::derivative(x))
    }
}

impl<T> Fn<(&T,)> for LinearActivation
where
    T: Clone,
{
    extern "rust-call" fn call(&self, args: (&T,)) -> T {
        args.0.clone()
    }
}

impl<T> FnMut<(&T,)> for LinearActivation
where
    T: Clone,
{
    extern "rust-call" fn call_mut(&mut self, args: (&T,)) -> T {
        args.0.clone()
    }
}

impl<T> FnOnce<(&T,)> for LinearActivation
where
    T: Clone,
{
    type Output = T;

    extern "rust-call" fn call_once(self, args: (&T,)) -> Self::Output {
        args.0.clone()
    }
}
