/*
    Appellation: activate <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # activate
//!
//! This module contains the activation functions for the neurons.
pub use self::{activator::*, binary::*, nonlinear::*, utils::*};

pub(crate) mod activator;
pub(crate) mod binary;
pub(crate) mod nonlinear;

pub type ActivationFn<T = f64> = fn(T) -> T;

pub struct LinearActivation;

impl LinearActivation {
    pub fn method<T>() -> ActivationFn<T> {
        |x| x
    }
}

impl<T> Activate<T> for LinearActivation {
    fn activate(&self, x: T) -> T {
        Self::method()(x)
    }
}

pub trait ActivationMethod {
    fn method_name(&self) -> &str;
}

pub trait Activate<T> {
    fn activate(&self, x: T) -> T;
}

impl<F, T> Activate<T> for F
where
    F: Fn(T) -> T,
{
    fn activate(&self, x: T) -> T {
        self.call((x,))
    }
}

pub(crate) mod utils {}
