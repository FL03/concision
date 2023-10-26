/*
    Appellation: activate <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # activate
//!
//! This module contains the activation functions for the neurons.
pub use self::{binary::*, nonlinear::*, utils::*};

pub(crate) mod binary;
pub(crate) mod nonlinear;

pub type ActivationFn<T = f64> = fn(T) -> T;

pub struct Linear;

impl Linear {
    pub fn act() -> ActivationFn {
        |x| x
    }
}

impl<T> Activator<T> for Linear {
    fn rho(x: T) -> T {
        x
    }
}

pub trait Activate<T> {
    fn activate(&self, x: T) -> T;
}

pub trait ActivateMethod<T> {
    fn method() -> fn(T) -> T;
}

pub trait Activator<T> {
    fn activate(&self, x: T) -> T {
        Self::rho(x)
    }

    fn rho(x: T) -> T;
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
