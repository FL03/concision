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

pub trait Activable<T> {
    fn activate(&self, rho: impl Activator<T>) -> T;
}

pub trait ActivateMethod<T> {
    fn activate(&self, x: T) -> T;
}

pub trait Activator<T> {
    fn activate(&self, x: T) -> T {
        Self::rho(x)
    }

    fn rho(x: T) -> T;
}

pub(crate) mod utils {}
