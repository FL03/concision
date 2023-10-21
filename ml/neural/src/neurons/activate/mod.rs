/*
    Appellation: activate <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # activate
//!
//! This module contains the activation functions for the neurons.
pub use self::{common::*, utils::*};

pub(crate) mod common;

pub type ActivationFn<T = f64> = fn(T) -> T;

pub trait Activable<T> {
    fn activate(&self, args: &ndarray::Array1<T>) -> T;
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
