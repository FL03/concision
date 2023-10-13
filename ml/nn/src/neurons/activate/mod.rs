/*
    Appellation: activate <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # activate
//! 
//! This module contains the activation functions for the neurons.
pub use self::utils::*;

pub type ActivationFn<T = f64> = fn(T) -> T;

pub(crate) mod utils {}