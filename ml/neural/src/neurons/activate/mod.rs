/*
    Appellation: activate <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # activate
//! 
//! This module contains the activation functions for the neurons.
pub use self::utils::*;

pub type ActivationFn<T = f64> = fn(T) -> T;

pub trait Activate<T> {

    fn activate(&mut self, args: &[T]) -> T;
}

pub(crate) mod utils {
    
    pub fn heavyside(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}