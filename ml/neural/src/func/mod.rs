/*
   Appellation: func <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Functional
//! 
//! This module implements several functional aspects of the neural network.
//! 
//! ## Activation
//! 
//! The activation functions are implemented as structs that implement the `Fn` trait.
//! 
//! ## Loss
//! 
//! The loss functions are implemented as structs that implement the `Fn` trait.
pub use self::utils::*;

pub mod activate;
pub mod loss;


pub(crate) mod utils{
}

#[cfg(test)]
mod tests {
    // use super::*;

}
