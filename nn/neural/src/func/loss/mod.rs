/*
    Appellation: loss <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Loss Functions
//!
//! Loss functions consider the differences between predicted and target outputs.
//! Overall, neural network models aim to minimize the average loss by adjusting certain hyperparameters,
//! the weights and biases.

pub use self::kinds::*;

pub(crate) mod kinds;

pub mod reg;

pub trait Loss<T = f64> {
    type Output;

    fn loss(&self, pred: &T, target: &T) -> Self::Output;
}

pub trait LossWith {
    type Output;

    fn loss(&self, other: &Self) -> Self::Output;
}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
