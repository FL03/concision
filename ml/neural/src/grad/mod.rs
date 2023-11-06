/*
    Appellation: grad <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Gradient Descent
pub use self::{descent::*, utils::*};

pub(crate) mod descent;

pub mod sgd;

pub trait Descent<T> {
    fn descent(&self, params: &[f64], grads: &[f64]) -> Vec<f64>;
}

pub trait LearningRate {
    fn gamma(&self) -> f64;
}

pub trait Momentum {
    fn mu(&self) -> f64;

    fn nestrov(&self) -> bool;
}

pub struct DescentParams {
    pub batch_size: usize,
    pub epochs: usize,
    pub gamma: f64, // learning rate
}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
