/*
    Appellation: grad <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Gradient Descent
pub use self::{descent::*, utils::*};

pub(crate) mod descent;

pub mod sgd;

use num::Float;

pub trait Descent<T: Float = f64> {
    type Params;

    fn descent(&self, ) -> Vec<f64>;
}

pub trait LearningRate {
    fn gamma(&self) -> f64;
}

pub trait Momentum {

    fn mu(&self) -> f64; // Momentum Rate

    fn nestrov(&self) -> bool;

    fn tau(&self) -> f64; // Momentum Damper
}

pub trait Nesterov {
    fn nestrov(&self) -> bool;
}

pub trait Decay {
    fn lambda(&self) -> f64; // Decay Rate
}

pub trait Dampener {
    fn tau(&self) -> f64; // Momentum Damper
}

pub struct DescentParams {
    pub batch_size: usize,
    pub epochs: usize,
    pub gamma: f64, // learning rate
}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
