/*
    Appellation: grad <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Gradient Descent
pub use self::{descent::*, gradient::*, utils::*};

pub(crate) mod descent;
pub(crate) mod gradient;

pub mod sgd;

use num::Float;

pub trait Descent<T: Float = f64> {
    type Params;

    fn descent(&self) -> Vec<f64>;
}

pub trait LearningRate<T = f64>
where
    T: Float,
{
    fn gamma(&self) -> T;
}

pub trait Momentum<T = f64>
where
    T: Float,
{
    fn mu(&self) -> T; // Momentum Rate
}

pub trait Nesterov: Momentum {
    fn nestrov(&self) -> bool;
}

pub trait Decay<T = f64>
where
    T: Float,
{
    fn lambda(&self) -> T; // Decay Rate
}

pub trait Dampener<T = f64>
where
    T: Float,
{
    fn tau(&self) -> T; // Momentum Damper
}

pub struct DescentParams {
    pub batch_size: usize,
    pub epochs: usize,
    pub gamma: f64, // learning rate
}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
