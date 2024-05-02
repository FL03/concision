/*
    Appellation: neurons <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # neurons
pub use self::{node::*, perceptron::*};

pub(crate) mod node;
pub(crate) mod perceptron;

pub trait ArtificialNeuron {
    type Rho: for<'a> Fn(&'a Self::Output) -> Self::Output;
    type Output;

    fn activate(&self, x: &Self::Output) -> Self::Output {
        (self.rho())(x)
    }

    fn rho(&self) -> &Self::Rho;
}

#[cfg(test)]
mod tests {}
