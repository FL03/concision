/*
    Appellation: nn <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Neural Network
pub use self::{network::*, utils::*};

pub mod loss;

pub(crate) mod network;

use crate::layers::Layer;
use crate::Trainable;
use num::Float;

pub trait NeuralNet<T: Float = f64>: Trainable<T> {
    fn depth(&self) -> usize {
        self.layers().len()
    }

    fn layers(&self) -> &[Layer<T>];
}

pub(crate) mod utils {}
