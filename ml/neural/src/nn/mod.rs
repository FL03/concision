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

pub trait NeuralNet: Trainable {
    fn depth(&self) -> usize {
        self.layers().len()
    }

    fn layers(&self) -> &[Layer];
}

pub(crate) mod utils {}
