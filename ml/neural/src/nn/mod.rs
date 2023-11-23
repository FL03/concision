/*
    Appellation: nn <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Neural Network
pub use self::{position::*, sequential::*, utils::*};

pub(crate) mod position;
pub(crate) mod sequential;

use crate::layers::Layer;
use crate::Trainable;
use num::Float;

pub trait NeuralNet<T = f64>: Trainable<T>
where
    T: Float,
{
    fn depth(&self) -> usize {
        self.layers().len()
    }

    fn layers(&self) -> &[Layer<T>];

    fn input_layer(&self) -> &Layer<T> {
        &self.layers()[0]
    }

    fn output_layer(&self) -> &Layer<T> {
        &self.layers()[self.depth() - 1]
    }
}

pub trait DeepNeuralNet<T = f64>: NeuralNet<T>
where
    T: Float,
{
    fn hidden_layers(&self) -> &[Layer<T>];
}

pub(crate) mod utils {}
