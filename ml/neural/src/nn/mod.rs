/*
    Appellation: nn <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Neural Network
pub use self::{position::*, sequential::*, utils::*};

pub(crate) mod position;
pub(crate) mod sequential;

pub mod ffn;
pub mod gnn;

use crate::core::BoxResult;
use crate::layers::Layer;
use crate::Trainable;
use ndarray::prelude::{Array, Dimension, Ix2};
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

pub trait Compile<T = f64> {}

pub trait Train<T = f64> {}

pub trait Predict<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    type Output;

    fn predict(&self, input: &Array<T, D>) -> BoxResult<Self::Output>;

    fn predict_batch(&self, input: &[Array<T, D>]) -> BoxResult<Vec<Self::Output>> {
        let res = input.iter().map(|x| self.predict(x).expect("")).collect();
        Ok(res)
    }
}

pub(crate) mod utils {}
