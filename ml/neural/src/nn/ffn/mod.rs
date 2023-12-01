/*
    Appellation: ffn <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Feed Forward Neural Network
//!
pub use self::{mlp::*, model::*, utils::*};

pub(crate) mod mlp;
pub(crate) mod model;

use ndarray::prelude::{Array, Array2, Dimension, Ix2};
use num::Float;

pub trait Optimizer<T = f64>
where
    T: Float,
{
    fn step(&mut self, grad: &Array2<T>) -> Array2<T>;
}

pub trait FeedForward<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    type Opt;

    fn backward(&mut self, args: &Array2<T>, targets: &Array<T, D>, opt: &Self::Opt) -> Array2<T>;

    fn forward(&self, args: &Array2<T>) -> Array<T, D>;
}

pub(crate) mod utils {}

#[cfg(tets)]
mod tests {
    use super::*;
    use crate::core::prelude::linarr;
    use crate::func::activate::{ReLU, Softmax};
    use crate::prelude::{Forward, Layer, LayerShape, Stack};
    use ndarray::prelude::Ix2;

    #[test]
    fn test_mlp() {
        let samples = 20;
        let (inputs, outputs) = (5, 3);
        let shapes = [(outputs, 4), (4, 4), (4, inputs)];

        // sample data
        let x = linarr::<f64, Ix2>((samples, inputs)).unwrap();
        let _y = linarr::<f64, Ix2>((samples, outputs)).unwrap();

        // layers
        let hidden = Stack::<f64, ReLU>::new()
            .build_layers(shapes)
            .init_layers(true);
        let input = Layer::<f64>::from(LayerShape::new(inputs, outputs)).init(false);
        let output = Layer::<f64>::from(LayerShape::new(inputs, outputs)).init(false);

        let network = MLP::new(input, hidden, output);
        assert!(network.validate_dims());

        let pred = network.forward(&x);
        assert_eq!(&pred.dim(), &(samples, outputs));
    }
}
