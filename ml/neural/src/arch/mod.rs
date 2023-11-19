/*
    Appellation: arch <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Architecture
//!
//! This module describes the architecture of various components of the neural network.
pub use self::{architecture::*, deep::*, shallow::*, utils::*};

pub(crate) mod architecture;
pub(crate) mod deep;
pub(crate) mod shallow;

pub trait Arch {}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::prelude::linarr;
    use crate::models::stack::Stack;
    use crate::prelude::{Forward, Layer, LayerShape, Sigmoid};
    use ndarray::prelude::Ix2;

    #[test]
    fn test_arch() {
        assert!(true);
    }

    #[test]
    fn test_deep_network() {
        let samples = 20;
        let (inputs, outputs) = (5, 3);
        let shapes = [(outputs, 4), (4, 4), (4, inputs)];

        // sample data
        let x = linarr::<f64, Ix2>((samples, inputs)).unwrap();
        let _y = linarr::<f64, Ix2>((samples, outputs)).unwrap();

        // layers
        let hidden = Stack::<f64, Sigmoid>::new()
            .build_layers(shapes)
            .init_layers(true);
        let input = Layer::<f64>::from(LayerShape::new(inputs, outputs)).init(false);
        let output = Layer::<f64>::from(LayerShape::new(inputs, outputs)).init(false);

        let network = DeepNetwork::new(input, hidden, output);
        assert!(network.validate_dims());

        let pred = network.forward(&x);
        assert_eq!(&pred.dim(), &(samples, outputs));
    }

    #[test]
    fn test_shallow_network() {
        let samples = 20;
        let (inputs, outputs) = (5, 3);

        // sample data
        let x = linarr::<f64, Ix2>((samples, inputs)).unwrap();
        let _y = linarr::<f64, Ix2>((samples, outputs)).unwrap();

        // layers
        let input = Layer::<f64>::from(LayerShape::new(inputs, outputs)).init(false);
        let output = Layer::<f64>::from(LayerShape::new(outputs, outputs)).init(false);

        let network = ShallowNetwork::new(input, output);
        assert!(network.validate_dims());

        let pred = network.forward(&x);
        assert_eq!(&pred.dim(), &(samples, outputs));
    }
}
