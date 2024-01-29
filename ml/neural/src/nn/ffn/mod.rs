/*
    Appellation: ffn <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Feed Forward Neural Network
//!
pub use self::{mlp::*, model::*};

pub(crate) mod mlp;
pub(crate) mod model;

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
