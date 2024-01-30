/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Linear Model
//!
pub use self::{config::*, layer::*, module::*};

pub(crate) mod config;
pub(crate) mod layer;
pub(crate) mod module;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmp::neurons::Node;
    use crate::cmp::params::LayerShape;
    use crate::core::prelude::linarr;
    use crate::neural::prelude::{Forward, Softmax};
    use ndarray::prelude::Ix2;

    #[test]
    fn test_linear() {
        let (samples, inputs, outputs) = (20, 5, 3);
        let features = LayerShape::new(inputs, outputs);

        let args = linarr::<f64, Ix2>((samples, inputs)).unwrap();

        let layer = Linear::<f64, Softmax>::from(features).init(true);

        let pred = layer.forward(&args);

        assert_eq!(pred.dim(), (samples, outputs));

        let nodes = (0..outputs)
            .map(|_| Node::<f64>::new(inputs).init(true))
            .collect::<Vec<_>>();
        let layer = Linear::<f64, Softmax>::from_iter(nodes);
        assert_eq!(layer.features(), &features);
    }

    #[test]
    fn test_linear_iter() {
        let (_samples, inputs, outputs) = (20, 5, 3);
        let features = LayerShape::new(inputs, outputs);

        let layer = Linear::<f64, Softmax>::from(features).init(true);

        for node in layer.into_iter() {
            assert!(node.is_biased());
            assert_eq!(node.features(), inputs);
            assert_eq!(node.bias().as_ref().unwrap().dim(), ());
        }
    }
}
