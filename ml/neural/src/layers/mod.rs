/*
    Appellation: layers <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Layers
pub use self::{cmp::*, layer::*, params::*, stack::*, utils::*};

pub(crate) mod cmp;
pub(crate) mod layer;
pub(crate) mod params;
pub(crate) mod stack;

pub mod exp;

use crate::func::activate::{Activate, ActivateDyn};
use crate::prelude::Node;
use ndarray::prelude::Ix2;
// use ndarray::IntoDimension;
use num::Float;

pub type LayerDyn<T = f64, D = Ix2> = Layer<T, ActivateDyn<T, D>>;

pub trait L<T, A>: IntoIterator<Item = Node<T>>
where
    A: Activate<T>,
    T: Float,
{
    fn features(&self) -> LayerShape;
    fn name(&self) -> &str;
    fn params(&self) -> &LayerParams<T>;
    fn position(&self) -> LayerPosition;

    fn is_biased(&self) -> bool;
}

// pub trait LayerExt<T = f64>: L<T>
// where
//     T: Float,
// {
//     type Rho: Activate<T, Ix2>;
// }

pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::prelude::linarr;
    use crate::func::activate::Softmax;
    use crate::prelude::{Biased, Forward, Node, Parameterized};
    use ndarray::prelude::Ix2;

    #[test]
    fn test_layer() {
        let (samples, inputs, outputs) = (20, 5, 3);
        let features = LayerShape::new(inputs, outputs);

        let args = linarr::<f64, Ix2>((samples, inputs)).unwrap();

        let layer = Layer::<f64, Softmax>::from(features).init(true);

        let pred = layer.forward(&args);

        assert_eq!(pred.dim(), (samples, outputs));

        let nodes = (0..outputs)
            .map(|_| Node::<f64>::new(inputs).init(true))
            .collect::<Vec<_>>();
        let layer = Layer::<f64, Softmax>::from_iter(nodes);
        assert_eq!(layer.features(), &features);
    }

    #[test]
    fn test_layer_iter() {
        let (_samples, inputs, outputs) = (20, 5, 3);
        let features = LayerShape::new(inputs, outputs);

        let layer = Layer::<f64, Softmax>::from(features).init(true);

        for node in layer.into_iter() {
            assert_eq!(node.features(), inputs);
            assert_eq!(node.bias().dim(), ());
        }
    }
}
