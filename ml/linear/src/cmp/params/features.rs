/*
   Appellation: features <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neural::prelude::Features;
use ndarray::prelude::{Dimension, Ix2};
use ndarray::IntoDimension;
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct LayerShape {
    pub inputs: usize,
    pub outputs: usize,
}

impl LayerShape {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        Self { inputs, outputs }
    }

    pub fn from_dimension(shape: impl IntoDimension<Dim = Ix2>) -> Self {
        let dim = shape.into_dimension();
        let (outputs, inputs) = dim.into_pattern();
        Self::new(inputs, outputs)
    }

    pub fn neuron(inputs: usize) -> Self {
        Self::new(inputs, 1)
    }

    pub fn uniform_scale<T: num::Float>(&self) -> T {
        (T::one() / T::from(self.inputs()).unwrap()).sqrt()
    }
}

impl std::fmt::Display for LayerShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.inputs, self.outputs)
    }
}

impl Features for LayerShape {
    fn inputs(&self) -> usize {
        self.inputs
    }

    fn outputs(&self) -> usize {
        self.outputs
    }
}

impl IntoDimension for LayerShape {
    type Dim = Ix2;

    fn into_dimension(self) -> Self::Dim {
        ndarray::Ix2(self.outputs, self.inputs)
    }
}

impl From<LayerShape> for Ix2 {
    fn from(features: LayerShape) -> Self {
        features.into_dimension()
    }
}

impl From<LayerShape> for ndarray::IxDyn {
    fn from(features: LayerShape) -> Self {
        ndarray::IxDyn(&[features.outputs, features.inputs])
    }
}

impl From<LayerShape> for [usize; 2] {
    fn from(features: LayerShape) -> Self {
        [features.outputs, features.inputs]
    }
}

impl From<[usize; 2]> for LayerShape {
    fn from(features: [usize; 2]) -> Self {
        Self {
            inputs: features[1],
            outputs: features[0],
        }
    }
}

impl From<LayerShape> for (usize, usize) {
    fn from(features: LayerShape) -> Self {
        (features.outputs, features.inputs)
    }
}

impl From<(usize, usize)> for LayerShape {
    fn from((inputs, outputs): (usize, usize)) -> Self {
        Self { inputs, outputs }
    }
}

impl From<usize> for LayerShape {
    fn from(inputs: usize) -> Self {
        Self { inputs, outputs: 1 }
    }
}
