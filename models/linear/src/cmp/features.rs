/*
   Appellation: features <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neural::prelude::Features;
use ndarray::prelude::{Dimension, Ix2};
use ndarray::IntoDimension;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct LinearShape {
    pub inputs: usize,
    pub outputs: usize,
}

impl LinearShape {
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
        T::from(self.inputs()).unwrap().recip().sqrt()
    }
}

impl std::fmt::Display for LinearShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.inputs, self.outputs)
    }
}

impl Features for LinearShape {
    fn inputs(&self) -> usize {
        self.inputs
    }

    fn outputs(&self) -> usize {
        self.outputs
    }
}

impl IntoDimension for LinearShape {
    type Dim = Ix2;

    fn into_dimension(self) -> Self::Dim {
        ndarray::Ix2(self.outputs, self.inputs)
    }
}

impl From<LinearShape> for Ix2 {
    fn from(features: LinearShape) -> Self {
        features.into_dimension()
    }
}

impl From<LinearShape> for ndarray::IxDyn {
    fn from(features: LinearShape) -> Self {
        ndarray::IxDyn(&[features.outputs, features.inputs])
    }
}

impl From<LinearShape> for [usize; 2] {
    fn from(features: LinearShape) -> Self {
        [features.outputs, features.inputs]
    }
}

impl From<[usize; 2]> for LinearShape {
    fn from(features: [usize; 2]) -> Self {
        Self {
            inputs: features[1],
            outputs: features[0],
        }
    }
}

impl From<LinearShape> for (usize, usize) {
    fn from(features: LinearShape) -> Self {
        (features.outputs, features.inputs)
    }
}

impl From<(usize, usize)> for LinearShape {
    fn from((inputs, outputs): (usize, usize)) -> Self {
        Self { inputs, outputs }
    }
}

impl From<usize> for LinearShape {
    fn from(inputs: usize) -> Self {
        Self { inputs, outputs: 1 }
    }
}
