/*
   Appellation: features <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Dimension, Ix2};
use ndarray::IntoDimension;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Features {
    pub inputs: usize,
    pub outputs: usize,
}

impl Features {
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

    pub fn inputs(&self) -> usize {
        self.inputs
    }

    pub fn outputs(&self) -> usize {
        self.outputs
    }

    pub fn uniform_scale<T: num::Float>(&self) -> T {
        T::from(self.inputs()).unwrap().recip().sqrt()
    }
}

impl core::fmt::Display for Features {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "({}, {})", self.inputs, self.outputs)
    }
}

impl IntoDimension for Features {
    type Dim = Ix2;

    fn into_dimension(self) -> Self::Dim {
        ndarray::Ix2(self.outputs, self.inputs)
    }
}

impl From<Features> for Ix2 {
    fn from(features: Features) -> Self {
        features.into_dimension()
    }
}

impl From<Features> for ndarray::IxDyn {
    fn from(features: Features) -> Self {
        ndarray::IxDyn(&[features.outputs, features.inputs])
    }
}

impl From<Features> for [usize; 2] {
    fn from(features: Features) -> Self {
        [features.outputs, features.inputs]
    }
}

impl From<[usize; 2]> for Features {
    fn from(features: [usize; 2]) -> Self {
        Self {
            inputs: features[1],
            outputs: features[0],
        }
    }
}

impl From<Features> for (usize, usize) {
    fn from(features: Features) -> Self {
        (features.outputs, features.inputs)
    }
}

impl From<(usize, usize)> for Features {
    fn from((inputs, outputs): (usize, usize)) -> Self {
        Self { inputs, outputs }
    }
}

impl From<usize> for Features {
    fn from(inputs: usize) -> Self {
        Self { inputs, outputs: 1 }
    }
}