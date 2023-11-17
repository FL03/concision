/*
   Appellation: features <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::IntoDimension;
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct Features {
    pub inputs: usize,
    pub outputs: usize,
}

impl Features {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        Self { inputs, outputs }
    }

    pub fn neuron(inputs: usize) -> Self {
        Self::new(inputs, 1)
    }

    pub fn uniform_scale<T: num::Float>(&self) -> T {
        (T::one() / T::from(self.inputs()).unwrap()).sqrt()
    }

    pub fn inputs(&self) -> usize {
        self.inputs
    }

    pub fn outputs(&self) -> usize {
        self.outputs
    }

    pub fn in_by_out(&self) -> (usize, usize) {
        (self.inputs, self.outputs)
    }

    pub fn out_by_in(&self) -> (usize, usize) {
        (self.outputs, self.inputs)
    }
}

impl std::fmt::Display for Features {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.inputs, self.outputs)
    }
}

impl IntoDimension for Features {
    type Dim = ndarray::Ix2;

    fn into_dimension(self) -> Self::Dim {
        ndarray::Ix2(self.outputs, self.inputs)
    }
}

impl From<Features> for ndarray::Ix2 {
    fn from(features: Features) -> Self {
        ndarray::Ix2(features.outputs, features.inputs)
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
    fn from((outputs, inputs): (usize, usize)) -> Self {
        Self { inputs, outputs }
    }
}

impl From<usize> for Features {
    fn from(inputs: usize) -> Self {
        Self { inputs, outputs: 1 }
    }
}
