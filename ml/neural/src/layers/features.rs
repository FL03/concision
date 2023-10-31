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

    pub fn inputs(&self) -> usize {
        self.inputs
    }

    pub fn outputs(&self) -> usize {
        self.outputs
    }
}

impl std::fmt::Display for Features {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.inputs, self.outputs)
    }
}

impl IntoDimension for Features {
    type Dim = ndarray::IxDyn;

    fn into_dimension(self) -> Self::Dim {
        ndarray::IxDyn(&[self.inputs, self.outputs])
    }
}

impl From<Features> for ndarray::Ix2 {
    fn from(features: Features) -> Self {
        ndarray::Ix2(features.inputs, features.outputs)
    }
}

impl From<Features> for ndarray::IxDyn {
    fn from(features: Features) -> Self {
        ndarray::IxDyn(&[features.inputs, features.outputs])
    }
}

impl From<Features> for [usize; 2] {
    fn from(features: Features) -> Self {
        [features.inputs, features.outputs]
    }
}

impl From<Features> for (usize, usize) {
    fn from(features: Features) -> Self {
        (features.inputs, features.outputs)
    }
}
