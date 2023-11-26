/*
   Appellation: features <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Dimension, Ix2};
use ndarray::IntoDimension;
use serde::{Deserialize, Serialize};

pub trait Features {
    fn inputs(&self) -> usize;

    fn network(&self) -> usize {
        self.inputs() * self.outputs()
    }

    fn outputs(&self) -> usize;

    fn in_by_out(&self) -> (usize, usize) {
        (self.inputs(), self.outputs())
    }

    fn out_by_in(&self) -> (usize, usize) {
        (self.outputs(), self.inputs())
    }

    fn input_scale<T: num::Float>(&self) -> T {
        (T::one() / T::from(self.inputs()).unwrap()).sqrt()
    }
}

impl Features for usize {
    fn inputs(&self) -> usize {
        *self
    }

    fn outputs(&self) -> usize {
        1
    }
}

impl Features for (usize, usize) {
    fn inputs(&self) -> usize {
        self.1
    }

    fn outputs(&self) -> usize {
        self.0
    }
}

impl Features for [usize; 2] {
    fn inputs(&self) -> usize {
        self[1]
    }

    fn outputs(&self) -> usize {
        self[0]
    }
}

pub trait FeaturesExt<D = Ix2>: Features + IntoDimension<Dim = D>
where
    D: Dimension,
{
    fn new(inputs: usize, outputs: usize) -> Self;

    fn single(inputs: usize) -> Self
    where
        Self: Sized,
    {
        Self::new(inputs, 1)
    }
}

// impl<T> FeaturesExt for T
// where
//     T: Features + IntoDimension<Dim = Ix2>,
// {
//     fn new(inputs: usize, outputs: usize) -> Self {
//         Self::from_dimension(ndarray::Ix2(outputs, inputs))
//     }
// }

pub trait FromFeatures<Sh: Features> {
    fn from_features(features: LayerShape) -> Self;
}

pub trait IntoFeatures {
    fn into_features(self) -> LayerShape;
}

impl<S> IntoFeatures for S
where
    S: Into<LayerShape>,
{
    fn into_features(self) -> LayerShape {
        self.into()
    }
}

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
        ndarray::Ix2(features.outputs, features.inputs)
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
