/*
    Appellation: cmp <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Layers
pub use self::{features::*, kinds::*};

pub(crate) mod features;
pub(crate) mod kinds;

use ndarray::prelude::{Dimension, Ix2};
use ndarray::IntoDimension;
use num::Float;

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

    fn input_scale<T: Float>(&self) -> T {
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

impl FeaturesExt<Ix2> for (usize, usize) {
    fn new(inputs: usize, outputs: usize) -> Self {
        (outputs, inputs)
    }
}

pub trait FromFeatures<Sh: Features> {
    fn from_features(features: Sh) -> Self;
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

#[cfg(test)]
mod tests {}
