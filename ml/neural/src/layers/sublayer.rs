/*
    Appellation: sublayers <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Layer;
use crate::ops::LayerNorm;
use crate::prop::Forward;

use ndarray::prelude::Array2;
use ndarray::ScalarOperand;
use num::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Sublayer<T: Float = f64> {
    layer: Layer<T>,
    norm: LayerNorm<T>,
}

impl<T: Float> Sublayer<T> {
    pub fn new(layer: Layer<T>, norm: LayerNorm<T>) -> Self {
        Self { layer, norm }
    }

    pub fn forward(&self, data: &Array2<T>) -> Array2<T>
    where
        T: FromPrimitive + ScalarOperand,
    {
        let norm = self.norm.forward(data);
        let layer = data + self.layer.forward(&norm);
        layer
    }
}
