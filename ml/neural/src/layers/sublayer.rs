/*
    Appellation: sublayers <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Layer;
use crate::ops::LayerNorm;
use crate::prelude::Forward;

use ndarray::prelude::{Array2, NdFloat};
use num::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Sublayer<T: Float = f64> {
    layer: Layer<T>,
    norm: LayerNorm<T>,
}

impl<T> Sublayer<T>
where
    T: Float,
{
    pub fn new(layer: Layer<T>, norm: LayerNorm<T>) -> Self {
        Self { layer, norm }
    }
}

impl<T> Sublayer<T>
where
    T: FromPrimitive + NdFloat,
{
    pub fn forward(&self, data: &Array2<T>) -> Array2<T> {
        let norm = self.norm.forward(data);
        let layer = data + self.layer.forward(&norm);
        layer
    }
}
