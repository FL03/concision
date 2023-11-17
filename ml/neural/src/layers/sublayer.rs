/*
    Appellation: sublayers <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Layer;
use crate::func::activate::{Activate, LinearActivation};
use crate::ops::LayerNorm;
use crate::prelude::Forward;

use ndarray::prelude::{Array2, NdFloat};
use num::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Sublayer<T = f64, A = LinearActivation> where A: Activate<Array2<T>>, T: Float {
    layer: Layer<T, A>,
    norm: LayerNorm<T>,
}

impl<T, A> Sublayer<T, A>
where
    A: Activate<Array2<T>>,
    T: Float,
{
    pub fn new(layer: Layer<T, A>, norm: LayerNorm<T>) -> Self {
        Self { layer, norm }
    }
}

impl<T, A> Forward<Array2<T>> for Sublayer<T, A>
where
    A: Activate<Array2<T>>,
    T: FromPrimitive + NdFloat,
{
    type Output = Array2<T>;
    
    fn forward(&self, data: &Array2<T>) -> Self::Output {
        let norm = self.norm.forward(data);
        let layer = data + self.layer.forward(&norm);
        layer
    }
}
