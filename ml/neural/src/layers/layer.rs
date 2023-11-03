/*
    Appellation: layer <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Features, LayerType};
use crate::bias::Bias;
use crate::prop::Forward;

use ndarray::prelude::Array2;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Layer<T: Float = f64> {
    bias: Bias<T>,
    features: Features,
    layer: LayerType,
    weights: Array2<T>,
}

impl<T: Float> Layer<T> {
    pub fn new(inputs: usize, outputs: usize, layer: LayerType) -> Self
    where
        T: SampleUniform,
    {
        let features = Features::new(inputs, outputs);
        let weights = Array2::ones((features.inputs(), features.outputs()));

        Self {
            bias: Bias::default(),
            features,
            layer,
            weights,
        }
    }

    pub fn bias(&self) -> &Bias<T> {
        &self.bias
    }

    pub fn layer(&self) -> &LayerType {
        &self.layer
    }

    pub fn features(&self) -> Features {
        self.features
    }

    pub fn set_layer(&mut self, layer: LayerType) {
        self.layer = layer;
    }

    pub fn weights(&self) -> &Array2<T> {
        &self.weights
    }
}

impl<T> Layer<T> where T: Float + SampleUniform {
        pub fn biased(inputs: usize, outputs: usize, layer: LayerType) -> Self
    where
        T: SampleUniform,
    {
        let features = Features::new(inputs, outputs);
        let weights = Array2::ones((features.inputs(), features.outputs()));

        Self {
            bias: Bias::biased(outputs),
            features,
            layer,
            weights,
        }
    }
}

impl<T: Float + 'static> Forward<Array2<T>> for Layer<T> {
    type Output = Array2<T>;

    fn forward(&self, data: &Array2<T>) -> Self::Output {
        data.dot(&self.weights().t()) + self.bias()
    }
}
