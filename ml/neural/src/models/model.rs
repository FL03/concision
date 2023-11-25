/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::ModelConfig;
use crate::prelude::{Forward, ForwardDyn};
use ndarray::prelude::Array2;
use num::Float;

pub struct Model<T = f64>
where
    T: Float,
{
    config: ModelConfig,
    layers: Vec<ForwardDyn<T>>,
}

impl<T> Model<T>
where
    T: Float,
{
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            layers: Vec::new(),
        }
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut ModelConfig {
        &mut self.config
    }

    pub fn layers(&self) -> &[ForwardDyn<T>] {
        &self.layers
    }

    pub fn layers_mut(&mut self) -> &mut [ForwardDyn<T>] {
        &mut self.layers
    }

    pub fn add_layer(&mut self, layer: ForwardDyn<T>) {
        self.layers.push(layer);
    }
}

impl<T> IntoIterator for Model<T>
where
    T: Float,
{
    type Item = ForwardDyn<T>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.layers.into_iter()
    }
}

impl<T> Forward<Array2<T>> for Model<T>
where
    T: Float,
{
    type Output = Array2<T>;

    fn forward(&self, input: &Array2<T>) -> Array2<T> {
        let mut iter = self.layers().into_iter();

        let mut output = iter.next().unwrap().forward(input);
        for layer in iter {
            output = layer.forward(&output);
        }
        output
    }
}
