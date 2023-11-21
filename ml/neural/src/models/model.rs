/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::ModelConfig;
use crate::layers::LayerDyn;
use num::Float;

pub struct Model<T = f64> where T: Float {
    config: ModelConfig,
    layers: Vec<LayerDyn<T>>,
}

impl<T> Model<T> where T: Float {
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

    pub fn layers(&self) -> &[LayerDyn<T>] {
        &self.layers
    }

    pub fn layers_mut(&mut self) -> &mut [LayerDyn<T>] {
        &mut self.layers
    }
}

impl<T> IntoIterator for Model<T> where T: Float {
    type Item = LayerDyn<T>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.layers.into_iter()
    }
}


