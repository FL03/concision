/*
    Appellation: config <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct ModelConfig {
    pub layers: usize,
}

impl ModelConfig {
    pub fn new(layers: usize) -> Self {
        Self { layers }
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    pub fn n_hidden(&self) -> usize {
        self.layers - 2
    }
}
