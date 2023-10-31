/*
   Appellation: params <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct FFNParams {
    pub model: usize,
    pub network: usize,
}

impl FFNParams {
    pub fn new(model: usize, network: usize) -> Self {
        Self { model, network }
    }

    pub fn model_size(&self) -> usize {
        self.model
    }

    pub fn network_size(&self) -> usize {
        self.network
    }
}

impl Default for FFNParams {
    fn default() -> Self {
        Self {
            model: crate::MODEL_SIZE,
            network: crate::NETWORK_SIZE,
        }
    }
}
