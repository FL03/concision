/*
   Appellation: params <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neural::prelude::Features;
use crate::prelude::{MODEL, NETWORK};
use ndarray::prelude::Ix2;
use ndarray::IntoDimension;
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

    pub fn model(&self) -> usize {
        self.model
    }

    pub fn network(&self) -> usize {
        self.network
    }

    pub fn features(&self) -> usize {
        self.network / self.model
    }
}

impl Default for FFNParams {
    fn default() -> Self {
        Self::new(MODEL, NETWORK)
    }
}

impl Features for FFNParams {
    fn inputs(&self) -> usize {
        self.model
    }

    fn outputs(&self) -> usize {
        self.network
    }
}

impl IntoDimension for FFNParams {
    type Dim = Ix2;

    fn into_dimension(self) -> Ix2 {
        (self.model(), self.network()).into_dimension()
    }
}
