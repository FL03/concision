/*
    Appellation: config <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct SSMConfig {
    pub decode: bool,
    pub features: usize,
}

impl SSMConfig {
    pub fn new(decode: bool, features: usize) -> Self {
        Self { decode, features }
    }

    pub fn decode(&self) -> bool {
        self.decode
    }

    pub fn features(&self) -> usize {
        self.features
    }
}
