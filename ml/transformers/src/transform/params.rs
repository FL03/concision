/*
   Appellation: params <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct TransformerParams {
    pub batch: usize,
    pub heads: usize,
    pub layers: usize,
    pub model: usize,
    pub samples: usize,
}