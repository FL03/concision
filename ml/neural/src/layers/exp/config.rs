/*
    Appellation: config <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::layers::{LayerKind, LayerShape};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct LayerConfig {
    pub features: LayerShape,
    kind: LayerKind,
    name: String,
}
