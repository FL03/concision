/*
    Appellation: config <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::cmp::LayerShape;

pub struct LinearConfig {
    pub biased: bool,
    pub name: String,
    pub shape: LayerShape,
}
