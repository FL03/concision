/*
    Appellation: config <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::traits::Config;

pub struct ModelConfig {
    pub name: String,
    _children: Vec<Box<dyn Config>>,
}

impl Config for ModelConfig {}

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize,))]
pub struct ConfigBase {
    pub id: usize,
    pub name: &'static str,
}
