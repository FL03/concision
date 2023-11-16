/*
    Appellation: shapes <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};
use strum::{Display, EnumIs, EnumIter, EnumString, EnumVariantNames};

pub trait LayerFeatures {
    fn inputs(&self) -> usize;
    fn outputs(&self) -> usize;
}

impl LayerFeatures for ParameterShapes {
    fn inputs(&self) -> usize {
        match self {
            ParameterShapes::Layer { inputs, .. } => *inputs,
            ParameterShapes::Neuron { inputs } => *inputs,
        }
    }

    fn outputs(&self) -> usize {
        match self {
            ParameterShapes::Layer { outputs, .. } => *outputs,
            ParameterShapes::Neuron { .. } => 1,
        }
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    Deserialize,
    EnumIs,
    EnumIter,
    EnumVariantNames,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
)]
pub enum ParameterShapes {
    Layer { inputs: usize, outputs: usize },
    Neuron { inputs: usize },
}

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Deserialize,
    Display,
    EnumIs,
    EnumIter,
    EnumString,
    EnumVariantNames,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
)]
#[repr(usize)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum Parameters {
    Bias,
    #[default]
    Weights,
}

impl Parameters {
    pub fn bias() -> Self {
        Self::Bias
    }

    pub fn weights() -> Self {
        Self::Weights
    }
}
