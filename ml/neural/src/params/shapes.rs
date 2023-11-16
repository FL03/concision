/*
    Appellation: shapes <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::{Dimension, IntoDimension};
use serde::{Deserialize, Serialize};
use strum::{EnumIs, EnumIter, EnumVariantNames};

pub trait LayerFeatures {
    fn inputs(&self) -> usize;
    fn outputs(&self) -> usize;
}

pub trait NeuronFeatures {
    fn inputs(&self) -> usize;
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
