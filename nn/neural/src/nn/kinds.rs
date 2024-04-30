/*
    Appellation: kinds <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};
use strum::{Display, EnumIs, EnumIter, EnumString, VariantNames};

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
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
    VariantNames,
)]
#[repr(usize)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum NetworkKind {
    #[serde(alias = "convolution", alias = "conv", alias = "cnn")]
    Convolution = 0,
    #[default]
    #[serde(alias = "feed_forward", alias = "ffn")]
    FeedForward = 1,
    #[serde(alias = "graph", alias = "gnn")]
    Graph = 2,
    #[serde(alias = "recurrent", alias = "rnn")]
    Recurrent = 3,
}

impl NetworkKind {
    pub fn cnn() -> Self {
        Self::Convolution
    }

    pub fn ffn() -> Self {
        Self::FeedForward
    }

    pub fn gnn() -> Self {
        Self::Graph
    }

    pub fn rnn() -> Self {
        Self::Recurrent
    }
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
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
    VariantNames,
)]
#[repr(usize)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum Learning {
    Reinforcement = 0,
    #[default]
    Supervised = 1,
    Unsupervised = 2,
}

impl Learning {
    pub fn reinforcement() -> Self {
        Self::Reinforcement
    }

    pub fn supervised() -> Self {
        Self::Supervised
    }

    pub fn unsupervised() -> Self {
        Self::Unsupervised
    }
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
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
    VariantNames,
)]
#[repr(usize)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum NetworkType {
    #[serde(alias = "autoencoder", alias = "ae")]
    Autoencoder = 0,
    #[default]
    #[serde(alias = "classifier", alias = "clf")]
    Classifier = 1,
    #[serde(alias = "regressor", alias = "reg")]
    Regressor = 2,
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
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
    VariantNames,
)]
#[repr(usize)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum NetworkStyle {
    #[default]
    Deep,
    Shallow,
}
