/*
   Appellation: error <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};
use smart_default::SmartDefault;
use strum::{Display, EnumCount, EnumIs, EnumIter, VariantNames};

#[derive(
    Clone,
    Debug,
    Deserialize,
    Display,
    EnumCount,
    EnumIs,
    EnumIter,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
    SmartDefault,
    VariantNames,
)]
#[non_exhaustive]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum MlError {
    Compute(ComputeError),
    Data(String),
    Dimension(String),
    #[default]
    Error(String),
    Network(NetworkError),
}

impl std::error::Error for MlError {}

impl From<Box<dyn std::error::Error>> for MlError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        Self::Error(err.to_string())
    }
}

impl From<String> for MlError {
    fn from(err: String) -> Self {
        Self::Error(err)
    }
}

impl From<&str> for MlError {
    fn from(err: &str) -> Self {
        Self::Error(err.to_string())
    }
}

impl From<NetworkError> for MlError {
    fn from(err: NetworkError) -> Self {
        Self::Network(err)
    }
}

#[derive(
    Clone,
    Debug,
    Deserialize,
    Display,
    EnumCount,
    EnumIs,
    EnumIter,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
    SmartDefault,
    VariantNames,
)]
#[non_exhaustive]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum PredictError {
    Activation(String),
    Arithmetic(String),
    Layer(String),
    Format(String),
    #[default]
    Other(String),
}

impl std::error::Error for PredictError {}

impl From<Box<dyn std::error::Error>> for PredictError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        Self::Other(err.to_string())
    }
}

impl From<String> for PredictError {
    fn from(err: String) -> Self {
        Self::Other(err)
    }
}

impl From<&str> for PredictError {
    fn from(err: &str) -> Self {
        Self::Other(err.to_string())
    }
}

impl From<anyhow::Error> for PredictError {
    fn from(err: anyhow::Error) -> Self {
        Self::Other(err.to_string())
    }
}

impl From<ndarray::ShapeError> for PredictError {
    fn from(err: ndarray::ShapeError) -> Self {
        Self::Format(err.to_string())
    }
}

impl From<ndarray_linalg::error::LinalgError> for PredictError {
    fn from(err: ndarray_linalg::error::LinalgError) -> Self {
        Self::Arithmetic(err.to_string())
    }
}

#[derive(
    Clone,
    Debug,
    Deserialize,
    Display,
    EnumCount,
    EnumIs,
    EnumIter,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
    SmartDefault,
    VariantNames,
)]
#[non_exhaustive]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum ComputeError {
    Arithmetic(String),
    #[default]
    Process(String),
    ShapeError(String),
}

#[derive(
    Clone,
    Debug,
    Deserialize,
    Display,
    EnumCount,
    EnumIs,
    EnumIter,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
    SmartDefault,
    VariantNames,
)]
#[non_exhaustive]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum NetworkError {
    Layer(String),
    #[default]
    Network(String),
}

pub enum ActivationError {}

pub enum LayerError {}

pub enum ModelError {}
