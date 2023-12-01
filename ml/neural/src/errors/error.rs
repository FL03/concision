/*
   Appellation: error <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};
use smart_default::SmartDefault;
use strum::{Display, EnumIs, EnumIter, EnumVariantNames};

#[derive(
    Clone,
    Debug,
    Deserialize,
    Display,
    EnumIs,
    EnumIter,
    EnumVariantNames,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
    SmartDefault,
)]
#[non_exhaustive]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum Errors {
    Async,
    Codec,
    Connection,
    Custom(String),
    Data,
    Dimension,
    #[default]
    Error,
    Execution,
    IO,
    Null,
    Parse,
    Process,
    Runtime,
    Syntax,
    Unknown,
}

pub enum NetworkError {}

pub enum ActivationError {}

pub enum LayerError {}

pub enum ModelError {}
