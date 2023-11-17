/*
    Appellation: kinds <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};
use strum::{Display, EnumIs, EnumIter, EnumString, EnumVariantNames};

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
pub enum LayerType {
    #[default]
    Input = 0,
    Hidden(usize),
    Output,
}

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct Position {
    pub idx: usize,
    pub kind: LayerType,
}

impl Position {
    pub fn new(idx: usize, kind: LayerType) -> Self {
        Self { idx, kind }
    }

    pub fn input() -> Self {
        Self::new(0, LayerType::Input)
    }

    pub fn hidden(idx: usize) -> Self {
        Self::new(idx, LayerType::Hidden(idx))
    }

    pub fn output(idx: usize) -> Self {
        Self::new(idx, LayerType::Output)
    }

    pub fn is_input(&self) -> bool {
        self.kind().is_input()
    }

    pub fn is_hidden(&self) -> bool {
        self.kind().is_hidden()
    }

    pub fn is_output(&self) -> bool {
        self.kind().is_output()
    }

    pub fn kind(&self) -> &LayerType {
        &self.kind
    }

    pub fn position(&self) -> usize {
        self.idx
    }
}

impl AsRef<usize> for Position {
    fn as_ref(&self) -> &usize {
        &self.idx
    }
}

impl AsRef<LayerType> for Position {
    fn as_ref(&self) -> &LayerType {
        &self.kind
    }
}

impl AsMut<usize> for Position {
    fn as_mut(&mut self) -> &mut usize {
        &mut self.idx
    }
}

impl AsMut<LayerType> for Position {
    fn as_mut(&mut self) -> &mut LayerType {
        &mut self.kind
    }
}

impl From<Position> for usize {
    fn from(pos: Position) -> Self {
        pos.idx
    }
}

impl From<Position> for LayerType {
    fn from(pos: Position) -> Self {
        pos.kind
    }
}
