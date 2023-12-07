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
pub enum LayerKind {
    #[default]
    Input = 0,
    Hidden,
    Output,
}

impl LayerKind {
    pub fn input() -> Self {
        Self::Input
    }

    pub fn hidden() -> Self {
        Self::Hidden
    }

    pub fn output() -> Self {
        Self::Output
    }

    pub fn create_kind(idx: usize, layers: usize) -> Self {
        if idx == 0 {
            Self::Input
        } else if idx == layers - 1 {
            Self::Output
        } else {
            Self::Hidden
        }
    }
}

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct LayerPosition {
    pub idx: usize,
    pub kind: LayerKind,
}

impl LayerPosition {
    pub fn new(idx: usize, kind: LayerKind) -> Self {
        Self { idx, kind }
    }

    pub fn input() -> Self {
        Self::new(0, LayerKind::Input)
    }

    pub fn hidden(idx: usize) -> Self {
        Self::new(idx, LayerKind::Hidden)
    }

    pub fn output(idx: usize) -> Self {
        Self::new(idx, LayerKind::Output)
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

    pub fn index(&self) -> usize {
        self.idx
    }

    pub fn kind(&self) -> &LayerKind {
        &self.kind
    }
}

impl AsRef<usize> for LayerPosition {
    fn as_ref(&self) -> &usize {
        &self.idx
    }
}

impl AsRef<LayerKind> for LayerPosition {
    fn as_ref(&self) -> &LayerKind {
        &self.kind
    }
}

impl AsMut<usize> for LayerPosition {
    fn as_mut(&mut self) -> &mut usize {
        &mut self.idx
    }
}

impl AsMut<LayerKind> for LayerPosition {
    fn as_mut(&mut self) -> &mut LayerKind {
        &mut self.kind
    }
}

impl From<LayerPosition> for usize {
    fn from(pos: LayerPosition) -> Self {
        pos.idx
    }
}

impl From<LayerPosition> for LayerKind {
    fn from(pos: LayerPosition) -> Self {
        pos.kind
    }
}
