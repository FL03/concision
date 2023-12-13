/*
    Appellation: config <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::layers::{LayerKind, LayerShape};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct LayerConfig {
    biased: bool,
    features: LayerShape,
    kind: LayerKind,
    name: String,
}

impl LayerConfig {
    pub fn new(biased: bool, features: LayerShape, kind: LayerKind, name: impl ToString) -> Self {
        Self {
            biased,
            features,
            kind,
            name: name.to_string(),
        }
    }

    pub fn is_biased(&self) -> bool {
        self.biased
    }

    pub fn features(&self) -> &LayerShape {
        &self.features
    }

    pub fn features_mut(&mut self) -> &mut LayerShape {
        &mut self.features
    }

    pub fn kind(&self) -> LayerKind {
        self.kind
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn set_biased(&mut self, biased: bool) {
        self.biased = biased;
    }

    pub fn set_features(&mut self, features: LayerShape) {
        self.features = features;
    }

    pub fn set_kind(&mut self, kind: LayerKind) {
        self.kind = kind;
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    pub fn biased(mut self) -> Self {
        self.biased = true;
        self
    }

    pub fn unbiased(mut self) -> Self {
        self.biased = false;
        self
    }

    pub fn with_features(mut self, features: LayerShape) -> Self {
        self.features = features;
        self
    }

    pub fn with_kind(mut self, kind: LayerKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }
}

impl From<LayerShape> for LayerConfig {
    fn from(features: LayerShape) -> Self {
        Self {
            biased: false,
            features,
            kind: LayerKind::default(),
            name: String::new(),
        }
    }
}
