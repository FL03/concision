/*
    Appellation: config <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Config {
    pub d_model: usize,
    pub heads: usize,
}

impl Config {
    pub fn new() -> ConfigBuilder {
        ConfigBuilder::new()
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    pub fn dk(&self) -> usize {
        self.d_model() / self.heads()
    }

    pub fn heads(&self) -> usize {
        self.heads
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            d_model: crate::D_MODEL,
            heads: crate::HEADS,
        }
    }
}

concision::builder! {
    ConfigBuilder(Config) {
        d_model: usize,
        heads: usize,
    }
}