/*
    Appellation: config <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Config {
    pub heads: usize,
}

impl Config {
    pub fn new() -> ConfigBuilder {
        ConfigBuilder::new()
    }
    pub fn heads(&self) -> usize {
        self.heads
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            heads: crate::HEADS,
        }
    }
}

#[derive(Default)]
pub struct ConfigBuilder {
    heads: Option<usize>,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        Self { heads: None }
    }

    pub fn heads(mut self, heads: usize) -> Self {
        self.heads = Some(heads);
        self
    }

    pub fn build(&self) -> Config {
        Config {
            heads: self.heads.unwrap_or(crate::HEADS),
        }
    }
}
