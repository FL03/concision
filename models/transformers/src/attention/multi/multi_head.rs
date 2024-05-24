/*
    Appellation: multi_head <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Config;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MultiHeadAttention {
    pub(crate) config: Config,
}

impl MultiHeadAttention {
    pub const fn config(&self) -> &Config {
        &self.config
    }
}
