/*
    Appellation: multi_head <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub struct Config {
    pub heads: usize,
}

pub struct MultiHeadAttention {
    pub(crate) config: Config,
}

impl MultiHeadAttention {
    pub const fn config(&self) -> &Config {
        &self.config
    }
}
