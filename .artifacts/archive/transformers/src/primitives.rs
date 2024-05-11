/*
    Appellation: primitives <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{constants::*, statics::*, types::*};

pub(crate) mod constants {
    /// The default dropout rate
    pub const DROPOUT: f64 = 0.1;
    /// The default number of heads in the multi-head attention layer
    pub const HEADS: usize = 8;
    /// The default dimension of the model (embedding size)
    pub const MODEL: usize = 512;
    /// The default number of parameters in the feed-forward network
    pub const NETWORK: usize = 2048;
    /// The default number of samples to draw from the attention distribution
    pub const SAMPLES: usize = 10000;
}

pub(crate) mod statics {
    use super::constants::*;
    use lazy_static::lazy_static;

    lazy_static! {
        /// The default dimensions of the query, key, and value tensors w/r/2 a single head
        pub static ref QUERY_SIZE: usize = MODEL / HEADS;
    }
}

pub(crate) mod types {
    /// The dimension of all inputs and outputs for each layer of the model (batch, seq, model)
    pub type BaseDim = ndarray::Ix3;
}
