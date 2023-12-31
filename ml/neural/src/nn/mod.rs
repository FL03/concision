/*
    Appellation: nn <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Neural Network
pub use self::{kinds::*, position::*, sequential::*, utils::*};

pub(crate) mod kinds;
pub(crate) mod position;
pub(crate) mod sequential;

pub mod cnn;
pub mod ffn;
pub mod gnn;
pub mod rnn;

pub(crate) mod utils {}
