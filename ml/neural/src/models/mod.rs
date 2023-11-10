/*
    Appellation: models <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Model
//!
pub use self::{model::*, utils::*};

pub(crate) mod model;

pub trait Module {
    fn add_module(&mut self, module: impl Module);
}

pub(crate) mod utils {}
