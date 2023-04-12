/*
   Appellation: core <library>
   Contrib: FL03 <jo3mccain@icloud.com>
   Description: Implements the core functionality of the concision. Concision is an advanced data-science and machine-learning crate written in pure Rust and optimized for WASM environments.
*/
pub use self::{primitives::*, utils::*};

mod primitives;
mod utils;

pub mod linstep;
