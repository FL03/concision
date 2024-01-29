/*
   Appellation: flows <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Flows
pub use self::{direction::*, flow::*};

pub(crate) mod direction;
pub(crate) mod flow;

#[cfg(test)]
mod tests {}
