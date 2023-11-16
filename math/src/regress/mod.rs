/*
   Appellation: epochs <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{linear::*, utils::*};

pub(crate) mod linear;

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
