/*
   Appellation: const <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{epsilon::*, utils::*};

pub(crate) mod epsilon;

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
