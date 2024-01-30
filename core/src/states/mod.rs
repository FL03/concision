/*
   Appellation: states <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::state::*;

pub(crate) mod state;

pub mod weighted;

#[cfg(test)]
mod tests {}
