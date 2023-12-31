/*
   Appellation: store <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Store
pub use self::storage::*;

pub(crate) mod storage;

pub trait Store {}

#[cfg(test)]
mod tests {}
