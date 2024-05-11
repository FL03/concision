/*
   Appellation: transform <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Transform
pub use self::{config::*, transformer::*};

pub(crate) mod config;
pub(crate) mod transformer;

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_transformer() {}
}
