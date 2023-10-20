/*
    Appellation: regression <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # regression
//!
pub use self::utils::*;

pub mod linear;

pub trait Regression {
    type Item: num::Float;

    fn fit(&mut self, args: &[Self::Item], target: &[Self::Item]);

    fn predict(&self, args: &[Self::Item]) -> Vec<Self::Item>;
}

pub(crate) mod utils {}
