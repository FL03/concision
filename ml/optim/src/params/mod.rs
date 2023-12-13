/*
   Appellation: params <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Parameters
//!
//! ## Overview
//!
pub use self::utils::*;

use num::Float;

pub trait Minimize<T> {
    fn minimize(&self, scale: T) -> Self;
}

pub trait Dampener<T = f64>
where
    T: Float,
{
    fn tau(&self) -> T; // Momentum Damper
}

pub trait Decay<T = f64>
where
    T: Float,
{
    fn lambda(&self) -> T; // Decay Rate
}

pub trait LearningRate<T = f64>
where
    T: Float,
{
    fn gamma(&self) -> T;
}

pub trait Momentum<T = f64>
where
    T: Float,
{
    fn mu(&self) -> T; // Momentum Rate

    fn nestrov(&self) -> bool;
}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
