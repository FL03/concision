/*
   Appellation: bias <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Bias
pub use self::{biases::*, mask::*, utils::*};

pub(crate) mod biases;
pub(crate) mod mask;

use num::Float;

pub trait Biased<T: Float = f64> {
    fn bias(&self) -> &Bias<T>;
    fn bias_mut(&mut self) -> &mut Bias<T>;
}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
