/*
    Appellation: models <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Model
//!
pub use self::{model::*, params::*, utils::*};

pub(crate) mod model;
pub(crate) mod params;

pub mod stack;

use ndarray::prelude::Array2;

pub trait Module<T = f64> {
    fn add_module(&mut self, module: impl Module<T>);

    fn forward(&self, args: &Array2<T>) -> Array2<T>;
}

pub(crate) mod utils {}
