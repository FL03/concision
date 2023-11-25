/*
    Appellation: models <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Model
//!
pub use self::{config::*, model::*, stack::*, utils::*};

pub(crate) mod config;
pub(crate) mod model;
pub(crate) mod stack;

use crate::prelude::Forward;
use ndarray::prelude::Array2;
use num::Float;

pub trait Module<T = f64>: Forward<Array2<T>, Output = Array2<T>>
where
    T: Float,
{
    fn add_module(&mut self, module: impl Module<T>);

    fn layers(&self) -> &[impl Forward<Array2<T>, Output = Array2<T>>];

    fn name(&self) -> &str;

    fn forward(&self, input: Array2<T>) -> Array2<T> {
        let mut output = input;
        for layer in self.layers() {
            output = layer.forward(&output);
        }
        output
    }
}

pub(crate) mod utils {}
