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

pub trait Module<T = f64>: Forward<Array2<T>, Output=Array2<T>> {
    fn add_module(&mut self, module: impl Module<T>);

    fn layers(&self) -> &[impl Module<T>];

    fn name(&self) -> &str;
}

pub(crate) mod utils {}
