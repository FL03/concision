/*
    Appellation: models <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Model
//!
pub use self::{features::*, model::*, params::*, utils::*};

pub(crate) mod features;
pub(crate) mod model;
pub(crate) mod params;

pub mod stack;

pub trait Module<T = f64> {
    fn add_module(&mut self, module: impl Module<T>);

    fn params(&self) -> &ModelParams<T>;

    fn params_mut(&mut self) -> &mut ModelParams<T>;
}

pub(crate) mod utils {}
