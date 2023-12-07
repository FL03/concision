/*
    Appellation: modules <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Model
//!
use crate::prelude::Forward;
use ndarray::prelude::Array2;
use num::Float;

pub trait Module<T = f64>: Forward<Array2<T>, Output = Array2<T>>
where
    T: Float,
{
    type Config;

    fn add_module(&mut self, module: impl Module<T>);

    fn compile(&mut self);
    /// Returns a collection of all proceeding [Module]s in the network
    fn children(&self) -> &Vec<impl Module<T>>;

    fn children_mut(&mut self) -> &mut Vec<impl Module<T>>;
    /// Returns a collection of all [Module]s in the network
    fn modules(&self) -> Vec<&impl Module<T>>;

    fn name(&self) -> &str;
}