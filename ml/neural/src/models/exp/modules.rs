/*
    Appellation: modules <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Model
//!
use crate::prelude::Predict;
use ndarray::prelude::Array2;
use num::Float;
use std::collections::HashMap;

pub type ModuleParams<K, V> = HashMap<K, Array2<V>>;



pub trait Module<T = f64>: Predict<T>
where
    T: Float,
{
    fn name(&self) -> &str;

    fn parameters(&self) -> &ModuleParams<&str, T>;

    fn parameters_mut(&mut self) -> &mut ModuleParams<&str, T>;
}

