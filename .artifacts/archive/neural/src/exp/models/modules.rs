/*
    Appellation: modules <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Model
//!
use crate::prelude::Predict;
use ndarray::prelude::Array2;
use std::collections::HashMap;

pub type ModuleParams<K, V> = HashMap<K, Array2<V>>;

pub trait Module<T = f64>: Predict<Array2<T>> {
    fn get_param(&self, name: &str) -> Option<&Array2<T>> {
        self.parameters().get(name)
    }

    fn name(&self) -> &str;

    fn parameters(&self) -> &HashMap<String, Array2<T>>;

    fn parameters_mut(&mut self) -> &mut HashMap<String, Array2<T>>;
}
