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

// pub struct M<T, O>(Box<dyn Module<T, Output = O>>);

pub trait Store<K, V> {
    fn get(&self, name: &str) -> Option<&V>;
    fn get_mut(&mut self, name: &str) -> Option<&mut V>;
    fn insert(&mut self, name: K, value: V) -> Option<V>;
    fn remove(&mut self, name: &str) -> Option<V>;
}

pub trait Module<T = f64>: Predict<Array2<T>> {
    fn get_param(&self, name: &str) -> Option<&Array2<T>> {
        self.parameters().get(name)
    }

    fn name(&self) -> &str;

    fn parameters(&self) -> &HashMap<String, Array2<T>>;

    fn parameters_mut(&mut self) -> &mut HashMap<String, Array2<T>>;
}
