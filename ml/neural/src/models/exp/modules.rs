/*
    Appellation: modules <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Model
//!
use crate::prelude::Forward;
use ndarray::prelude::Array2;
use num::Float;
use std::collections::HashMap;

pub type ModuleParams<K, V> = HashMap<K, Array2<V>>;

pub struct M<T: Float = f64>(Box<dyn Module<T, Output = Array2<T>>>);

pub trait Module<T = f64>: Forward<Array2<T>>
where
    T: Float,
{
    fn get_param(&self, name: &str) -> Option<&Array2<T>> {
        self.parameters().get(name)
    }

    fn name(&self) -> &str;

    fn parameters(&self) -> &ModuleParams<String, T>;

    fn parameters_mut(&mut self) -> &mut ModuleParams<String, T>;
}

pub trait ModuleExt<T = f64>: Module<T>
where
    T: Float,
{
}
