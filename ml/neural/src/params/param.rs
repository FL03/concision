/*
    Appellation: param <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Dimension, Ix2};
use num::Float;

pub struct ParamBuilder {
    name: String,
}

impl ParamBuilder {
    pub fn new() -> Self {
        Self {
            name: String::new(),
        }
    }

    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    pub fn build<T, D>(self, params: Array<T, D>) -> Param<T, D>
    where
        T: Float,
        D: Dimension,
    {
        Param::new(self.name, params)
    }
}

pub struct Param<T = f64, D = Ix2>
where
    T: Float,
    D: Dimension,
{
    name: String,
    params: Array<T, D>,
}

impl<T, D> Param<T, D>
where
    T: Float,
    D: Dimension,
{
    pub fn new(name: String, params: Array<T, D>) -> Self {
        Self { name, params }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn params(&self) -> &Array<T, D> {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut Array<T, D> {
        &mut self.params
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }
}
