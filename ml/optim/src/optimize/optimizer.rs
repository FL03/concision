/*
    Appellation: optimizer <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neural::prelude::Params;
use ndarray::prelude::Dimension;

pub struct Optimizer {
    params: Vec<Box<dyn Params>>,
}

impl Optimizer {
    pub fn new() -> Self {
        Self { params: Vec::new() }
    }

    pub fn params(&self) -> &[Box<dyn Params>] {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut [Box<dyn Params>] {
        &mut self.params
    }

    pub fn set_params(&mut self, params: Vec<Box<dyn Params>>) {
        self.params = params;
    }

    pub fn with_params(mut self, params: Vec<Box<dyn Params>>) -> Self {
        self.params = params;
        self
    }
}
