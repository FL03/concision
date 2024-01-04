/*
    Appellation: scan <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::params::SSMStore;

use num::Float;

pub struct Scanner<'a, T = f64>
where
    T: Float,
{
    model: &'a mut SSMStore<T>,
}

impl<'a, T> Scanner<'a, T>
where
    T: Float,
{
    pub fn new(model: &'a mut SSMStore<T>) -> Self {
        Self { model }
    }

    pub fn model(&self) -> &SSMStore<T> {
        self.model
    }

    pub fn model_mut(&mut self) -> &mut SSMStore<T> {
        self.model
    }
}
