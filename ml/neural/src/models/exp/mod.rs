/*
    Appellation: exp <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Experimental Models
pub use self::{modules::*, store::*};

pub(crate) mod modules;
pub(crate) mod store;

use crate::prelude::Forward;
use ndarray::prelude::Array2;
use num::Float;

pub trait Model<T = f64>: Forward<Array2<T>>
where
    T: Float,
{
    type Config;

    fn name(&self) -> &str;

    fn modules(&self) -> &Vec<Box<dyn Module<T, Output = Self::Output>>>;

    fn modules_mut(&mut self) -> &mut Vec<Box<dyn Module<T, Output = Self::Output>>>;

    fn register_module(&mut self, module: Box<dyn Module<T, Output = Self::Output>>) -> &mut Self {
        self.modules_mut().push(module);
        self
    }

    fn get_module(&self, name: &str) -> Option<&Box<dyn Module<T, Output = Self::Output>>> {
        self.modules().iter().find(|m| m.name() == name)
    }
}

#[cfg(test)]
mod tests {}
